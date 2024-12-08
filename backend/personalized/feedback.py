# backend/personalized/feedback.py

"""
CineBrain Feedback Processing & Online Learning
Real-time feedback integration and adaptive recommendation tuning
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import logging
import math
from typing import List, Dict, Any, Tuple, Optional
from scipy import stats
import pickle

from .utils import safe_json_loads, CacheManager, calculate_content_quality_score
from .metrics import RecommendationMetrics, PerformanceTracker

logger = logging.getLogger(__name__)

class ThompsonSamplingBandit:
    """Thompson Sampling for contextual bandits - explore/exploit trade-off"""
    
    def __init__(self, n_arms: int, context_dim: int = 10):
        self.n_arms = n_arms
        self.context_dim = context_dim
        
        # Beta distribution parameters for each arm
        self.alpha = np.ones(n_arms)  # Successes + 1
        self.beta = np.ones(n_arms)   # Failures + 1
        
        # Linear model parameters for contextual bandits
        self.theta = np.zeros((n_arms, context_dim))
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        
        self.total_rounds = 0
        self.arm_counts = np.zeros(n_arms)
    
    def select_arm(self, context: np.ndarray = None) -> int:
        """Select arm using Thompson Sampling"""
        if context is not None:
            # Contextual Thompson Sampling
            samples = []
            for i in range(self.n_arms):
                # Sample from posterior
                theta_sample = np.random.multivariate_normal(
                    self.theta[i], 
                    np.linalg.inv(self.A[i])
                )
                expected_reward = np.dot(context, theta_sample)
                samples.append(expected_reward)
            
            return np.argmax(samples)
        else:
            # Non-contextual Thompson Sampling
            samples = [np.random.beta(self.alpha[i], self.beta[i]) 
                      for i in range(self.n_arms)]
            return np.argmax(samples)
    
    def update(self, arm: int, reward: float, context: np.ndarray = None):
        """Update bandit parameters based on feedback"""
        self.total_rounds += 1
        self.arm_counts[arm] += 1
        
        if context is not None:
            # Update linear model
            self.A[arm] += np.outer(context, context)
            self.b[arm] += reward * context
            self.theta[arm] = np.linalg.solve(self.A[arm], self.b[arm])
        else:
            # Update Beta parameters
            if reward > 0.5:  # Binary reward
                self.alpha[arm] += 1
            else:
                self.beta[arm] += 1
    
    def get_arm_statistics(self) -> Dict[int, Dict[str, float]]:
        """Get statistics for each arm"""
        stats = {}
        for i in range(self.n_arms):
            stats[i] = {
                'expected_reward': self.alpha[i] / (self.alpha[i] + self.beta[i]),
                'pull_count': int(self.arm_counts[i]),
                'pull_ratio': self.arm_counts[i] / max(self.total_rounds, 1)
            }
        return stats

class UCBBandit:
    """Upper Confidence Bound algorithm for multi-armed bandits"""
    
    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = n_arms
        self.c = c  # Exploration parameter
        
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_rounds = 0
    
    def select_arm(self) -> int:
        """Select arm using UCB algorithm"""
        if self.total_rounds < self.n_arms:
            # Initial exploration
            return self.total_rounds
        
        ucb_values = []
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                ucb_values.append(float('inf'))
            else:
                average_reward = self.values[i] / self.counts[i]
                confidence_bound = self.c * np.sqrt(
                    np.log(self.total_rounds) / self.counts[i]
                )
                ucb_values.append(average_reward + confidence_bound)
        
        return np.argmax(ucb_values)
    
    def update(self, arm: int, reward: float):
        """Update arm statistics"""
        self.counts[arm] += 1
        self.values[arm] += reward
        self.total_rounds += 1

class FeedbackProcessor:
    """Process and analyze user feedback in real-time"""
    
    def __init__(self, cache_manager: CacheManager = None):
        self.cache = cache_manager or CacheManager()
        
        # Feedback type weights for learning
        self.feedback_weights = {
            'like': 1.0,
            'favorite': 1.5,
            'dislike': -1.0,
            'skip': -0.5,
            'view': 0.3,
            'complete_view': 0.7,
            'share': 2.0,
            'rate': 1.0,  # Actual rating value used
            'click': 0.1,
            'hover': 0.05
        }
        
        # Context importance weights
        self.context_weights = {
            'position_bias': 0.2,
            'time_of_day': 0.1,
            'device_type': 0.15,
            'session_length': 0.1,
            'recommendation_category': 0.3,
            'viewing_history': 0.15
        }
        
        # Store recent feedback for pattern analysis
        self.feedback_buffer = defaultdict(lambda: deque(maxlen=1000))
        
        # User-specific feedback patterns
        self.user_patterns = defaultdict(lambda: {
            'feedback_velocity': deque(maxlen=100),
            'preference_consistency': 0.5,
            'exploration_tendency': 0.5,
            'feedback_reliability': 1.0
        })
    
    def process_feedback(self, user_id: int, content_id: int,
                        feedback_type: str, feedback_value: Any = None,
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user feedback and extract learning signals"""
        
        try:
            # Validate feedback
            if feedback_type not in self.feedback_weights:
                logger.warning(f"Unknown feedback type: {feedback_type}")
                return {'success': False, 'error': 'Unknown feedback type'}
            
            # Calculate feedback signal
            if feedback_type == 'rate' and feedback_value is not None:
                # Normalize rating to [-1, 1]
                normalized_rating = (float(feedback_value) - 5.5) / 4.5
                feedback_signal = normalized_rating
            else:
                feedback_signal = self.feedback_weights[feedback_type]
            
            # Apply context adjustments
            context_multiplier = self._calculate_context_multiplier(context or {})
            adjusted_signal = feedback_signal * context_multiplier
            
            # Extract features for learning
            features = self._extract_feedback_features(
                user_id, content_id, feedback_type, context
            )
            
            # Store feedback
            feedback_entry = {
                'user_id': user_id,
                'content_id': content_id,
                'feedback_type': feedback_type,
                'feedback_value': feedback_value,
                'feedback_signal': feedback_signal,
                'adjusted_signal': adjusted_signal,
                'features': features,
                'context': context,
                'timestamp': datetime.utcnow()
            }
            
            self.feedback_buffer[user_id].append(feedback_entry)
            
            # Update user patterns
            self._update_user_patterns(user_id, feedback_entry)
            
            # Calculate immediate learning impact
            learning_impact = self._calculate_learning_impact(user_id, feedback_entry)
            
            # Cache the processed feedback
            cache_key = f"cinebrain:feedback:{user_id}:{content_id}:{feedback_type}"
            self.cache.set(cache_key, feedback_entry, ttl=3600)
            
            return {
                'success': True,
                'feedback_signal': adjusted_signal,
                'learning_impact': learning_impact,
                'features': features,
                'user_reliability': self.user_patterns[user_id]['feedback_reliability']
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_context_multiplier(self, context: Dict[str, Any]) -> float:
        """Calculate context-based adjustment multiplier"""
        multiplier = 1.0
        
        # Position bias adjustment
        if 'position_in_list' in context:
            position = context['position_in_list']
            # Reduce weight for items lower in the list
            position_factor = 1.0 / (1 + 0.1 * position)
            multiplier *= (1 - self.context_weights['position_bias'] + 
                          self.context_weights['position_bias'] * position_factor)
        
        # Time of day adjustment
        if 'timestamp' in context:
            hour = datetime.fromisoformat(context['timestamp']).hour
            # Peak hours (evening) get higher weight
            if 18 <= hour <= 22:
                time_factor = 1.2
            elif 6 <= hour <= 9:
                time_factor = 0.9
            else:
                time_factor = 1.0
            
            multiplier *= (1 - self.context_weights['time_of_day'] + 
                          self.context_weights['time_of_day'] * time_factor)
        
        # Device type adjustment
        if 'device_type' in context:
            device = context['device_type']
            if device == 'mobile':
                device_factor = 0.9  # Mobile interactions might be more casual
            elif device == 'tv':
                device_factor = 1.2  # TV viewing is more intentional
            else:
                device_factor = 1.0
            
            multiplier *= (1 - self.context_weights['device_type'] + 
                          self.context_weights['device_type'] * device_factor)
        
        # Viewing time adjustment
        if 'viewing_time' in context and context.get('recommendation_category'):
            viewing_time = context['viewing_time']
            if viewing_time > 120:  # More than 2 minutes
                time_factor = min(viewing_time / 300, 2.0)  # Cap at 2x
                multiplier *= time_factor
        
        return multiplier
    
    def _extract_feedback_features(self, user_id: int, content_id: int,
                                 feedback_type: str, 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from feedback for learning"""
        
        features = {
            'feedback_type': feedback_type,
            'user_id': user_id,
            'content_id': content_id,
            'hour_of_day': datetime.utcnow().hour,
            'day_of_week': datetime.utcnow().weekday(),
            'user_activity_level': self._get_user_activity_level(user_id),
            'content_age_days': self._get_content_age(content_id),
            'session_position': context.get('position_in_list', 0),
            'recommendation_category': context.get('recommendation_category', 'unknown'),
            'device_type': context.get('device_type', 'unknown'),
            'is_weekend': datetime.utcnow().weekday() >= 5
        }
        
        # Add user pattern features
        user_pattern = self.user_patterns[user_id]
        features.update({
            'user_exploration_tendency': user_pattern['exploration_tendency'],
            'user_preference_consistency': user_pattern['preference_consistency'],
            'user_feedback_reliability': user_pattern['feedback_reliability']
        })
        
        return features
    
    def _update_user_patterns(self, user_id: int, feedback_entry: Dict[str, Any]):
        """Update user behavior patterns based on feedback"""
        
        patterns = self.user_patterns[user_id]
        
        # Update feedback velocity
        patterns['feedback_velocity'].append(feedback_entry['timestamp'])
        
        # Calculate preference consistency
        recent_feedback = list(self.feedback_buffer[user_id])[-20:]
        if len(recent_feedback) >= 5:
            signals = [f['feedback_signal'] for f in recent_feedback]
            # Higher std = less consistent
            consistency = 1 - min(np.std(signals) / 2, 1)
            patterns['preference_consistency'] = 0.7 * patterns['preference_consistency'] + 0.3 * consistency
        
        # Update exploration tendency
        unique_content = len(set(f['content_id'] for f in self.feedback_buffer[user_id]))
        total_feedback = len(self.feedback_buffer[user_id])
        if total_feedback > 0:
            exploration = unique_content / total_feedback
            patterns['exploration_tendency'] = 0.8 * patterns['exploration_tendency'] + 0.2 * exploration
        
        # Update feedback reliability (based on consistency with popular opinion)
        # This would need aggregate data in production
        patterns['feedback_reliability'] = self._calculate_user_reliability(user_id)
    
    def _calculate_learning_impact(self, user_id: int, 
                                 feedback_entry: Dict[str, Any]) -> str:
        """Calculate the impact of this feedback on learning"""
        
        signal_strength = abs(feedback_entry['adjusted_signal'])
        user_reliability = self.user_patterns[user_id]['feedback_reliability']
        
        # Consider feedback novelty
        recent_content_ids = [f['content_id'] for f in list(self.feedback_buffer[user_id])[-50:]]
        is_novel = feedback_entry['content_id'] not in recent_content_ids[:-1]
        
        impact_score = signal_strength * user_reliability
        if is_novel:
            impact_score *= 1.5
        
        if impact_score > 1.5:
            return 'high'
        elif impact_score > 0.8:
            return 'medium'
        else:
            return 'low'
    
    def _get_user_activity_level(self, user_id: int) -> float:
        """Get user's recent activity level"""
        recent_feedback = [f for f in self.feedback_buffer[user_id]
                          if f['timestamp'] > datetime.utcnow() - timedelta(days=7)]
        
        # Normalize to 0-1 scale
        return min(len(recent_feedback) / 50, 1.0)
    
    def _get_content_age(self, content_id: int) -> int:
        """Get content age in days (would query DB in production)"""
        # Placeholder - in production, query content release date
        return 30
    
    def _calculate_user_reliability(self, user_id: int) -> float:
        """Calculate user's feedback reliability score"""
        # In production, compare with aggregate ratings
        # For now, use consistency as proxy
        return self.user_patterns[user_id]['preference_consistency']
    
    def get_user_feedback_summary(self, user_id: int) -> Dict[str, Any]:
        """Get summary of user's feedback patterns"""
        
        user_feedback = list(self.feedback_buffer[user_id])
        if not user_feedback:
            return {'status': 'no_feedback'}
        
        # Aggregate by feedback type
        feedback_counts = Counter(f['feedback_type'] for f in user_feedback)
        
        # Calculate sentiment
        total_signal = sum(f['adjusted_signal'] for f in user_feedback)
        avg_signal = total_signal / len(user_feedback)
        
        # Recent trends
        recent_feedback = [f for f in user_feedback 
                          if f['timestamp'] > datetime.utcnow() - timedelta(days=7)]
        recent_signal = sum(f['adjusted_signal'] for f in recent_feedback) / max(len(recent_feedback), 1)
        
        trend = 'improving' if recent_signal > avg_signal else 'declining'
        
        return {
            'total_feedback_count': len(user_feedback),
            'feedback_distribution': dict(feedback_counts),
            'average_sentiment': avg_signal,
            'recent_sentiment': recent_signal,
            'sentiment_trend': trend,
            'user_patterns': {
                'exploration_tendency': self.user_patterns[user_id]['exploration_tendency'],
                'preference_consistency': self.user_patterns[user_id]['preference_consistency'],
                'feedback_reliability': self.user_patterns[user_id]['feedback_reliability']
            },
            'most_common_feedback': feedback_counts.most_common(1)[0] if feedback_counts else None,
            'last_feedback_time': user_feedback[-1]['timestamp'].isoformat() if user_feedback else None
        }

class OnlineLearner:
    """Online learning system for real-time model updates"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        
        # User-specific model adjustments
        self.user_models = defaultdict(lambda: {
            'preference_weights': defaultdict(float),
            'content_embeddings': {},
            'algorithm_weights': {
                'collaborative': 0.35,
                'content_based': 0.25,
                'popularity': 0.15,
                'cinematic_dna': 0.15,
                'feedback': 0.10
            },
            'exploration_bonus': 0.1,
            'last_update': datetime.utcnow()
        })
        
        # Global model parameters
        self.global_weights = {
            'genre_importance': defaultdict(float),
            'language_importance': defaultdict(float),
            'quality_threshold_adjustment': 0.0,
            'diversity_preference': 0.3
        }
        
        # Multi-armed bandits for algorithm selection
        self.algorithm_bandits = defaultdict(
            lambda: ThompsonSamplingBandit(n_arms=5, context_dim=10)
        )
        
        # Feature importance tracking
        self.feature_importance = defaultdict(float)
    
    def update_user_model(self, user_id: int, feedback_data: Dict[str, Any]):
        """Update user-specific model based on feedback"""
        
        user_model = self.user_models[user_id]
        
        # Extract learning signal
        signal = feedback_data.get('feedback_signal', 0)
        content_id = feedback_data.get('content_id')
        features = feedback_data.get('features', {})
        
        # Update preference weights using gradient descent
        self._update_preference_weights(user_model, features, signal)
        
        # Update algorithm weights using bandits
        if 'recommendation_category' in features:
            context = self._extract_bandit_context(user_id, features)
            arm = self._category_to_arm(features['recommendation_category'])
            reward = (signal + 1) / 2  # Normalize to [0, 1]
            
            self.algorithm_bandits[user_id].update(arm, reward, context)
            
            # Update algorithm weights based on bandit statistics
            self._update_algorithm_weights(user_id)
        
        # Update exploration bonus
        self._update_exploration_bonus(user_id, feedback_data)
        
        # Mark update time
        user_model['last_update'] = datetime.utcnow()
        
        logger.debug(f"Updated model for user {user_id} with signal {signal}")
    
    def _update_preference_weights(self, user_model: Dict[str, Any],
                                 features: Dict[str, Any], signal: float):
        """Update preference weights using online gradient descent"""
        
        # Simple SGD update for preference weights
        for feature, value in features.items():
            if isinstance(value, (int, float)):
                # Calculate gradient (simplified)
                gradient = signal * value
                
                # Update weight with learning rate and momentum
                old_weight = user_model['preference_weights'][feature]
                new_weight = old_weight + self.learning_rate * gradient
                
                # Apply weight decay
                new_weight *= 0.99
                
                user_model['preference_weights'][feature] = new_weight
                
                # Update global feature importance
                self.feature_importance[feature] += abs(gradient)
    
    def _extract_bandit_context(self, user_id: int, 
                               features: Dict[str, Any]) -> np.ndarray:
        """Extract context vector for contextual bandits"""
        
        context = [
            features.get('hour_of_day', 12) / 24,
            features.get('day_of_week', 3) / 7,
            features.get('user_activity_level', 0.5),
            features.get('user_exploration_tendency', 0.5),
            features.get('user_preference_consistency', 0.5),
            features.get('session_position', 0) / 20,
            1.0 if features.get('is_weekend') else 0.0,
            features.get('content_age_days', 30) / 365,
            1.0 if features.get('device_type') == 'mobile' else 0.0,
            1.0 if features.get('device_type') == 'tv' else 0.0
        ]
        
        return np.array(context)
    
    def _category_to_arm(self, category: str) -> int:
        """Map recommendation category to bandit arm"""
        category_map = {
            'cinebrain_for_you': 0,
            'collaborative': 0,
            'content_based': 1,
            'popularity': 2,
            'cinematic_dna': 3,
            'hybrid': 4
        }
        return category_map.get(category, 4)
    
    def _update_algorithm_weights(self, user_id: int):
        """Update algorithm weights based on bandit performance"""
        
        stats = self.algorithm_bandits[user_id].get_arm_statistics()
        user_model = self.user_models[user_id]
        
        # Map arms back to algorithms
        arm_to_algo = {
            0: 'collaborative',
            1: 'content_based',
            2: 'popularity',
            3: 'cinematic_dna',
            4: 'feedback'
        }
        
        # Update weights based on expected rewards
        total_reward = sum(s['expected_reward'] for s in stats.values())
        
        if total_reward > 0:
            for arm, stat in stats.items():
                algo = arm_to_algo.get(arm, 'feedback')
                # Smooth update to prevent drastic changes
                new_weight = stat['expected_reward'] / total_reward
                old_weight = user_model['algorithm_weights'][algo]
                
                user_model['algorithm_weights'][algo] = (
                    0.7 * old_weight + 0.3 * new_weight
                )
        
        # Normalize weights
        total_weight = sum(user_model['algorithm_weights'].values())
        for algo in user_model['algorithm_weights']:
            user_model['algorithm_weights'][algo] /= total_weight
    
    def _update_exploration_bonus(self, user_id: int, feedback_data: Dict[str, Any]):
        """Update exploration bonus based on user behavior"""
        
        user_model = self.user_models[user_id]
        
        # Increase exploration if user gives consistent negative feedback
        signal = feedback_data.get('feedback_signal', 0)
        
        if signal < -0.5:
            # Negative feedback - increase exploration
            user_model['exploration_bonus'] = min(
                user_model['exploration_bonus'] * 1.1, 0.5
            )
        elif signal > 0.5:
            # Positive feedback - can reduce exploration
            user_model['exploration_bonus'] = max(
                user_model['exploration_bonus'] * 0.95, 0.05
            )
    
    def get_user_adjustments(self, user_id: int) -> Dict[int, float]:
        """Get content-specific adjustments for user"""
        
        user_model = self.user_models[user_id]
        adjustments = {}
        
        # Apply preference weights to adjust scores
        preference_weights = user_model['preference_weights']
        
        # This would be more sophisticated in production
        # For now, return algorithm weight adjustments
        return user_model['algorithm_weights'].copy()
    
    def get_personalized_weights(self, user_id: int) -> Dict[str, float]:
        """Get personalized algorithm weights for user"""
        return self.user_models[user_id]['algorithm_weights'].copy()
    
    def should_explore(self, user_id: int) -> bool:
        """Decide whether to explore (vs exploit) for this user"""
        
        exploration_bonus = self.user_models[user_id]['exploration_bonus']
        
        # Epsilon-greedy with personalized epsilon
        return np.random.random() < exploration_bonus
    
    def save_models(self, filepath: str):
        """Save learned models to disk"""
        try:
            model_data = {
                'user_models': dict(self.user_models),
                'global_weights': self.global_weights,
                'feature_importance': dict(self.feature_importance),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved online learning models to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load learned models from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.user_models.update(model_data['user_models'])
            self.global_weights.update(model_data['global_weights'])
            self.feature_importance.update(model_data['feature_importance'])
            
            logger.info(f"Loaded online learning models from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")

class ReinforcementLearner:
    """Deep reinforcement learning for recommendation optimization"""
    
    def __init__(self, state_dim: int = 50, action_dim: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Q-learning parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Simple neural network would go here
        # For now, use tabular Q-learning approximation
        self.q_table = defaultdict(lambda: np.zeros(action_dim))
        
        self.training_step = 0
    
    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        
        state_key = self._discretize_state(state)
        
        if explore and np.random.random() < self.epsilon:
            # Explore
            return np.random.randint(self.action_dim)
        else:
            # Exploit
            return np.argmax(self.q_table[state_key])
    
    def update(self, state: np.ndarray, action: int, 
              reward: float, next_state: np.ndarray, done: bool):
        """Update Q-values based on experience"""
        
        # Store experience
        self.replay_buffer.append((state, action, reward, next_state, done))
        
        # Sample batch for training
        if len(self.replay_buffer) > 32:
            self._train_on_batch(32)
        
        # Decay exploration
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def _train_on_batch(self, batch_size: int):
        """Train on a batch of experiences"""
        
        # Sample random batch
        batch_indices = np.random.choice(len(self.replay_buffer), batch_size)
        
        for idx in batch_indices:
            state, action, reward, next_state, done = self.replay_buffer[idx]
            
            state_key = self._discretize_state(state)
            next_state_key = self._discretize_state(next_state)
            
            # Q-learning update
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.q_table[next_state_key])
            
            # Update Q-value
            current_q = self.q_table[state_key][action]
            self.q_table[state_key][action] = current_q + self.learning_rate * (target - current_q)
        
        self.training_step += 1
    
    def _discretize_state(self, state: np.ndarray) -> str:
        """Discretize continuous state for tabular Q-learning"""
        # Simple binning for demonstration
        discretized = np.round(state * 10).astype(int)
        return str(discretized.tolist())
    
    @property
    def learning_rate(self) -> float:
        """Adaptive learning rate"""
        return max(0.01, 0.1 / (1 + self.training_step / 1000))

class AdaptiveFilterManager:
    """Manage adaptive filters based on user feedback"""
    
    def __init__(self):
        self.user_filters = defaultdict(lambda: {
            'quality_threshold': 7.0,
            'diversity_preference': 0.3,
            'language_strictness': 0.5,
            'genre_concentration': 0.7,
            'recency_bias': 0.5,
            'popularity_weight': 0.3
        })
        
        self.filter_bandits = defaultdict(
            lambda: UCBBandit(n_arms=5)  # 5 filter configurations
        )
    
    def get_filter_config(self, user_id: int) -> Dict[str, float]:
        """Get adaptive filter configuration for user"""
        
        # Select filter configuration using bandit
        config_idx = self.filter_bandits[user_id].select_arm()
        
        # Map to filter configurations
        configs = [
            {'quality_threshold': 6.0, 'diversity_preference': 0.5},  # Diverse
            {'quality_threshold': 7.5, 'diversity_preference': 0.2},  # Quality-focused
            {'quality_threshold': 7.0, 'diversity_preference': 0.3},  # Balanced
            {'quality_threshold': 6.5, 'diversity_preference': 0.4},  # Exploratory
            {'quality_threshold': 8.0, 'diversity_preference': 0.1},  # Conservative
        ]
        
        base_config = self.user_filters[user_id].copy()
        base_config.update(configs[config_idx])
        
        return base_config
    
    def update_filter_performance(self, user_id: int, 
                                config_idx: int, performance: float):
        """Update filter performance based on user satisfaction"""
        
        self.filter_bandits[user_id].update(config_idx, performance)
        
        # Also update base preferences
        if performance > 0.7:
            # Good performance - adjust base filters toward this config
            self._adjust_base_filters(user_id, config_idx, weight=0.1)
    
    def _adjust_base_filters(self, user_id: int, config_idx: int, weight: float):
        """Adjust base filter preferences"""
        
        configs = [
            {'quality_threshold': 6.0, 'diversity_preference': 0.5},
            {'quality_threshold': 7.5, 'diversity_preference': 0.2},
            {'quality_threshold': 7.0, 'diversity_preference': 0.3},
            {'quality_threshold': 6.5, 'diversity_preference': 0.4},
            {'quality_threshold': 8.0, 'diversity_preference': 0.1},
        ]
        
        selected_config = configs[config_idx]
        base_filters = self.user_filters[user_id]
        
        # Smooth update
        for key, value in selected_config.items():
            if key in base_filters:
                base_filters[key] = (1 - weight) * base_filters[key] + weight * value