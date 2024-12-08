# backend/personalized/metric.py

"""
CineBrain Performance Metrics & Tracking
KPI monitoring, A/B testing, and recommendation quality measurement
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, ndcg_score
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import redis
from sqlalchemy import func, desc, and_

logger = logging.getLogger(__name__)

class RecommendationMetrics:
    """Calculate standard recommendation quality metrics"""
    
    @staticmethod
    def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Precision@K: What fraction of recommended items are relevant?
        """
        if k == 0 or not recommended:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        relevant_in_k = sum(1 for item in recommended_k if item in relevant_set)
        
        return relevant_in_k / min(k, len(recommended_k))
    
    @staticmethod
    def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Recall@K: What fraction of relevant items were recommended?
        """
        if not relevant or not recommended:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        recommended_set = set(recommended_k)
        
        relevant_found = len(relevant_set.intersection(recommended_set))
        
        return relevant_found / len(relevant_set)
    
    @staticmethod
    def ndcg_at_k(recommended: List[int], relevance_scores: Dict[int, float], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain@K
        Accounts for graded relevance and position bias
        """
        if k == 0 or not recommended:
            return 0.0
        
        recommended_k = recommended[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item_id in enumerate(recommended_k):
            rel = relevance_scores.get(item_id, 0)
            dcg += (2**rel - 1) / np.log2(i + 2)  # i+2 because positions start at 1
        
        # Calculate ideal DCG
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_scores):
            idcg += (2**rel - 1) / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def map_at_k(recommended_lists: List[List[int]], 
                 relevant_lists: List[List[int]], k: int) -> float:
        """
        Mean Average Precision@K across multiple queries
        """
        if not recommended_lists:
            return 0.0
        
        ap_scores = []
        
        for recommended, relevant in zip(recommended_lists, relevant_lists):
            if not relevant:
                continue
            
            relevant_set = set(relevant)
            num_relevant = 0
            precision_sum = 0
            
            for i, item_id in enumerate(recommended[:k]):
                if item_id in relevant_set:
                    num_relevant += 1
                    precision_sum += num_relevant / (i + 1)
            
            if num_relevant > 0:
                ap = precision_sum / min(len(relevant), k)
                ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    @staticmethod
    def diversity_score(recommended: List[Any], feature_extractor=None) -> float:
        """
        Intra-list diversity: How different are the recommended items?
        """
        if len(recommended) < 2:
            return 0.0
        
        if not feature_extractor:
            # Default: use genres as features
            def feature_extractor(item):
                if hasattr(item, 'genres'):
                    return set(json.loads(item.genres or '[]'))
                return set()
        
        diversity = 0.0
        comparisons = 0
        
        for i in range(len(recommended)):
            for j in range(i + 1, len(recommended)):
                features_i = feature_extractor(recommended[i])
                features_j = feature_extractor(recommended[j])
                
                if features_i or features_j:
                    # Jaccard distance
                    intersection = len(features_i.intersection(features_j))
                    union = len(features_i.union(features_j))
                    
                    if union > 0:
                        distance = 1 - (intersection / union)
                        diversity += distance
                        comparisons += 1
        
        return diversity / comparisons if comparisons > 0 else 0.0
    
    @staticmethod
    def novelty_score(recommended: List[int], 
                      item_popularity: Dict[int, float]) -> float:
        """
        Novelty: How surprising/non-obvious are the recommendations?
        """
        if not recommended or not item_popularity:
            return 0.0
        
        novelty_scores = []
        
        for item_id in recommended:
            popularity = item_popularity.get(item_id, 0.5)
            # Self-information: -log(p)
            if popularity > 0:
                novelty = -np.log2(popularity)
                novelty_scores.append(novelty)
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    @staticmethod
    def coverage(recommended_items_all_users: List[int], 
                 catalog_size: int) -> float:
        """
        Catalog coverage: What fraction of items can be recommended?
        """
        if catalog_size == 0:
            return 0.0
        
        unique_recommended = len(set(recommended_items_all_users))
        
        return unique_recommended / catalog_size
    
    @staticmethod
    def serendipity_score(recommended: List[Any], 
                          user_profile: Dict[str, Any],
                          relevance_threshold: float = 0.7) -> float:
        """
        Serendipity: Relevant but unexpected recommendations
        """
        if not recommended:
            return 0.0
        
        serendipity_scores = []
        
        user_genres = set(user_profile.get('genre_preferences', {}).get('top_genres', []))
        user_languages = set(user_profile.get('language_preferences', {}).get('preferred_languages', []))
        
        for item in recommended:
            # Check if relevant (based on rating or other metric)
            if hasattr(item, 'rating') and item.rating:
                is_relevant = item.rating >= relevance_threshold * 10
            else:
                is_relevant = True  # Assume relevant if no rating
            
            if is_relevant:
                # Check unexpectedness
                item_genres = set(json.loads(item.genres or '[]')) if hasattr(item, 'genres') else set()
                item_languages = set(json.loads(item.languages or '[]')) if hasattr(item, 'languages') else set()
                
                genre_overlap = len(user_genres.intersection(item_genres)) / max(len(user_genres), 1)
                lang_overlap = len(user_languages.intersection(item_languages)) / max(len(user_languages), 1)
                
                # Lower overlap = more unexpected
                unexpectedness = 1 - ((genre_overlap + lang_overlap) / 2)
                serendipity_scores.append(unexpectedness)
        
        return np.mean(serendipity_scores) if serendipity_scores else 0.0

class PerformanceTracker:
    """Track recommendation system performance over time"""
    
    def __init__(self, redis_client: redis.Redis = None):
        self.redis = redis_client
        self.metrics_buffer = defaultdict(list)
        self.user_metrics = defaultdict(lambda: defaultdict(list))
        
    def log_recommendation_served(self, user_id: int, 
                                categories: List[str],
                                recommendation_count: int):
        """Log when recommendations are served"""
        event = {
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat(),
            'categories': categories,
            'recommendation_count': recommendation_count,
            'event_type': 'recommendation_served'
        }
        
        self._store_event('recommendation_served', event)
        
        # Update user-specific metrics
        self.user_metrics[user_id]['recommendations_served'].append(event)
    
    def log_recommendation_interaction(self, user_id: int,
                                     content_id: int,
                                     interaction_type: str,
                                     position: int = None,
                                     recommendation_category: str = None):
        """Log user interaction with recommendation"""
        event = {
            'user_id': user_id,
            'content_id': content_id,
            'interaction_type': interaction_type,
            'position': position,
            'recommendation_category': recommendation_category,
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'recommendation_interaction'
        }
        
        self._store_event('recommendation_interaction', event)
        
        # Update CTR metrics
        if interaction_type == 'click':
            self.user_metrics[user_id]['clicks'].append(event)
        elif interaction_type == 'view':
            self.user_metrics[user_id]['views'].append(event)
    
    def log_feedback_received(self, user_id: int,
                            feedback_type: str,
                            content_id: int):
        """Log explicit user feedback"""
        event = {
            'user_id': user_id,
            'content_id': content_id,
            'feedback_type': feedback_type,
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'user_feedback'
        }
        
        self._store_event('user_feedback', event)
    
    def log_similarity_request(self, user_id: int,
                             base_content_id: int,
                             results_count: int):
        """Log similarity recommendation request"""
        event = {
            'user_id': user_id,
            'base_content_id': base_content_id,
            'results_count': results_count,
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'similarity_request'
        }
        
        self._store_event('similarity_request', event)
    
    def calculate_user_ctr(self, user_id: int, 
                          time_window: timedelta = timedelta(days=30)) -> float:
        """Calculate user's click-through rate"""
        cutoff_time = datetime.utcnow() - time_window
        
        served = [e for e in self.user_metrics[user_id]['recommendations_served']
                  if datetime.fromisoformat(e['timestamp']) > cutoff_time]
        
        clicks = [e for e in self.user_metrics[user_id]['clicks']
                  if datetime.fromisoformat(e['timestamp']) > cutoff_time]
        
        total_served = sum(e['recommendation_count'] for e in served)
        total_clicks = len(clicks)
        
        return total_clicks / total_served if total_served > 0 else 0.0
    
    def calculate_recommendation_accuracy(self, user_id: int) -> float:
        """Calculate recommendation accuracy based on user feedback"""
        feedback_events = [e for e in self.metrics_buffer.get('user_feedback', [])
                          if e['user_id'] == user_id]
        
        if not feedback_events:
            return 0.0
        
        positive_feedback = sum(1 for e in feedback_events 
                               if e['feedback_type'] in ['like', 'favorite', 'rate_high'])
        
        return positive_feedback / len(feedback_events)
    
    def get_user_metrics(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive metrics for a user"""
        
        # Calculate various metrics
        ctr = self.calculate_user_ctr(user_id)
        accuracy = self.calculate_recommendation_accuracy(user_id)
        
        # Engagement metrics
        total_interactions = sum(len(v) for v in self.user_metrics[user_id].values())
        recent_interactions = sum(
            1 for k, events in self.user_metrics[user_id].items()
            for e in events
            if datetime.fromisoformat(e['timestamp']) > datetime.utcnow() - timedelta(days=7)
        )
        
        # Category preferences
        category_interactions = defaultdict(int)
        for event in self.user_metrics[user_id].get('clicks', []):
            if event.get('recommendation_category'):
                category_interactions[event['recommendation_category']] += 1
        
        return {
            'user_id': user_id,
            'click_through_rate': round(ctr * 100, 2),
            'recommendation_accuracy': round(accuracy * 100, 2),
            'total_interactions': total_interactions,
            'recent_interactions_7d': recent_interactions,
            'preferred_categories': dict(category_interactions),
            'engagement_score': self._calculate_engagement_score(user_id),
            'metrics_period': {
                'start': (datetime.utcnow() - timedelta(days=30)).isoformat(),
                'end': datetime.utcnow().isoformat()
            }
        }
    
    def get_system_performance_summary(self) -> Dict[str, Any]:
        """Get overall system performance metrics"""
        
        # Aggregate metrics across all users
        all_users = set()
        for event_list in self.metrics_buffer.values():
            for event in event_list:
                if 'user_id' in event:
                    all_users.add(event['user_id'])
        
        total_users = len(all_users)
        
        # Calculate average metrics
        avg_ctr = np.mean([self.calculate_user_ctr(uid) for uid in all_users]) if all_users else 0
        avg_accuracy = np.mean([self.calculate_recommendation_accuracy(uid) for uid in all_users]) if all_users else 0
        
        # Event counts
        event_counts = {event_type: len(events) 
                       for event_type, events in self.metrics_buffer.items()}
        
        # Popular content
        content_interactions = Counter()
        for event in self.metrics_buffer.get('recommendation_interaction', []):
            content_interactions[event.get('content_id')] += 1
        
        return {
            'total_users': total_users,
            'avg_ctr': round(avg_ctr * 100, 2),
            'avg_accuracy': round(avg_accuracy * 100, 2),
            'avg_engagement': round(np.mean([self._calculate_engagement_score(uid) for uid in all_users]) if all_users else 0, 2),
            'event_counts': event_counts,
            'popular_content_ids': [cid for cid, _ in content_interactions.most_common(10)],
            'system_health': self._calculate_system_health(),
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def _store_event(self, event_type: str, event_data: Dict[str, Any]):
        """Store event in buffer and Redis if available"""
        self.metrics_buffer[event_type].append(event_data)
        
        # Keep buffer size manageable
        if len(self.metrics_buffer[event_type]) > 10000:
            self.metrics_buffer[event_type] = self.metrics_buffer[event_type][-5000:]
        
        # Store in Redis if available
        if self.redis:
            try:
                key = f"cinebrain:metrics:{event_type}:{datetime.utcnow().strftime('%Y%m%d')}"
                self.redis.lpush(key, json.dumps(event_data))
                self.redis.expire(key, 86400 * 30)  # Keep for 30 days
            except Exception as e:
                logger.warning(f"Failed to store metric in Redis: {e}")
    
    def _calculate_engagement_score(self, user_id: int) -> float:
        """Calculate user engagement score (0-100)"""
        
        # Factors for engagement
        interactions = sum(len(v) for v in self.user_metrics[user_id].values())
        recent_activity = sum(
            1 for k, events in self.user_metrics[user_id].items()
            for e in events
            if datetime.fromisoformat(e['timestamp']) > datetime.utcnow() - timedelta(days=7)
        )
        
        # Variety of interactions
        interaction_types = len([k for k, v in self.user_metrics[user_id].items() if v])
        
        # Calculate score
        score = 0.0
        
        # Activity volume (max 40 points)
        score += min(interactions / 100 * 40, 40)
        
        # Recent activity (max 30 points)
        score += min(recent_activity / 20 * 30, 30)
        
        # Interaction variety (max 30 points)
        score += (interaction_types / 5) * 30
        
        return min(score, 100)
    
    def _calculate_system_health(self) -> str:
        """Determine overall system health status"""
        
        # Check various health indicators
        recent_events = sum(
            1 for events in self.metrics_buffer.values()
            for e in events
            if datetime.fromisoformat(e['timestamp']) > datetime.utcnow() - timedelta(hours=1)
        )
        
        if recent_events < 10:
            return 'low_activity'
        elif recent_events < 100:
            return 'moderate'
        else:
            return 'healthy'

class ABTestManager:
    """Manage A/B tests for recommendation algorithms"""
    
    def __init__(self):
        self.active_tests = {}
        self.test_results = defaultdict(lambda: defaultdict(list))
        
    def create_test(self, test_name: str,
                   control_algorithm: str,
                   variant_algorithm: str,
                   traffic_split: float = 0.5,
                   min_sample_size: int = 1000):
        """Create a new A/B test"""
        
        self.active_tests[test_name] = {
            'control': control_algorithm,
            'variant': variant_algorithm,
            'traffic_split': traffic_split,
            'min_sample_size': min_sample_size,
            'created_at': datetime.utcnow(),
            'status': 'active',
            'users_assigned': {
                'control': set(),
                'variant': set()
            }
        }
        
        logger.info(f"Created A/B test: {test_name}")
    
    def assign_user_to_variant(self, test_name: str, user_id: int) -> str:
        """Assign user to test variant"""
        
        if test_name not in self.active_tests:
            return 'control'
        
        test = self.active_tests[test_name]
        
        # Check if already assigned
        if user_id in test['users_assigned']['control']:
            return 'control'
        elif user_id in test['users_assigned']['variant']:
            return 'variant'
        
        # Random assignment based on traffic split
        if np.random.random() < test['traffic_split']:
            variant = 'variant'
        else:
            variant = 'control'
        
        test['users_assigned'][variant].add(user_id)
        
        return variant
    
    def record_test_event(self, test_name: str,
                         user_id: int,
                         metric_name: str,
                         metric_value: float):
        """Record metric for A/B test"""
        
        if test_name not in self.active_tests:
            return
        
        # Determine user's variant
        variant = None
        for v in ['control', 'variant']:
            if user_id in self.active_tests[test_name]['users_assigned'][v]:
                variant = v
                break
        
        if variant:
            self.test_results[test_name][variant].append({
                'user_id': user_id,
                'metric_name': metric_name,
                'metric_value': metric_value,
                'timestamp': datetime.utcnow()
            })
    
    def analyze_test_results(self, test_name: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        
        if test_name not in self.active_tests:
            return {'error': 'Test not found'}
        
        test = self.active_tests[test_name]
        control_data = self.test_results[test_name]['control']
        variant_data = self.test_results[test_name]['variant']
        
        # Group metrics by type
        control_metrics = defaultdict(list)
        variant_metrics = defaultdict(list)
        
        for event in control_data:
            control_metrics[event['metric_name']].append(event['metric_value'])
        
        for event in variant_data:
            variant_metrics[event['metric_name']].append(event['metric_value'])
        
        # Calculate statistics for each metric
        results = {
            'test_name': test_name,
            'control_algorithm': test['control'],
            'variant_algorithm': test['variant'],
            'sample_sizes': {
                'control': len(test['users_assigned']['control']),
                'variant': len(test['users_assigned']['variant'])
            },
            'metrics': {}
        }
        
        for metric_name in set(control_metrics.keys()) | set(variant_metrics.keys()):
            control_values = control_metrics.get(metric_name, [])
            variant_values = variant_metrics.get(metric_name, [])
            
            if control_values and variant_values:
                # Calculate statistics
                control_mean = np.mean(control_values)
                variant_mean = np.mean(variant_values)
                
                # Simple t-test for significance
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(control_values, variant_values)
                
                # Calculate lift
                lift = ((variant_mean - control_mean) / control_mean * 100) if control_mean > 0 else 0
                
                results['metrics'][metric_name] = {
                    'control_mean': round(control_mean, 4),
                    'variant_mean': round(variant_mean, 4),
                    'lift_percentage': round(lift, 2),
                    'p_value': round(p_value, 4),
                    'is_significant': p_value < 0.05,
                    'control_samples': len(control_values),
                    'variant_samples': len(variant_values)
                }
        
        # Overall recommendation
        significant_improvements = sum(
            1 for m in results['metrics'].values()
            if m['is_significant'] and m['lift_percentage'] > 0
        )
        
        if significant_improvements > len(results['metrics']) / 2:
            results['recommendation'] = 'adopt_variant'
        else:
            results['recommendation'] = 'keep_control'
        
        return results
    
    def get_active_tests(self) -> List[Dict[str, Any]]:
        """Get list of active A/B tests"""
        
        active = []
        for test_name, test_data in self.active_tests.items():
            if test_data['status'] == 'active':
                active.append({
                    'test_name': test_name,
                    'control': test_data['control'],
                    'variant': test_data['variant'],
                    'traffic_split': test_data['traffic_split'],
                    'users_assigned': {
                        'control': len(test_data['users_assigned']['control']),
                        'variant': len(test_data['users_assigned']['variant'])
                    },
                    'created_at': test_data['created_at'].isoformat()
                })
        
        return active