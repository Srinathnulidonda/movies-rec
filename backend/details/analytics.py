"""
CineBrain Analytics Module
Analytics like popularity trends, user engagement metrics, etc.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy import func, desc, and_, or_
from collections import defaultdict, Counter
import json

logger = logging.getLogger(__name__)

class AnalyticsManager:
    """Manages analytics for the details module"""
    
    def __init__(self, db, models):
        self.db = db
        self.models = models
        self.Content = models.get('Content')
        self.UserInteraction = models.get('UserInteraction')
        self.AnonymousInteraction = models.get('AnonymousInteraction')
        self.Review = models.get('Review')
        
        logger.info("AnalyticsManager initialized successfully")
    
    def get_content_analytics(self, content_id: int) -> Dict:
        """Get comprehensive analytics for specific content"""
        try:
            content = self.Content.query.get(content_id)
            if not content:
                return {}
            
            analytics = {
                'basic_metrics': self._get_basic_metrics(content),
                'engagement_metrics': self._get_engagement_metrics(content_id),
                'trend_metrics': self._get_trend_metrics(content_id),
                'comparative_metrics': self._get_comparative_metrics(content),
                'demographic_insights': self._get_demographic_insights(content_id),
                'temporal_patterns': self._get_temporal_patterns(content_id)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting content analytics: {e}")
            return {}
    
    def get_popularity_trends(self, content_type: Optional[str] = None, 
                            days: int = 30) -> Dict:
        """Get popularity trends over time"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Build base query
            query = self.Content.query
            if content_type:
                query = query.filter(self.Content.content_type == content_type)
            
            # Get content with interactions in the time period
            if self.UserInteraction:
                trending_content = query.join(
                    self.UserInteraction,
                    self.Content.id == self.UserInteraction.content_id
                ).filter(
                    self.UserInteraction.timestamp >= start_date
                ).group_by(
                    self.Content.id
                ).order_by(
                    func.count(self.UserInteraction.id).desc()
                ).limit(50).all()
            else:
                # Fallback to popularity score
                trending_content = query.filter(
                    self.Content.popularity > 0
                ).order_by(
                    desc(self.Content.popularity)
                ).limit(50).all()
            
            trends = []
            for content in trending_content:
                trend_data = {
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'popularity_score': content.popularity or 0,
                    'rating': content.rating or 0,
                    'interaction_count': self._get_recent_interaction_count(content.id, days),
                    'trend_direction': self._calculate_trend_direction(content.id, days)
                }
                trends.append(trend_data)
            
            return {
                'trends': trends,
                'period': f"{days} days",
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting popularity trends: {e}")
            return {}
    
    def get_genre_analytics(self, content_type: Optional[str] = None) -> Dict:
        """Get analytics by genre"""
        try:
            query = self.Content.query
            if content_type:
                query = query.filter(self.Content.content_type == content_type)
            
            content_items = query.filter(
                self.Content.genres.isnot(None)
            ).all()
            
            genre_stats = defaultdict(lambda: {
                'count': 0,
                'avg_rating': 0,
                'total_interactions': 0,
                'popularity_sum': 0,
                'top_content': []
            })
            
            for content in content_items:
                try:
                    genres = json.loads(content.genres or '[]')
                    for genre in genres:
                        stats = genre_stats[genre]
                        stats['count'] += 1
                        stats['avg_rating'] += content.rating or 0
                        stats['popularity_sum'] += content.popularity or 0
                        
                        # Track top content for this genre
                        if len(stats['top_content']) < 5:
                            stats['top_content'].append({
                                'id': content.id,
                                'slug': content.slug,
                                'title': content.title,
                                'rating': content.rating or 0
                            })
                        else:
                            # Replace if this content has higher rating
                            min_rating_item = min(stats['top_content'], key=lambda x: x['rating'])
                            if (content.rating or 0) > min_rating_item['rating']:
                                stats['top_content'].remove(min_rating_item)
                                stats['top_content'].append({
                                    'id': content.id,
                                    'slug': content.slug,
                                    'title': content.title,
                                    'rating': content.rating or 0
                                })
                except (json.JSONDecodeError, TypeError):
                    continue
            
            # Calculate averages
            for genre, stats in genre_stats.items():
                if stats['count'] > 0:
                    stats['avg_rating'] = round(stats['avg_rating'] / stats['count'], 1)
                    stats['avg_popularity'] = round(stats['popularity_sum'] / stats['count'], 1)
                    stats['top_content'].sort(key=lambda x: x['rating'], reverse=True)
                del stats['popularity_sum']  # Remove intermediate calculation
            
            # Sort by count
            sorted_genres = dict(sorted(
                genre_stats.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            ))
            
            return {
                'genre_analytics': sorted_genres,
                'total_genres': len(sorted_genres),
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting genre analytics: {e}")
            return {}
    
    def get_performance_insights(self, content_id: int) -> Dict:
        """Get performance insights for content"""
        try:
            content = self.Content.query.get(content_id)
            if not content:
                return {}
            
            insights = {
                'performance_score': self._calculate_performance_score(content),
                'strengths': [],
                'areas_for_improvement': [],
                'benchmark_comparison': self._get_benchmark_comparison(content),
                'recommendations': []
            }
            
            # Analyze strengths and weaknesses
            rating = content.rating or 0
            popularity = content.popularity or 0
            vote_count = content.vote_count or 0
            
            if rating >= 8.0:
                insights['strengths'].append("High critical rating")
            elif rating <= 5.0:
                insights['areas_for_improvement'].append("Low critical rating")
            
            if popularity >= 50:
                insights['strengths'].append("High popularity score")
            elif popularity <= 10:
                insights['areas_for_improvement'].append("Low visibility/popularity")
            
            if vote_count >= 1000:
                insights['strengths'].append("Strong audience engagement")
            elif vote_count <= 100:
                insights['areas_for_improvement'].append("Limited audience feedback")
            
            # Generate recommendations
            if rating >= 7.5 and popularity < 30:
                insights['recommendations'].append("Consider marketing boost - high quality, low visibility")
            
            if popularity >= 40 and rating < 6.0:
                insights['recommendations'].append("Investigate quality issues - high visibility, low satisfaction")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting performance insights: {e}")
            return {}
    
    def _get_basic_metrics(self, content: Any) -> Dict:
        """Get basic content metrics"""
        try:
            return {
                'rating': content.rating or 0,
                'vote_count': content.vote_count or 0,
                'popularity': content.popularity or 0,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'content_age_days': (datetime.utcnow().date() - content.release_date).days if content.release_date else None,
                'is_trending': getattr(content, 'is_trending', False),
                'is_new_release': getattr(content, 'is_new_release', False),
                'is_critics_choice': getattr(content, 'is_critics_choice', False)
            }
        except Exception as e:
            logger.error(f"Error getting basic metrics: {e}")
            return {}
    
    def _get_engagement_metrics(self, content_id: int) -> Dict:
        """Get user engagement metrics"""
        try:
            engagement = {
                'total_interactions': 0,
                'unique_users': 0,
                'interaction_types': {},
                'recent_activity': 0,
                'engagement_rate': 0
            }
            
            if not self.UserInteraction:
                return engagement
            
            # Get all interactions for this content
            interactions = self.UserInteraction.query.filter_by(content_id=content_id).all()
            
            engagement['total_interactions'] = len(interactions)
            engagement['unique_users'] = len(set(i.user_id for i in interactions))
            
            # Count interaction types
            interaction_counts = Counter(i.interaction_type for i in interactions)
            engagement['interaction_types'] = dict(interaction_counts)
            
            # Recent activity (last 7 days)
            recent_date = datetime.utcnow() - timedelta(days=7)
            recent_interactions = [i for i in interactions if i.timestamp >= recent_date]
            engagement['recent_activity'] = len(recent_interactions)
            
            # Calculate engagement rate (interactions per unique user)
            if engagement['unique_users'] > 0:
                engagement['engagement_rate'] = round(
                    engagement['total_interactions'] / engagement['unique_users'], 2
                )
            
            return engagement
            
        except Exception as e:
            logger.error(f"Error getting engagement metrics: {e}")
            return {}
    
    def _get_trend_metrics(self, content_id: int) -> Dict:
        """Get trend metrics over time"""
        try:
            trends = {
                'weekly_trends': [],
                'monthly_trends': [],
                'peak_periods': [],
                'trend_direction': 'stable'
            }
            
            if not self.UserInteraction:
                return trends
            
            # Get interactions for the last 12 weeks
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(weeks=12)
            
            interactions = self.UserInteraction.query.filter(
                self.UserInteraction.content_id == content_id,
                self.UserInteraction.timestamp >= start_date
            ).all()
            
            # Group by week
            weekly_data = defaultdict(int)
            for interaction in interactions:
                week_start = interaction.timestamp - timedelta(days=interaction.timestamp.weekday())
                week_key = week_start.strftime('%Y-%W')
                weekly_data[week_key] += 1
            
            # Convert to list format
            for week, count in sorted(weekly_data.items()):
                trends['weekly_trends'].append({
                    'period': week,
                    'interaction_count': count
                })
            
            # Calculate trend direction
            if len(trends['weekly_trends']) >= 4:
                recent_avg = sum(w['interaction_count'] for w in trends['weekly_trends'][-2:]) / 2
                older_avg = sum(w['interaction_count'] for w in trends['weekly_trends'][-4:-2]) / 2
                
                if recent_avg > older_avg * 1.2:
                    trends['trend_direction'] = 'rising'
                elif recent_avg < older_avg * 0.8:
                    trends['trend_direction'] = 'declining'
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting trend metrics: {e}")
            return {}
    
    def _get_comparative_metrics(self, content: Any) -> Dict:
        """Get comparative metrics against similar content"""
        try:
            # Compare against similar content type and genre
            try:
                genres = json.loads(content.genres or '[]')
                primary_genre = genres[0] if genres else None
            except (json.JSONDecodeError, TypeError, IndexError):
                primary_genre = None
            
            comparison_query = self.Content.query.filter(
                self.Content.content_type == content.content_type,
                self.Content.id != content.id
            )
            
            if primary_genre:
                comparison_query = comparison_query.filter(
                    self.Content.genres.contains(primary_genre)
                )
            
            similar_content = comparison_query.limit(100).all()
            
            if not similar_content:
                return {}
            
            # Calculate percentiles
            ratings = [c.rating for c in similar_content if c.rating]
            popularities = [c.popularity for c in similar_content if c.popularity]
            vote_counts = [c.vote_count for c in similar_content if c.vote_count]
            
            comparison = {}
            
            if ratings:
                content_rating = content.rating or 0
                better_ratings = sum(1 for r in ratings if r < content_rating)
                comparison['rating_percentile'] = round((better_ratings / len(ratings)) * 100, 1)
            
            if popularities:
                content_popularity = content.popularity or 0
                better_popularity = sum(1 for p in popularities if p < content_popularity)
                comparison['popularity_percentile'] = round((better_popularity / len(popularities)) * 100, 1)
            
            if vote_counts:
                content_votes = content.vote_count or 0
                better_votes = sum(1 for v in vote_counts if v < content_votes)
                comparison['engagement_percentile'] = round((better_votes / len(vote_counts)) * 100, 1)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error getting comparative metrics: {e}")
            return {}
    
    def _get_demographic_insights(self, content_id: int) -> Dict:
        """Get demographic insights (placeholder for future implementation)"""
        try:
            # This would require user demographic data
            # For now, return placeholder structure
            return {
                'age_groups': {},
                'geographic_distribution': {},
                'platform_preferences': {},
                'viewing_patterns': {}
            }
        except Exception as e:
            logger.error(f"Error getting demographic insights: {e}")
            return {}
    
    def _get_temporal_patterns(self, content_id: int) -> Dict:
        """Get temporal interaction patterns"""
        try:
            patterns = {
                'peak_hours': [],
                'peak_days': [],
                'seasonal_trends': {},
                'activity_heatmap': {}
            }
            
            if not self.UserInteraction:
                return patterns
            
            # Get interactions from last 90 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=90)
            
            interactions = self.UserInteraction.query.filter(
                self.UserInteraction.content_id == content_id,
                self.UserInteraction.timestamp >= start_date
            ).all()
            
            if not interactions:
                return patterns
            
            # Analyze by hour of day
            hour_counts = defaultdict(int)
            day_counts = defaultdict(int)
            
            for interaction in interactions:
                hour_counts[interaction.timestamp.hour] += 1
                day_counts[interaction.timestamp.strftime('%A')] += 1
            
            # Find peak hours and days
            if hour_counts:
                peak_hour = max(hour_counts.items(), key=lambda x: x[1])
                patterns['peak_hours'] = [peak_hour[0]]
            
            if day_counts:
                peak_day = max(day_counts.items(), key=lambda x: x[1])
                patterns['peak_days'] = [peak_day[0]]
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting temporal patterns: {e}")
            return {}
    
    def _get_recent_interaction_count(self, content_id: int, days: int) -> int:
        """Get recent interaction count"""
        try:
            if not self.UserInteraction:
                return 0
            
            start_date = datetime.utcnow() - timedelta(days=days)
            count = self.UserInteraction.query.filter(
                self.UserInteraction.content_id == content_id,
                self.UserInteraction.timestamp >= start_date
            ).count()
            
            return count
            
        except Exception as e:
            logger.error(f"Error getting recent interaction count: {e}")
            return 0
    
    def _calculate_trend_direction(self, content_id: int, days: int) -> str:
        """Calculate trend direction"""
        try:
            if not self.UserInteraction:
                return 'stable'
            
            # Compare recent period vs previous period
            end_date = datetime.utcnow()
            mid_date = end_date - timedelta(days=days//2)
            start_date = end_date - timedelta(days=days)
            
            recent_count = self.UserInteraction.query.filter(
                self.UserInteraction.content_id == content_id,
                self.UserInteraction.timestamp >= mid_date
            ).count()
            
            previous_count = self.UserInteraction.query.filter(
                self.UserInteraction.content_id == content_id,
                self.UserInteraction.timestamp >= start_date,
                self.UserInteraction.timestamp < mid_date
            ).count()
            
            if recent_count > previous_count * 1.2:
                return 'rising'
            elif recent_count < previous_count * 0.8:
                return 'declining'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error calculating trend direction: {e}")
            return 'stable'
    
    def _calculate_performance_score(self, content: Any) -> float:
        """Calculate overall performance score"""
        try:
            score = 0.0
            
            # Rating component (40%)
            rating = content.rating or 0
            score += (rating / 10.0) * 0.4
            
            # Popularity component (30%)
            popularity = content.popularity or 0
            normalized_popularity = min(popularity / 100.0, 1.0)
            score += normalized_popularity * 0.3
            
            # Engagement component (30%)
            vote_count = content.vote_count or 0
            import math
            normalized_votes = min(math.log10(vote_count + 1) / 5.0, 1.0)
            score += normalized_votes * 0.3
            
            return round(score * 100, 1)  # Convert to 0-100 scale
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0
    
    def _get_benchmark_comparison(self, content: Any) -> Dict:
        """Get benchmark comparison"""
        try:
            # Simple benchmark against all content of same type
            same_type_content = self.Content.query.filter_by(
                content_type=content.content_type
            ).all()
            
            if not same_type_content:
                return {}
            
            avg_rating = sum(c.rating for c in same_type_content if c.rating) / len([c for c in same_type_content if c.rating])
            avg_popularity = sum(c.popularity for c in same_type_content if c.popularity) / len([c for c in same_type_content if c.popularity])
            
            return {
                'vs_average_rating': round((content.rating or 0) - avg_rating, 1),
                'vs_average_popularity': round((content.popularity or 0) - avg_popularity, 1),
                'category': content.content_type,
                'benchmark_size': len(same_type_content)
            }
            
        except Exception as e:
            logger.error(f"Error getting benchmark comparison: {e}")
            return {}