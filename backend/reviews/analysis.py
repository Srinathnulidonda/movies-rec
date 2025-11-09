# reviews/analysis.py
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
from sqlalchemy import func, desc, asc

logger = logging.getLogger(__name__)

class ReviewAnalyticsService:
    def __init__(self, db, models, cache=None):
        self.db = db
        self.models = models
        self.cache = cache
        self.Content = models.get('Content')
        self.User = models.get('User')
        self.Review = models.get('Review')
        self.UserInteraction = models.get('UserInteraction')
        
        # Cache settings
        self.CACHE_TIMEOUT_STATS = 600    # 10 minutes
    
    def get_content_review_stats(self, content_id: int) -> Dict:
        """Get review statistics for content"""
        try:
            cache_key = f"review_stats:{content_id}"
            if self.cache:
                cached = self.cache.get(cache_key)
                if cached:
                    return cached
            
            reviews = self.Review.query.filter_by(
                content_id=content_id,
                is_approved=True
            ).all()
            
            if not reviews:
                stats = {
                    'total_reviews': 0,
                    'average_rating': 0,
                    'rating_distribution': {},
                    'recent_reviews': 0
                }
            else:
                total_reviews = len(reviews)
                average_rating = sum(r.rating for r in reviews) / total_reviews
                
                # Rating distribution
                rating_distribution = defaultdict(int)
                for review in reviews:
                    rating_key = int(review.rating)
                    rating_distribution[rating_key] += 1
                
                # Recent reviews (last 30 days)
                recent_date = datetime.utcnow() - timedelta(days=30)
                recent_reviews = sum(1 for r in reviews if r.created_at >= recent_date)
                
                stats = {
                    'total_reviews': total_reviews,
                    'average_rating': round(average_rating, 1),
                    'rating_distribution': dict(rating_distribution),
                    'recent_reviews': recent_reviews,
                    'reviews_with_text': sum(1 for r in reviews if r.review_text.strip()),
                    'spoiler_reviews': sum(1 for r in reviews if r.has_spoilers)
                }
            
            # Cache stats
            if self.cache:
                self.cache.set(cache_key, stats, timeout=self.CACHE_TIMEOUT_STATS)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting review stats: {e}")
            return {
                'total_reviews': 0,
                'average_rating': 0,
                'rating_distribution': {},
                'recent_reviews': 0
            }
    
    def get_user_review_stats(self, user_id: int) -> Dict:
        """Get review statistics for user"""
        try:
            reviews = self.Review.query.filter_by(user_id=user_id).all()
            approved_reviews = [r for r in reviews if r.is_approved]
            
            if not reviews:
                return {
                    'total_reviews': 0,
                    'approved_reviews': 0,
                    'pending_reviews': 0,
                    'average_rating_given': 0,
                    'total_helpful_votes': 0,
                    'reviews_with_text': 0,
                    'spoiler_reviews': 0
                }
            
            total_helpful = sum(r.helpful_count or 0 for r in approved_reviews)
            avg_rating = sum(r.rating for r in approved_reviews) / len(approved_reviews) if approved_reviews else 0
            
            return {
                'total_reviews': len(reviews),
                'approved_reviews': len(approved_reviews),
                'pending_reviews': len(reviews) - len(approved_reviews),
                'average_rating_given': round(avg_rating, 1),
                'total_helpful_votes': total_helpful,
                'reviews_with_text': sum(1 for r in approved_reviews if r.review_text.strip()),
                'spoiler_reviews': sum(1 for r in approved_reviews if r.has_spoilers),
                'rating_distribution': self._get_user_rating_distribution(approved_reviews)
            }
            
        except Exception as e:
            logger.error(f"Error getting user review stats: {e}")
            return {
                'total_reviews': 0,
                'approved_reviews': 0,
                'pending_reviews': 0,
                'average_rating_given': 0,
                'total_helpful_votes': 0,
                'reviews_with_text': 0,
                'spoiler_reviews': 0
            }
    
    def get_admin_review_stats(self) -> Dict:
        """Get admin review statistics"""
        try:
            cache_key = "admin_review_stats"
            if self.cache:
                cached = self.cache.get(cache_key)
                if cached:
                    return cached
            
            total_reviews = self.Review.query.count()
            pending_reviews = self.Review.query.filter_by(is_approved=False).count()
            approved_reviews = self.Review.query.filter_by(is_approved=True).count()
            
            # Recent activity (last 7 days)
            recent_date = datetime.utcnow() - timedelta(days=7)
            recent_reviews = self.Review.query.filter(
                self.Review.created_at >= recent_date
            ).count()
            
            # Average rating across all reviews
            avg_rating_result = self.db.session.query(
                func.avg(self.Review.rating)
            ).filter_by(is_approved=True).first()
            
            avg_rating = float(avg_rating_result[0]) if avg_rating_result[0] else 0
            
            # Most active users
            most_active_users = self.db.session.query(
                self.Review.user_id,
                func.count(self.Review.id).label('review_count')
            ).filter_by(is_approved=True).group_by(
                self.Review.user_id
            ).order_by(desc('review_count')).limit(5).all()
            
            # Most reviewed content
            most_reviewed_content = self.db.session.query(
                self.Review.content_id,
                func.count(self.Review.id).label('review_count')
            ).filter_by(is_approved=True).group_by(
                self.Review.content_id
            ).order_by(desc('review_count')).limit(5).all()
            
            stats = {
                'total_reviews': total_reviews,
                'approved_reviews': approved_reviews,
                'pending_reviews': pending_reviews,
                'recent_reviews': recent_reviews,
                'approval_rate': round((approved_reviews / total_reviews * 100), 1) if total_reviews > 0 else 0,
                'average_rating': round(avg_rating, 1),
                'most_active_users': [
                    {'user_id': user_id, 'review_count': count} 
                    for user_id, count in most_active_users
                ],
                'most_reviewed_content': [
                    {'content_id': content_id, 'review_count': count}
                    for content_id, count in most_reviewed_content
                ]
            }
            
            # Cache stats
            if self.cache:
                self.cache.set(cache_key, stats, timeout=self.CACHE_TIMEOUT_STATS)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting admin review stats: {e}")
            return {
                'total_reviews': 0,
                'approved_reviews': 0,
                'pending_reviews': 0,
                'recent_reviews': 0,
                'approval_rate': 0,
                'average_rating': 0
            }
    
    def get_content_rating_trends(self, content_id: int, days: int = 30) -> Dict:
        """Get rating trends for content over time"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            reviews = self.Review.query.filter(
                self.Review.content_id == content_id,
                self.Review.is_approved == True,
                self.Review.created_at >= start_date
            ).order_by(self.Review.created_at).all()
            
            if not reviews:
                return {'success': False, 'error': 'No reviews found for this period'}
            
            # Group by day
            daily_ratings = defaultdict(list)
            for review in reviews:
                day = review.created_at.date().isoformat()
                daily_ratings[day].append(review.rating)
            
            # Calculate daily averages
            trend_data = []
            for day, ratings in sorted(daily_ratings.items()):
                avg_rating = sum(ratings) / len(ratings)
                trend_data.append({
                    'date': day,
                    'average_rating': round(avg_rating, 1),
                    'review_count': len(ratings),
                    'ratings': ratings
                })
            
            return {
                'success': True,
                'content_id': content_id,
                'period_days': days,
                'trend_data': trend_data,
                'total_reviews': len(reviews),
                'overall_average': round(sum(r.rating for r in reviews) / len(reviews), 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting rating trends: {e}")
            return {'success': False, 'error': 'Failed to get rating trends'}
    
    def get_review_sentiment_analysis(self, content_id: Optional[int] = None) -> Dict:
        """Basic sentiment analysis of reviews"""
        try:
            query = self.Review.query.filter_by(is_approved=True)
            if content_id:
                query = query.filter_by(content_id=content_id)
            
            reviews = query.all()
            
            if not reviews:
                return {'success': False, 'error': 'No reviews found'}
            
            # Simple sentiment analysis based on ratings and keywords
            sentiment_data = {
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'total_analyzed': 0
            }
            
            positive_keywords = ['amazing', 'excellent', 'wonderful', 'fantastic', 'great', 'love', 'perfect']
            negative_keywords = ['terrible', 'awful', 'boring', 'waste', 'disappointing', 'bad', 'worst']
            
            for review in reviews:
                if not review.review_text.strip():
                    continue
                
                text = review.review_text.lower()
                rating = review.rating
                sentiment_data['total_analyzed'] += 1
                
                # Primary classification by rating
                if rating >= 7:
                    sentiment = 'positive'
                elif rating >= 4:
                    sentiment = 'neutral'
                else:
                    sentiment = 'negative'
                
                # Adjust based on keywords
                pos_count = sum(1 for word in positive_keywords if word in text)
                neg_count = sum(1 for word in negative_keywords if word in text)
                
                if pos_count > neg_count and pos_count > 0:
                    sentiment = 'positive'
                elif neg_count > pos_count and neg_count > 0:
                    sentiment = 'negative'
                
                sentiment_data[sentiment] += 1
            
            if sentiment_data['total_analyzed'] > 0:
                for key in ['positive', 'neutral', 'negative']:
                    sentiment_data[f'{key}_percentage'] = round(
                        (sentiment_data[key] / sentiment_data['total_analyzed']) * 100, 1
                    )
            
            return {
                'success': True,
                'content_id': content_id,
                'sentiment_data': sentiment_data
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'success': False, 'error': 'Failed to analyze sentiment'}
    
    def get_top_reviewers(self, limit: int = 10, period_days: Optional[int] = None) -> Dict:
        """Get top reviewers by activity and helpfulness"""
        try:
            query = self.db.session.query(
                self.Review.user_id,
                func.count(self.Review.id).label('review_count'),
                func.avg(self.Review.rating).label('avg_rating'),
                func.sum(self.Review.helpful_count).label('total_helpful')
            ).filter_by(is_approved=True)
            
            if period_days:
                start_date = datetime.utcnow() - timedelta(days=period_days)
                query = query.filter(self.Review.created_at >= start_date)
            
            top_reviewers_data = query.group_by(self.Review.user_id).order_by(
                desc('review_count')
            ).limit(limit).all()
            
            # Get user details
            top_reviewers = []
            for user_id, review_count, avg_rating, total_helpful in top_reviewers_data:
                user = self.User.query.get(user_id)
                if user:
                    top_reviewers.append({
                        'user_id': user_id,
                        'username': user.username,
                        'avatar_url': user.avatar_url,
                        'review_count': review_count,
                        'average_rating_given': round(float(avg_rating), 1) if avg_rating else 0,
                        'total_helpful_votes': int(total_helpful or 0),
                        'reviewer_score': self._calculate_reviewer_score(
                            review_count, float(avg_rating or 0), int(total_helpful or 0)
                        )
                    })
            
            return {
                'success': True,
                'top_reviewers': top_reviewers,
                'period_days': period_days,
                'total_active_reviewers': len(top_reviewers)
            }
            
        except Exception as e:
            logger.error(f"Error getting top reviewers: {e}")
            return {'success': False, 'error': 'Failed to get top reviewers'}
    
    # Private helper methods
    
    def _get_user_rating_distribution(self, reviews: List) -> Dict:
        """Get rating distribution for user's reviews"""
        distribution = defaultdict(int)
        for review in reviews:
            rating_key = int(review.rating)
            distribution[rating_key] += 1
        return dict(distribution)
    
    def _calculate_reviewer_score(self, review_count: int, avg_rating: float, helpful_votes: int) -> float:
        """Calculate a composite reviewer score"""
        # Weighted score based on activity, rating balance, and helpfulness
        activity_score = min(review_count / 10, 1.0)  # Max 1.0 for 10+ reviews
        balance_score = 1.0 - abs(avg_rating - 5.5) / 4.5  # Closer to middle ratings = more balanced
        helpful_score = min(helpful_votes / 20, 1.0)  # Max 1.0 for 20+ helpful votes
        
        # Weighted average
        score = (activity_score * 0.4 + balance_score * 0.3 + helpful_score * 0.3) * 10
        return round(score, 1)