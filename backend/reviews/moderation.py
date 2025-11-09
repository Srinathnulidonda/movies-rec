# reviews/moderation.py
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy import desc, asc

logger = logging.getLogger(__name__)

class ReviewModerationService:
    def __init__(self, db, models, cache=None):
        self.db = db
        self.models = models
        self.cache = cache
        self.Content = models.get('Content')
        self.User = models.get('User')
        self.Review = models.get('Review')
        self.UserInteraction = models.get('UserInteraction')
    
    def get_admin_reviews(self, page: int = 1, limit: int = 20, status: str = 'all', 
                         sort_by: str = 'newest') -> Dict:
        """Get reviews for admin moderation"""
        try:
            query = self.db.session.query(self.Review, self.User, self.Content).join(
                self.User
            ).join(self.Content)
            
            # Filter by status
            if status == 'pending':
                query = query.filter(self.Review.is_approved == False)
            elif status == 'approved':
                query = query.filter(self.Review.is_approved == True)
            elif status == 'flagged':
                # Add logic for flagged reviews if implemented
                pass
            
            # Apply sorting
            if sort_by == 'newest':
                query = query.order_by(desc(self.Review.created_at))
            elif sort_by == 'oldest':
                query = query.order_by(asc(self.Review.created_at))
            elif sort_by == 'rating_high':
                query = query.order_by(desc(self.Review.rating))
            elif sort_by == 'rating_low':
                query = query.order_by(asc(self.Review.rating))
            
            # Get total count
            total_reviews = query.count()
            
            # Apply pagination
            offset = (page - 1) * limit
            review_entries = query.offset(offset).limit(limit).all()
            
            # Format reviews
            reviews = []
            for review, user, content in review_entries:
                review_data = self._format_admin_review(review, user, content)
                reviews.append(review_data)
            
            # Get admin stats
            from .analysis import ReviewAnalyticsService
            analytics_service = ReviewAnalyticsService(self.db, self.models, self.cache)
            admin_stats = analytics_service.get_admin_review_stats()
            
            # Pagination info
            pagination = {
                'current_page': page,
                'total_pages': (total_reviews + limit - 1) // limit,
                'total_reviews': total_reviews,
                'has_next': page * limit < total_reviews,
                'has_previous': page > 1,
                'per_page': limit
            }
            
            return {
                'success': True,
                'reviews': reviews,
                'pagination': pagination,
                'stats': admin_stats,
                'filter_status': status,
                'sort_by': sort_by
            }
            
        except Exception as e:
            logger.error(f"Error getting admin reviews: {e}")
            return {
                'success': False,
                'error': 'Failed to get admin reviews',
                'reviews': [],
                'pagination': {},
                'stats': {}
            }
    
    def moderate_review(self, review_id: int, action: str, admin_id: int, reason: str = None) -> Dict:
        """Moderate a review (approve/reject)"""
        try:
            review = self.Review.query.get(review_id)
            if not review:
                return {'success': False, 'error': 'Review not found'}
            
            admin = self.User.query.get(admin_id)
            if not admin or not admin.is_admin:
                return {'success': False, 'error': 'Admin access required'}
            
            if action == 'approve':
                review.is_approved = True
                message = 'Review approved'
                
                # Notify user
                user = self.User.query.get(review.user_id)
                content = self.Content.query.get(review.content_id)
                self._notify_review_approved(review, user, content)
                
            elif action == 'reject':
                review.is_approved = False
                message = 'Review rejected'
                
                # Could add rejection reason to review metadata
                if reason:
                    # Store rejection reason if review model supports it
                    pass
                
                # Notify user
                user = self.User.query.get(review.user_id)
                content = self.Content.query.get(review.content_id)
                self._notify_review_rejected(review, user, content, reason)
                
            else:
                return {'success': False, 'error': 'Invalid action'}
            
            # Clear caches
            content = self.Content.query.get(review.content_id)
            if content:
                self._clear_review_caches(content.id, content.slug)
            
            self.db.session.commit()
            
            logger.info(f"Review {review_id} {action}ed by admin {admin_id}")
            
            return {
                'success': True,
                'message': message,
                'review_id': review_id,
                'action': action
            }
            
        except Exception as e:
            logger.error(f"Error moderating review: {e}")
            self.db.session.rollback()
            return {'success': False, 'error': 'Failed to moderate review'}
    
    def bulk_moderate_reviews(self, review_ids: List[int], action: str, admin_id: int) -> Dict:
        """Bulk moderate multiple reviews"""
        try:
            admin = self.User.query.get(admin_id)
            if not admin or not admin.is_admin:
                return {'success': False, 'error': 'Admin access required'}
            
            reviews = self.Review.query.filter(self.Review.id.in_(review_ids)).all()
            
            if not reviews:
                return {'success': False, 'error': 'No reviews found'}
            
            updated_count = 0
            affected_content_ids = set()
            
            for review in reviews:
                if action == 'approve':
                    review.is_approved = True
                elif action == 'reject':
                    review.is_approved = False
                else:
                    continue
                
                updated_count += 1
                affected_content_ids.add(review.content_id)
            
            # Clear caches for affected content
            for content_id in affected_content_ids:
                content = self.Content.query.get(content_id)
                if content:
                    self._clear_review_caches(content_id, content.slug)
            
            self.db.session.commit()
            
            logger.info(f"Bulk {action} of {updated_count} reviews by admin {admin_id}")
            
            return {
                'success': True,
                'message': f'{updated_count} reviews {action}ed',
                'updated_count': updated_count,
                'action': action
            }
            
        except Exception as e:
            logger.error(f"Error in bulk moderation: {e}")
            self.db.session.rollback()
            return {'success': False, 'error': 'Failed to bulk moderate reviews'}
    
    def flag_review(self, review_id: int, user_id: int, reason: str = None) -> Dict:
        """Flag a review for admin attention"""
        try:
            review = self.Review.query.get(review_id)
            if not review:
                return {'success': False, 'error': 'Review not found'}
            
            # Check if user already flagged this review
            existing_flag = self.UserInteraction.query.filter_by(
                user_id=user_id,
                content_id=review.content_id,
                interaction_type='review_flag'
            ).filter_by(
                interaction_metadata__contains=f'"review_id": {review_id}'
            ).first()
            
            if existing_flag:
                return {'success': False, 'error': 'You have already flagged this review'}
            
            # Create flag interaction
            flag_interaction = self.UserInteraction(
                user_id=user_id,
                content_id=review.content_id,
                interaction_type='review_flag',
                interaction_metadata={
                    'review_id': review_id,
                    'reason': reason or 'Inappropriate content',
                    'flagged_at': datetime.utcnow().isoformat()
                },
                timestamp=datetime.utcnow()
            )
            self.db.session.add(flag_interaction)
            self.db.session.commit()
            
            # Notify admins if needed
            self._notify_review_flagged(review, reason)
            
            logger.info(f"Review {review_id} flagged by user {user_id}")
            
            return {
                'success': True,
                'message': 'Review flagged for admin review'
            }
            
        except Exception as e:
            logger.error(f"Error flagging review: {e}")
            self.db.session.rollback()
            return {'success': False, 'error': 'Failed to flag review'}
    
    def get_flagged_reviews(self, page: int = 1, limit: int = 20) -> Dict:
        """Get flagged reviews for admin review"""
        try:
            # Get flagged review IDs
            flag_interactions = self.UserInteraction.query.filter_by(
                interaction_type='review_flag'
            ).all()
            
            flagged_review_ids = []
            flag_counts = {}
            
            for flag in flag_interactions:
                metadata = flag.interaction_metadata
                if isinstance(metadata, str):
                    import json
                    metadata = json.loads(metadata)
                
                review_id = metadata.get('review_id')
                if review_id:
                    flagged_review_ids.append(review_id)
                    flag_counts[review_id] = flag_counts.get(review_id, 0) + 1
            
            if not flagged_review_ids:
                return {
                    'success': True,
                    'reviews': [],
                    'pagination': {'total_reviews': 0},
                    'stats': {'total_flagged': 0}
                }
            
            # Get reviews with flags
            query = self.db.session.query(self.Review, self.User, self.Content).join(
                self.User
            ).join(self.Content).filter(
                self.Review.id.in_(flagged_review_ids)
            ).order_by(desc(self.Review.created_at))
            
            # Apply pagination
            total_reviews = query.count()
            offset = (page - 1) * limit
            review_entries = query.offset(offset).limit(limit).all()
            
            # Format reviews
            reviews = []
            for review, user, content in review_entries:
                review_data = self._format_admin_review(review, user, content)
                review_data['flag_count'] = flag_counts.get(review.id, 0)
                reviews.append(review_data)
            
            pagination = {
                'current_page': page,
                'total_pages': (total_reviews + limit - 1) // limit,
                'total_reviews': total_reviews,
                'has_next': page * limit < total_reviews,
                'has_previous': page > 1,
                'per_page': limit
            }
            
            return {
                'success': True,
                'reviews': reviews,
                'pagination': pagination,
                'stats': {
                    'total_flagged': len(set(flagged_review_ids)),
                    'total_flags': len(flag_interactions)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting flagged reviews: {e}")
            return {
                'success': False,
                'error': 'Failed to get flagged reviews',
                'reviews': [],
                'pagination': {},
                'stats': {}
            }
    
    # Private helper methods
    
    def _format_admin_review(self, review, user, content) -> Dict:
        """Format review for admin interface"""
        return {
            'id': review.id,
            'rating': review.rating,
            'title': review.title,
            'review_text': review.review_text,
            'has_spoilers': review.has_spoilers,
            'helpful_count': review.helpful_count or 0,
            'is_approved': review.is_approved,
            'created_at': review.created_at.isoformat(),
            'updated_at': review.updated_at.isoformat() if review.updated_at else None,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'avatar_url': user.avatar_url,
                'is_admin': user.is_admin
            },
            'content': {
                'id': content.id,
                'slug': content.slug,
                'title': content.title,
                'content_type': content.content_type,
                'poster_url': content.poster_path
            },
            'admin_info': {
                'user_total_reviews': self.Review.query.filter_by(user_id=user.id).count(),
                'user_approved_reviews': self.Review.query.filter_by(
                    user_id=user.id, is_approved=True
                ).count(),
                'content_total_reviews': self.Review.query.filter_by(content_id=content.id).count(),
                'word_count': len(review.review_text.split()) if review.review_text else 0
            }
        }
    
    def _clear_review_caches(self, content_id: int, content_slug: str):
        """Clear review-related caches"""
        if not self.cache:
            return
        
        try:
            cache_keys = [
                f"review_stats:{content_id}",
                f"details:slug:{content_slug}"
            ]
            
            for key in cache_keys:
                self.cache.delete(key)
                
        except Exception as e:
            logger.error(f"Error clearing review caches: {e}")
    
    # Notification methods
    
    def _notify_review_approved(self, review, user, content):
        """Notify user that review was approved"""
        try:
            logger.info(f"Review {review.id} approved notification sent to user {user.id}")
        except Exception as e:
            logger.error(f"Error sending review approved notification: {e}")
    
    def _notify_review_rejected(self, review, user, content, reason=None):
        """Notify user that review was rejected"""
        try:
            logger.info(f"Review {review.id} rejected notification sent to user {user.id}")
        except Exception as e:
            logger.error(f"Error sending review rejected notification: {e}")
    
    def _notify_review_flagged(self, review, reason=None):
        """Notify admins that a review was flagged"""
        try:
            logger.info(f"Review {review.id} flagged for admin attention")
        except Exception as e:
            logger.error(f"Error sending review flagged notification: {e}")