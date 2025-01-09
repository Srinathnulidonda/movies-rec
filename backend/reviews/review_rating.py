# reviews/review_rating.py
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from functools import wraps
from sqlalchemy import and_, or_, func, desc, text, asc
from sqlalchemy.orm import Session
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

@dataclass
class ReviewData:
    id: int
    content_id: int
    user_id: int
    rating: float
    title: Optional[str]
    review_text: str
    has_spoilers: bool
    helpful_count: int
    is_approved: bool
    created_at: str
    updated_at: Optional[str]
    user: Dict
    content: Dict
    
    def to_dict(self):
        return asdict(self)

class ReviewService:
    def __init__(self, db, models, cache=None):
        self.db = db
        self.models = models
        self.cache = cache
        self.Content = models.get('Content')
        self.User = models.get('User')
        self.Review = models.get('Review')
        self.UserInteraction = models.get('UserInteraction')
        
        # Review validation settings
        self.MIN_REVIEW_LENGTH = 10
        self.MAX_REVIEW_LENGTH = 2000
        self.MAX_TITLE_LENGTH = 100
        self.MIN_RATING = 1
        self.MAX_RATING = 10
        
        # Auto-approval settings
        self.AUTO_APPROVE_ADMIN = True
        self.AUTO_APPROVE_TRUSTED_USERS = True
        self.TRUSTED_USER_REVIEW_COUNT = 1
        self.AUTO_APPROVE_MIN_LENGTH = 20
        
        # Cache settings
        self.CACHE_TIMEOUT_REVIEWS = 300  # 5 minutes
        self.CACHE_TIMEOUT_STATS = 600    # 10 minutes
    
    def get_content_reviews(self, content_slug: str, page: int = 1, limit: int = 10, 
                          sort_by: str = 'newest', user_id: Optional[int] = None) -> Dict:
        """Get paginated reviews for content"""
        try:
            # Get content by slug
            content = self._get_content_by_slug(content_slug)
            if not content:
                return {
                    'success': False,
                    'error': 'Content not found',
                    'reviews': [],
                    'pagination': {},
                    'stats': {}
                }
            
            # Check cache
            cache_key = f"reviews:{content.id}:{page}:{limit}:{sort_by}"
            if self.cache and not user_id:
                cached = self.cache.get(cache_key)
                if cached:
                    return cached
            
            # Build query
            query = self.db.session.query(self.Review, self.User).join(
                self.User
            ).filter(
                self.Review.content_id == content.id,
                self.Review.is_approved == True
            )
            
            # Apply sorting
            if sort_by == 'newest':
                query = query.order_by(desc(self.Review.created_at))
            elif sort_by == 'oldest':
                query = query.order_by(asc(self.Review.created_at))
            elif sort_by == 'rating_high':
                query = query.order_by(desc(self.Review.rating), desc(self.Review.created_at))
            elif sort_by == 'rating_low':
                query = query.order_by(asc(self.Review.rating), desc(self.Review.created_at))
            elif sort_by == 'helpful':
                query = query.order_by(desc(self.Review.helpful_count), desc(self.Review.created_at))
            else:
                query = query.order_by(desc(self.Review.helpful_count), desc(self.Review.created_at))
            
            # Get total count
            total_reviews = query.count()
            
            # Apply pagination
            offset = (page - 1) * limit
            review_entries = query.offset(offset).limit(limit).all()
            
            # Format reviews
            reviews = []
            for review, user in review_entries:
                review_data = self._format_review(review, user, content, user_id)
                reviews.append(review_data)
            
            # Get review stats
            from .analysis import ReviewAnalyticsService
            analytics_service = ReviewAnalyticsService(self.db, self.models, self.cache)
            stats = analytics_service.get_content_review_stats(content.id)
            
            # Pagination info
            pagination = {
                'current_page': page,
                'total_pages': (total_reviews + limit - 1) // limit,
                'total_reviews': total_reviews,
                'has_next': page * limit < total_reviews,
                'has_previous': page > 1,
                'per_page': limit
            }
            
            result = {
                'success': True,
                'reviews': reviews,
                'pagination': pagination,
                'stats': stats,
                'sort_by': sort_by,
                'content': {
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type
                }
            }
            
            # Cache result (without user-specific data)
            if self.cache and not user_id:
                self.cache.set(cache_key, result, timeout=self.CACHE_TIMEOUT_REVIEWS)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting reviews for content {content_slug}: {e}")
            return {
                'success': False,
                'error': 'Failed to get reviews',
                'reviews': [],
                'pagination': {},
                'stats': {}
            }
    
    def submit_review(self, content_slug: str, user_id: int, review_data: Dict) -> Dict:
        """Submit a new review"""
        try:
            # Get content
            content = self._get_content_by_slug(content_slug)
            if not content:
                return {'success': False, 'error': 'Content not found'}
            
            # Get user
            user = self.User.query.get(user_id)
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            # Check if user already reviewed this content
            existing_review = self.Review.query.filter_by(
                content_id=content.id,
                user_id=user_id
            ).first()
            
            if existing_review:
                return {'success': False, 'error': 'You have already reviewed this content'}
            
            # Validate review data
            validation_result = self._validate_review_data(review_data)
            if not validation_result['valid']:
                return {'success': False, 'error': validation_result['error']}
            
            # Auto-approval logic
            should_auto_approve = self._should_auto_approve_review(user, review_data)
            
            # Create review
            review = self.Review(
                content_id=content.id,
                user_id=user_id,
                rating=float(review_data['rating']),
                title=review_data.get('title', '').strip() or None,
                review_text=review_data['review_text'].strip(),
                has_spoilers=review_data.get('has_spoilers', False),
                is_approved=should_auto_approve,
                helpful_count=0,
                created_at=datetime.utcnow()
            )
            
            self.db.session.add(review)
            self.db.session.flush()  # Get the review ID
            
            # Record user interaction
            self._record_review_interaction(user_id, content.id, float(review_data['rating']))
            
            # Clear caches
            self._clear_review_caches(content.id, content.slug)
            
            self.db.session.commit()
            
            # Send notifications if auto-approved
            if should_auto_approve:
                self._notify_review_published(review, user, content)
            else:
                self._notify_review_pending(review, user, content)
            
            message = 'Review published successfully!' if should_auto_approve else 'Review submitted for moderation'
            
            logger.info(f"Review submitted by user {user_id} for content {content.id}, auto-approved: {should_auto_approve}")
            
            return {
                'success': True,
                'review_id': review.id,
                'message': message,
                'auto_approved': should_auto_approve,
                'review': self._format_review(review, user, content, user_id) if should_auto_approve else None
            }
            
        except Exception as e:
            logger.error(f"Error submitting review: {e}")
            self.db.session.rollback()
            return {'success': False, 'error': 'Failed to submit review'}
    
    def update_review(self, review_id: int, user_id: int, review_data: Dict) -> Dict:
        """Update an existing review"""
        try:
            # Get review
            review = self.Review.query.get(review_id)
            if not review:
                return {'success': False, 'error': 'Review not found'}
            
            # Check ownership
            if review.user_id != user_id:
                return {'success': False, 'error': 'You can only edit your own reviews'}
            
            # Validate review data
            validation_result = self._validate_review_data(review_data)
            if not validation_result['valid']:
                return {'success': False, 'error': validation_result['error']}
            
            # Get user and content for auto-approval
            user = self.User.query.get(user_id)
            content = self.Content.query.get(review.content_id)
            
            # Check if review needs re-approval after significant changes
            needs_reapproval = self._needs_reapproval(review, review_data)
            should_auto_approve = self._should_auto_approve_review(user, review_data)
            
            # Update review
            review.rating = float(review_data['rating'])
            review.title = review_data.get('title', '').strip() or None
            review.review_text = review_data['review_text'].strip()
            review.has_spoilers = review_data.get('has_spoilers', False)
            review.updated_at = datetime.utcnow()
            
            if needs_reapproval:
                review.is_approved = should_auto_approve
            
            # Update user interaction
            self._record_review_interaction(user_id, content.id, float(review_data['rating']))
            
            # Clear caches
            self._clear_review_caches(content.id, content.slug)
            
            self.db.session.commit()
            
            message = 'Review updated successfully!'
            if needs_reapproval and not should_auto_approve:
                message = 'Review updated and submitted for moderation'
            
            logger.info(f"Review {review_id} updated by user {user_id}")
            
            return {
                'success': True,
                'review_id': review.id,
                'message': message,
                'needs_approval': needs_reapproval and not should_auto_approve,
                'review': self._format_review(review, user, content, user_id)
            }
            
        except Exception as e:
            logger.error(f"Error updating review: {e}")
            self.db.session.rollback()
            return {'success': False, 'error': 'Failed to update review'}
    
    def delete_review(self, review_id: int, user_id: int) -> Dict:
        """Delete a review"""
        try:
            # Get review
            review = self.Review.query.get(review_id)
            if not review:
                return {'success': False, 'error': 'Review not found'}
            
            # Check ownership or admin privileges
            user = self.User.query.get(user_id)
            if review.user_id != user_id and not (user and user.is_admin):
                return {'success': False, 'error': 'You can only delete your own reviews'}
            
            content = self.Content.query.get(review.content_id)
            
            # Delete review
            self.db.session.delete(review)
            
            # Remove user interaction
            interaction = self.UserInteraction.query.filter_by(
                user_id=review.user_id,
                content_id=review.content_id,
                interaction_type='rating'
            ).first()
            if interaction:
                self.db.session.delete(interaction)
            
            # Clear caches
            if content:
                self._clear_review_caches(content.id, content.slug)
            
            self.db.session.commit()
            
            logger.info(f"Review {review_id} deleted by user {user_id}")
            
            return {
                'success': True,
                'message': 'Review deleted successfully'
            }
            
        except Exception as e:
            logger.error(f"Error deleting review: {e}")
            self.db.session.rollback()
            return {'success': False, 'error': 'Failed to delete review'}
    
    def vote_helpful(self, review_id: int, user_id: int, is_helpful: bool = True) -> Dict:
        """Vote on review helpfulness"""
        try:
            # Get review
            review = self.Review.query.get(review_id)
            if not review:
                return {'success': False, 'error': 'Review not found'}
            
            # Check if user is voting on their own review
            if review.user_id == user_id:
                return {'success': False, 'error': 'You cannot vote on your own review'}
            
            # Check if user already voted (using UserInteraction)
            existing_vote = self.UserInteraction.query.filter_by(
                user_id=user_id,
                content_id=review.content_id,
                interaction_type='review_vote'
            ).filter(
                text(f"JSON_EXTRACT(interaction_metadata, '$.review_id') = {review_id}")
            ).first()
            
            if existing_vote:
                # Update existing vote
                metadata = json.loads(existing_vote.interaction_metadata or '{}')
                old_helpful = metadata.get('is_helpful', True)
                metadata['is_helpful'] = is_helpful
                existing_vote.interaction_metadata = json.dumps(metadata)
                existing_vote.timestamp = datetime.utcnow()
                
                # Adjust helpful count
                if old_helpful != is_helpful:
                    if is_helpful:
                        review.helpful_count = (review.helpful_count or 0) + 1
                    else:
                        review.helpful_count = max(0, (review.helpful_count or 0) - 1)
            else:
                # Create new vote
                vote_interaction = self.UserInteraction(
                    user_id=user_id,
                    content_id=review.content_id,
                    interaction_type='review_vote',
                    interaction_metadata=json.dumps({
                        'review_id': review_id,
                        'is_helpful': is_helpful
                    }),
                    timestamp=datetime.utcnow()
                )
                self.db.session.add(vote_interaction)
                
                # Adjust helpful count
                if is_helpful:
                    review.helpful_count = (review.helpful_count or 0) + 1
            
            # Clear caches
            content = self.Content.query.get(review.content_id)
            if content:
                self._clear_review_caches(content.id, content.slug)
            
            self.db.session.commit()
            
            return {
                'success': True,
                'helpful_count': review.helpful_count,
                'message': 'Vote recorded successfully'
            }
            
        except Exception as e:
            logger.error(f"Error voting on review: {e}")
            self.db.session.rollback()
            return {'success': False, 'error': 'Failed to record vote'}
    
    def get_user_reviews(self, user_id: int, page: int = 1, limit: int = 10, 
                        include_drafts: bool = False) -> Dict:
        """Get user's reviews"""
        try:
            query = self.db.session.query(self.Review, self.Content).join(
                self.Content
            ).filter(self.Review.user_id == user_id)
            
            if not include_drafts:
                query = query.filter(self.Review.is_approved == True)
            
            query = query.order_by(desc(self.Review.created_at))
            
            # Get total count
            total_reviews = query.count()
            
            # Apply pagination
            offset = (page - 1) * limit
            review_entries = query.offset(offset).limit(limit).all()
            
            # Format reviews
            reviews = []
            user = self.User.query.get(user_id)
            
            for review, content in review_entries:
                review_data = self._format_review(review, user, content, user_id)
                reviews.append(review_data)
            
            # Get user review stats
            from .analysis import ReviewAnalyticsService
            analytics_service = ReviewAnalyticsService(self.db, self.models, self.cache)
            stats = analytics_service.get_user_review_stats(user_id)
            
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
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Error getting user reviews: {e}")
            return {
                'success': False,
                'error': 'Failed to get user reviews',
                'reviews': [],
                'pagination': {},
                'stats': {}
            }
    
    def add_rating(self, content_slug: str, user_id: int, rating: float) -> Dict:
        """Add or update a quick rating (without review text)"""
        try:
            # Get content
            content = self._get_content_by_slug(content_slug)
            if not content:
                return {'success': False, 'error': 'Content not found'}
            
            # Validate rating
            if not (self.MIN_RATING <= rating <= self.MAX_RATING):
                return {'success': False, 'error': f'Rating must be between {self.MIN_RATING} and {self.MAX_RATING}'}
            
            # Check if user already has a review
            existing_review = self.Review.query.filter_by(
                content_id=content.id,
                user_id=user_id
            ).first()
            
            if existing_review:
                # Update existing review rating
                existing_review.rating = rating
                existing_review.updated_at = datetime.utcnow()
                message = 'Rating updated successfully'
            else:
                # Create rating-only review
                review = self.Review(
                    content_id=content.id,
                    user_id=user_id,
                    rating=rating,
                    review_text='',  # Empty text for rating-only
                    has_spoilers=False,
                    is_approved=True,  # Auto-approve ratings
                    helpful_count=0,
                    created_at=datetime.utcnow()
                )
                self.db.session.add(review)
                message = 'Rating added successfully'
            
            # Record user interaction
            self._record_review_interaction(user_id, content.id, rating)
            
            # Clear caches
            self._clear_review_caches(content.id, content.slug)
            
            self.db.session.commit()
            
            return {
                'success': True,
                'message': message,
                'rating': rating
            }
            
        except Exception as e:
            logger.error(f"Error adding rating: {e}")
            self.db.session.rollback()
            return {'success': False, 'error': 'Failed to add rating'}
    
    def get_user_rating(self, content_slug: str, user_id: int) -> Dict:
        """Get user's rating for specific content"""
        try:
            content = self._get_content_by_slug(content_slug)
            if not content:
                return {'success': False, 'error': 'Content not found'}
            
            review = self.Review.query.filter_by(
                content_id=content.id,
                user_id=user_id
            ).first()
            
            return {
                'success': True,
                'has_rating': review is not None,
                'rating': review.rating if review else None,
                'has_review': review and bool(review.review_text.strip()) if review else False,
                'review_id': review.id if review else None,
                'created_at': review.created_at.isoformat() if review else None
            }
            
        except Exception as e:
            logger.error(f"Error getting user rating: {e}")
            return {'success': False, 'error': 'Failed to get rating'}
    
    # Private helper methods
    
    def _get_content_by_slug(self, slug: str):
        """Get content by slug"""
        return self.Content.query.filter_by(slug=slug).first()
    
    def _validate_review_data(self, review_data: Dict) -> Dict:
        """Validate review data"""
        errors = []
        
        # Validate rating
        rating = review_data.get('rating')
        if not rating:
            errors.append('Rating is required')
        else:
            try:
                rating = float(rating)
                if rating < self.MIN_RATING or rating > self.MAX_RATING:
                    errors.append(f'Rating must be between {self.MIN_RATING} and {self.MAX_RATING}')
            except (ValueError, TypeError):
                errors.append('Invalid rating format')
        
        # Validate review text (optional for ratings)
        review_text = review_data.get('review_text', '').strip()
        if review_text:  # Only validate if text is provided
            if len(review_text) < self.MIN_REVIEW_LENGTH:
                errors.append(f'Review text must be at least {self.MIN_REVIEW_LENGTH} characters')
            elif len(review_text) > self.MAX_REVIEW_LENGTH:
                errors.append(f'Review text must be less than {self.MAX_REVIEW_LENGTH} characters')
        
        # Validate title if provided
        title = review_data.get('title', '').strip()
        if title and len(title) > self.MAX_TITLE_LENGTH:
            errors.append(f'Review title must be less than {self.MAX_TITLE_LENGTH} characters')
        
        # Check for spam patterns
        if review_text and self._is_spam_content(review_text, title):
            errors.append('Review content appears to be spam')
        
        return {
            'valid': len(errors) == 0,
            'error': '; '.join(errors) if errors else None
        }
    
    def _should_auto_approve_review(self, user, review_data: Dict) -> bool:
        """Determine if review should be auto-approved"""
        try:
            # Always auto-approve admin reviews
            if self.AUTO_APPROVE_ADMIN and user.is_admin:
                return True
            
            # Auto-approve rating-only reviews
            review_text = review_data.get('review_text', '').strip()
            if not review_text:
                return True
            
            # Check trusted user criteria
            if self.AUTO_APPROVE_TRUSTED_USERS:
                approved_count = self.Review.query.filter_by(
                    user_id=user.id,
                    is_approved=True
                ).count()
                
                if approved_count >= self.TRUSTED_USER_REVIEW_COUNT:
                    return True
            
            # Check minimum length for auto-approval
            if len(review_text) >= self.AUTO_APPROVE_MIN_LENGTH:
                return True
            
            # For now, auto-approve all valid reviews
            return True
            
        except Exception as e:
            logger.error(f"Error in auto-approval logic: {e}")
            return False
    
    def _needs_reapproval(self, review, new_data: Dict) -> bool:
        """Check if review needs re-approval after edit"""
        # Only check if review has text
        if not review.review_text.strip():
            return False
        
        # Check if review text changed significantly
        old_text = review.review_text.strip().lower()
        new_text = new_data.get('review_text', '').strip().lower()
        
        # Simple check: if more than 50% of content changed
        if len(old_text) > 0 and len(new_text) > 0:
            similarity = SequenceMatcher(None, old_text, new_text).ratio()
            if similarity < 0.5:  # Less than 50% similar
                return True
        
        # Check if rating changed significantly
        old_rating = review.rating
        new_rating = float(new_data.get('rating', 0))
        if abs(old_rating - new_rating) >= 3:  # Changed by 3+ points
            return True
        
        return False
    
    def _is_spam_content(self, review_text: str, title: str = None) -> bool:
        """Check if content appears to be spam"""
        try:
            text = (review_text + ' ' + (title or '')).lower()
            
            # Common spam patterns
            spam_patterns = [
                r'visit\s+our\s+website',
                r'click\s+here',
                r'free\s+money',
                r'100%\s+guaranteed',
                r'make\s+money\s+fast',
                r'www\.[a-z0-9\-]+\.[a-z]{2,}',
                r'http[s]?://',
                r'call\s+now',
                r'limited\s+time\s+offer'
            ]
            
            for pattern in spam_patterns:
                if re.search(pattern, text):
                    return True
            
            # Check for excessive repetition
            words = text.split()
            if len(words) > 10:
                word_count = defaultdict(int)
                for word in words:
                    if len(word) > 3:  # Only count longer words
                        word_count[word] += 1
                
                # If any word appears more than 30% of the time
                max_count = max(word_count.values()) if word_count else 0
                if max_count > len(words) * 0.3:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in spam detection: {e}")
            return False
    
    def _record_review_interaction(self, user_id: int, content_id: int, rating: float):
        """Record user review interaction"""
        try:
            # Update or create rating interaction
            interaction = self.UserInteraction.query.filter_by(
                user_id=user_id,
                content_id=content_id,
                interaction_type='rating'
            ).first()
            
            if interaction:
                interaction.rating = rating
                interaction.timestamp = datetime.utcnow()
            else:
                interaction = self.UserInteraction(
                    user_id=user_id,
                    content_id=content_id,
                    interaction_type='rating',
                    rating=rating,
                    timestamp=datetime.utcnow()
                )
                self.db.session.add(interaction)
                
        except Exception as e:
            logger.error(f"Error recording review interaction: {e}")
    
    def _format_review(self, review, user, content, current_user_id: Optional[int] = None) -> Dict:
        """Format review for API response"""
        try:
            # Check if current user voted on this review
            user_vote = None
            if current_user_id:
                vote_interaction = self.UserInteraction.query.filter_by(
                    user_id=current_user_id,
                    content_id=content.id,
                    interaction_type='review_vote'
                ).filter(
                    text(f"JSON_EXTRACT(interaction_metadata, '$.review_id') = {review.id}")
                ).first()
                
                if vote_interaction:
                    metadata = json.loads(vote_interaction.interaction_metadata or '{}')
                    user_vote = metadata.get('is_helpful')
            
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
                    'avatar_url': user.avatar_url,
                    'is_verified': getattr(user, 'is_verified', False)
                },
                'content': {
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'poster_url': content.poster_path
                },
                'user_vote': user_vote,
                'can_edit': current_user_id == user.id,
                'can_delete': current_user_id == user.id,
                'is_rating_only': not bool(review.review_text.strip())
            }
            
        except Exception as e:
            logger.error(f"Error formatting review: {e}")
            return {}
    
    def _clear_review_caches(self, content_id: int, content_slug: str):
        """Clear review-related caches"""
        if not self.cache:
            return
        
        try:
            # Clear content review caches
            cache_keys = [
                f"review_stats:{content_id}",
                f"details:slug:{content_slug}"
            ]
            
            for key in cache_keys:
                self.cache.delete(key)
                
        except Exception as e:
            logger.error(f"Error clearing review caches: {e}")
    
    # Notification methods (implement based on your notification system)
    
    def _notify_review_published(self, review, user, content):
        """Notify user that review was published"""
        try:
            logger.info(f"Review {review.id} published notification sent to user {user.id}")
        except Exception as e:
            logger.error(f"Error sending review published notification: {e}")
    
    def _notify_review_pending(self, review, user, content):
        """Notify user that review is pending moderation"""
        try:
            logger.info(f"Review {review.id} pending notification sent to user {user.id}")
        except Exception as e:
            logger.error(f"Error sending review pending notification: {e}")