# admin/users.py

import os
import csv
import io
import json
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import func, desc, and_, or_, text, case
from sqlalchemy.exc import IntegrityError
import pandas as pd

logger = logging.getLogger(__name__)

class AdminUserService:
    """Comprehensive user management service for CineBrain administrators"""
    
    def __init__(self, app, db, models, services):
        self.app = app
        self.db = db
        self.User = models['User']
        self.UserInteraction = models.get('UserInteraction')
        self.Content = models.get('Content')
        self.Review = models.get('Review')
        
        # Services
        self.email_service = services.get('email_service')
        self.notification_service = services.get('admin_notification_service')
        self.cache = services.get('cache')
        
        logger.info("✅ Admin User Service initialized")

    def get_users_list(self, page: int = 1, per_page: int = 25, 
                      sort_by: str = 'created_at', sort_direction: str = 'desc',
                      filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get paginated, filtered, and sorted user list
        
        @param page: Current page number
        @param per_page: Items per page
        @param sort_by: Field to sort by
        @param sort_direction: Sort direction (asc/desc)
        @param filters: Filter criteria
        @return: Paginated user data
        """
        try:
            query = self.User.query
            
            # Apply filters
            if filters:
                query = self._apply_user_filters(query, filters)
            
            # Apply sorting
            query = self._apply_user_sorting(query, sort_by, sort_direction)
            
            # Paginate
            paginated = query.paginate(
                page=page, 
                per_page=per_page, 
                error_out=False
            )
            
            # Format users data
            users = []
            for user in paginated.items:
                user_data = self._format_user_data(user)
                users.append(user_data)
            
            return {
                'success': True,
                'data': {
                    'users': users,
                    'total': paginated.total,
                    'page': page,
                    'per_page': per_page,
                    'total_pages': paginated.pages,
                    'has_prev': paginated.has_prev,
                    'has_next': paginated.has_next
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting users list: {e}")
            return {
                'success': False,
                'error': 'Failed to get users list',
                'data': {
                    'users': [],
                    'total': 0,
                    'page': 1,
                    'per_page': per_page,
                    'total_pages': 0
                }
            }

    def get_user_statistics(self, timeframe: str = 'week') -> Dict[str, Any]:
        """
        Get comprehensive user statistics
        
        @param timeframe: Statistics timeframe (today/week/month/year)
        @return: User statistics data
        """
        try:
            # Calculate date ranges
            now = datetime.utcnow()
            start_date, previous_start, previous_end = self._get_timeframe_dates(timeframe)
            
            # Current period stats
            current_stats = self._calculate_user_stats(start_date, now)
            
            # Previous period stats for comparison
            previous_stats = self._calculate_user_stats(previous_start, previous_end)
            
            # Engagement metrics
            engagement_stats = self._calculate_engagement_metrics(start_date, now)
            
            # Combine all statistics
            statistics = {
                'total_users': current_stats['total_users'],
                'total_users_previous': previous_stats['total_users'],
                
                'active_users': current_stats['active_users'],
                'active_users_previous': previous_stats['active_users'],
                
                'new_users': current_stats['new_users'],
                'new_users_previous': previous_stats['new_users'],
                
                'admin_users': current_stats['admin_users'],
                'admin_users_previous': previous_stats['admin_users'],
                
                'suspended_users': current_stats['suspended_users'],
                'suspended_users_previous': previous_stats['suspended_users'],
                
                'engagement_rate': engagement_stats['engagement_rate'],
                'engagement_rate_previous': engagement_stats['engagement_rate_previous']
            }
            
            return {
                'success': True,
                'data': statistics
            }
            
        except Exception as e:
            logger.error(f"Error getting user statistics: {e}")
            return {
                'success': False,
                'error': 'Failed to get user statistics',
                'data': {}
            }

    def get_user_analytics(self, period: str = '30d') -> Dict[str, Any]:
        """
        Get detailed user analytics with activity trends
        
        @param period: Analytics period (7d/30d/90d)
        @return: Analytics data
        """
        try:
            days = self._parse_period(period)
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # User activity trends
            activity_data = self._get_user_activity_trends(start_date, end_date)
            
            # Engagement metrics
            engagement_data = self._get_engagement_analytics(start_date, end_date)
            
            return {
                'success': True,
                'data': {
                    'user_activity': activity_data,
                    'engagement': engagement_data,
                    'period': period,
                    'generated_at': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting user analytics: {e}")
            return {
                'success': False,
                'error': 'Failed to get user analytics',
                'data': {}
            }

    def get_user_details(self, user_id: int) -> Dict[str, Any]:
        """
        Get comprehensive details for a specific user
        
        @param user_id: User ID
        @return: Detailed user information
        """
        try:
            user = self.User.query.get(user_id)
            if not user:
                return {
                    'success': False,
                    'error': 'User not found'
                }
            
            # Get interaction counts
            interaction_counts = self._get_user_interaction_counts(user_id)
            
            # Format detailed user data
            user_data = self._format_user_data(user, detailed=True)
            user_data.update(interaction_counts)
            
            return {
                'success': True,
                'data': user_data
            }
            
        except Exception as e:
            logger.error(f"Error getting user details for {user_id}: {e}")
            return {
                'success': False,
                'error': 'Failed to get user details'
            }

    def update_user(self, user_id: int, admin_user: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user information
        
        @param user_id: User ID to update
        @param admin_user: Admin performing the update
        @param data: Update data
        @return: Update result
        """
        try:
            user = self.User.query.get(user_id)
            if not user:
                return {
                    'success': False,
                    'error': 'User not found'
                }
            
            # Validate updates
            validation_error = self._validate_user_update(user, data)
            if validation_error:
                return {
                    'success': False,
                    'error': validation_error
                }
            
            # Track changes for audit
            changes = []
            
            # Update username if provided and changed
            if 'username' in data and data['username'] != user.username:
                old_username = user.username
                user.username = data['username']
                changes.append(f"Username: {old_username} → {data['username']}")
            
            # Update email if provided and changed
            if 'email' in data and data['email'] != user.email:
                old_email = user.email
                user.email = data['email']
                changes.append(f"Email: {old_email} → {data['email']}")
            
            # Update admin status
            if 'is_admin' in data and data['is_admin'] != user.is_admin:
                user.is_admin = data['is_admin']
                changes.append(f"Admin status: {'granted' if data['is_admin'] else 'revoked'}")
            
            # Update status
            if 'status' in data:
                status_updated = self._update_user_status(user, data['status'])
                if status_updated:
                    changes.append(f"Status: {status_updated}")
            
            # Update optional fields
            if 'full_name' in data:
                user.full_name = data['full_name']
            
            if 'location' in data:
                user.location = data['location']
            
            # Commit changes
            self.db.session.commit()
            
            # Send notification if significant changes
            if changes and self.notification_service:
                self.notification_service.create_notification(
                    'USER_ACTIVITY',
                    f'User Updated: {user.username}',
                    f'Admin {admin_user.username} updated user:\n' + '\n'.join(changes),
                    admin_id=admin_user.id,
                    metadata={'user_id': user_id, 'changes': changes}
                )
            
            # Clear user cache
            self._clear_user_cache(user_id)
            
            logger.info(f"✅ User {user_id} updated by admin {admin_user.id}: {', '.join(changes)}")
            
            return {
                'success': True,
                'message': 'User updated successfully',
                'changes': changes
            }
            
        except IntegrityError as e:
            self.db.session.rollback()
            if 'username' in str(e):
                return {'success': False, 'error': 'Username already exists'}
            elif 'email' in str(e):
                return {'success': False, 'error': 'Email already exists'}
            else:
                return {'success': False, 'error': 'Database constraint violation'}
                
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error updating user {user_id}: {e}")
            return {
                'success': False,
                'error': 'Failed to update user'
            }

    def toggle_user_status(self, user_id: int, admin_user: Any, action: str) -> Dict[str, Any]:
        """
        Toggle user status (suspend/activate)
        
        @param user_id: User ID
        @param admin_user: Admin performing the action
        @param action: Action to perform (suspend/activate)
        @return: Action result
        """
        try:
            user = self.User.query.get(user_id)
            if not user:
                return {
                    'success': False,
                    'error': 'User not found'
                }
            
            # Prevent self-suspension
            if user_id == admin_user.id and action == 'suspend':
                return {
                    'success': False,
                    'error': 'Cannot suspend your own account'
                }
            
            # Perform action
            if action == 'suspend':
                user.is_suspended = True
                user.suspended_at = datetime.utcnow()
                user.suspended_by = admin_user.id
                message = f"User {user.username} suspended"
                
            elif action == 'activate':
                user.is_suspended = False
                user.is_banned = False
                user.suspended_at = None
                user.suspended_by = None
                message = f"User {user.username} activated"
                
            else:
                return {
                    'success': False,
                    'error': 'Invalid action'
                }
            
            self.db.session.commit()
            
            # Send notification
            if self.notification_service:
                self.notification_service.create_notification(
                    'USER_ACTIVITY',
                    message,
                    f'Admin {admin_user.username} {action}d user {user.username}',
                    admin_id=admin_user.id,
                    metadata={'user_id': user_id, 'action': action}
                )
            
            # Send email to user if configured
            if self.email_service and self.email_service.email_enabled:
                self._send_status_change_email(user, action)
            
            logger.info(f"✅ User {user_id} {action}d by admin {admin_user.id}")
            
            return {
                'success': True,
                'message': message
            }
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error toggling user status: {e}")
            return {
                'success': False,
                'error': f'Failed to {action} user'
            }

    def bulk_user_operation(self, admin_user: Any, action: str, user_ids: List[int]) -> Dict[str, Any]:
        """
        Perform bulk operations on multiple users
        
        @param admin_user: Admin performing the operation
        @param action: Action to perform (suspend/activate/delete)
        @param user_ids: List of user IDs
        @return: Operation result
        """
        try:
            if not user_ids:
                return {
                    'success': False,
                    'error': 'No users selected'
                }
            
            # Prevent self-operations
            if admin_user.id in user_ids and action in ['suspend', 'delete']:
                user_ids.remove(admin_user.id)
                
            results = {
                'success_count': 0,
                'failed_count': 0,
                'skipped': [],
                'errors': []
            }
            
            for user_id in user_ids:
                try:
                    user = self.User.query.get(user_id)
                    if not user:
                        results['failed_count'] += 1
                        results['errors'].append(f"User {user_id} not found")
                        continue
                    
                    if action == 'suspend':
                        if not user.is_suspended:
                            user.is_suspended = True
                            user.suspended_at = datetime.utcnow()
                            user.suspended_by = admin_user.id
                            results['success_count'] += 1
                        else:
                            results['skipped'].append(user_id)
                            
                    elif action == 'activate':
                        if user.is_suspended or user.is_banned:
                            user.is_suspended = False
                            user.is_banned = False
                            user.suspended_at = None
                            user.suspended_by = None
                            results['success_count'] += 1
                        else:
                            results['skipped'].append(user_id)
                            
                    elif action == 'delete':
                        # Soft delete by anonymizing
                        user.username = f"deleted_user_{user.id}"
                        user.email = f"deleted_{user.id}@cinebrain.deleted"
                        user.is_deleted = True
                        user.deleted_at = datetime.utcnow()
                        user.deleted_by = admin_user.id
                        results['success_count'] += 1
                        
                except Exception as e:
                    results['failed_count'] += 1
                    results['errors'].append(f"Error processing user {user_id}: {str(e)}")
            
            # Commit all changes
            if results['success_count'] > 0:
                self.db.session.commit()
                
                # Send notification
                if self.notification_service:
                    self.notification_service.create_notification(
                        'USER_ACTIVITY',
                        f'Bulk {action} operation completed',
                        f'Admin {admin_user.username} {action}d {results["success_count"]} users',
                        admin_id=admin_user.id,
                        metadata={'action': action, 'results': results}
                    )
            
            logger.info(f"✅ Bulk {action} completed: {results['success_count']} succeeded, "
                       f"{results['failed_count']} failed, {len(results['skipped'])} skipped")
            
            return {
                'success': True,
                'message': f'{action.capitalize()} operation completed',
                'results': results
            }
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error in bulk operation: {e}")
            return {
                'success': False,
                'error': f'Failed to perform bulk {action}'
            }

    def export_users(self, filters: Optional[Dict[str, Any]] = None, 
                    format: str = 'csv') -> Tuple[Any, str, str]:
        """
        Export users data in specified format
        
        @param filters: Filter criteria
        @param format: Export format (csv/json)
        @return: File data, filename, mimetype
        """
        try:
            # Get filtered users
            query = self.User.query
            if filters:
                query = self._apply_user_filters(query, filters)
            
            users = query.all()
            
            if format == 'csv':
                return self._export_users_csv(users)
            elif format == 'json':
                return self._export_users_json(users)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting users: {e}")
            raise

    # ==================== PRIVATE HELPER METHODS ====================

    def _apply_user_filters(self, query, filters: Dict[str, Any]):
        """Apply filters to user query"""
        
        # Status filter
        if filters.get('status'):
            if filters['status'] == 'active':
                query = query.filter(
                    and_(
                        self.User.is_suspended == False,
                        self.User.is_banned == False,
                        or_(
                            self.User.is_deleted == False,
                            self.User.is_deleted == None
                        )
                    )
                )
            elif filters['status'] == 'inactive':
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                query = query.filter(
                    or_(
                        self.User.last_active < thirty_days_ago,
                        self.User.last_active == None
                    )
                )
            elif filters['status'] == 'new':
                week_ago = datetime.utcnow() - timedelta(days=7)
                query = query.filter(self.User.created_at >= week_ago)
        
        # Role filter
        if filters.get('role'):
            if filters['role'] == 'admin':
                query = query.filter(self.User.is_admin == True)
            elif filters['role'] == 'user':
                query = query.filter(self.User.is_admin == False)
        
        # Registration filter
        if filters.get('registration'):
            now = datetime.utcnow()
            if filters['registration'] == 'today':
                today = now.date()
                query = query.filter(func.date(self.User.created_at) == today)
            elif filters['registration'] == 'week':
                week_start = now - timedelta(days=now.weekday())
                query = query.filter(self.User.created_at >= week_start)
            elif filters['registration'] == 'month':
                month_start = now.replace(day=1)
                query = query.filter(self.User.created_at >= month_start)
        
        # Search filter
        if filters.get('search'):
            search_term = f"%{filters['search']}%"
            query = query.filter(
                or_(
                    self.User.username.ilike(search_term),
                    self.User.email.ilike(search_term),
                    self.User.full_name.ilike(search_term)
                )
            )
        
        return query

    def _apply_user_sorting(self, query, sort_by: str, sort_direction: str):
        """Apply sorting to user query"""
        
        # Map sort fields to model attributes
        sort_map = {
            'username': self.User.username,
            'email': self.User.email,
            'role': self.User.is_admin,
            'status': case(
                (self.User.is_suspended == True, 1),
                (self.User.is_banned == True, 2),
                else_=0
            ),
            'created_at': self.User.created_at,
            'last_active': self.User.last_active
        }
        
        if sort_by in sort_map:
            order_column = sort_map[sort_by]
            if sort_direction == 'desc':
                query = query.order_by(desc(order_column))
            else:
                query = query.order_by(order_column)
        else:
            # Default sort
            query = query.order_by(desc(self.User.created_at))
        
        return query

    def _format_user_data(self, user, detailed: bool = False) -> Dict[str, Any]:
        """Format user data for response"""
        data = {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'full_name': getattr(user, 'full_name', None),
            'is_admin': user.is_admin,
            'is_suspended': getattr(user, 'is_suspended', False),
            'is_banned': getattr(user, 'is_banned', False),
            'is_active': True,  # Calculate based on last_active
            'avatar_url': user.avatar_url,
            'created_at': user.created_at.isoformat() if user.created_at else None,
            'last_active': user.last_active.isoformat() if user.last_active else None,
            'location': user.location
        }
        
        # Calculate if user is active
        if user.last_active:
            days_inactive = (datetime.utcnow() - user.last_active).days
            data['is_active'] = days_inactive < 30
        
        if detailed:
            # Add additional details
            data['preferred_languages'] = json.loads(user.preferred_languages or '[]')
            data['preferred_genres'] = json.loads(user.preferred_genres or '[]')
            data['suspended_at'] = user.suspended_at.isoformat() if hasattr(user, 'suspended_at') and user.suspended_at else None
            data['suspended_by'] = getattr(user, 'suspended_by', None)
        
        return data

    def _get_timeframe_dates(self, timeframe: str) -> Tuple[datetime, datetime, datetime]:
        """Calculate date ranges for timeframe"""
        now = datetime.utcnow()
        
        if timeframe == 'today':
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            previous_start = start - timedelta(days=1)
            previous_end = start
            
        elif timeframe == 'week':
            start = now - timedelta(days=7)
            previous_start = start - timedelta(days=7)
            previous_end = start
            
        elif timeframe == 'month':
            start = now - timedelta(days=30)
            previous_start = start - timedelta(days=30)
            previous_end = start
            
        elif timeframe == 'year':
            start = now - timedelta(days=365)
            previous_start = start - timedelta(days=365)
            previous_end = start
            
        else:
            # Default to week
            start = now - timedelta(days=7)
            previous_start = start - timedelta(days=7)
            previous_end = start
        
        return start, previous_start, previous_end

    def _calculate_user_stats(self, start_date: datetime, end_date: datetime) -> Dict[str, int]:
        """Calculate user statistics for a date range"""
        
        # Total users (as of end date)
        total_users = self.User.query.filter(
            self.User.created_at <= end_date
        ).count()
        
        # Active users in period
        active_users = self.User.query.filter(
            and_(
                self.User.last_active >= start_date,
                self.User.last_active <= end_date
            )
        ).count()
        
        # New users in period
        new_users = self.User.query.filter(
            and_(
                self.User.created_at >= start_date,
                self.User.created_at <= end_date
            )
        ).count()
        
        # Admin users
        admin_users = self.User.query.filter(
            and_(
                self.User.is_admin == True,
                self.User.created_at <= end_date
            )
        ).count()
        
        # Suspended users
        suspended_users = self.User.query.filter(
            and_(
                or_(
                    self.User.is_suspended == True,
                    self.User.is_banned == True
                ),
                self.User.created_at <= end_date
            )
        ).count()
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'new_users': new_users,
            'admin_users': admin_users,
            'suspended_users': suspended_users
        }

    def _calculate_engagement_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Calculate engagement metrics"""
        
        total_users = self.User.query.filter(
            self.User.created_at <= end_date
        ).count()
        
        active_users = self.User.query.filter(
            and_(
                self.User.last_active >= start_date,
                self.User.last_active <= end_date
            )
        ).count()
        
        # Calculate for previous period
        period_length = (end_date - start_date).days
        previous_start = start_date - timedelta(days=period_length)
        previous_end = start_date
        
        previous_active = self.User.query.filter(
            and_(
                self.User.last_active >= previous_start,
                self.User.last_active <= previous_end
            )
        ).count()
        
        previous_total = self.User.query.filter(
            self.User.created_at <= previous_end
        ).count()
        
        # Calculate engagement rates
        engagement_rate = (active_users / total_users * 100) if total_users > 0 else 0
        engagement_rate_previous = (previous_active / previous_total * 100) if previous_total > 0 else 0
        
        return {
            'engagement_rate': round(engagement_rate, 1),
            'engagement_rate_previous': round(engagement_rate_previous, 1)
        }

    def _get_user_activity_trends(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get daily user activity trends"""
        
        # Generate date range
        date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
        
        activity_data = []
        
        for current_date in date_range:
            next_date = current_date + timedelta(days=1)
            
            # Active users for the day
            active_count = self.User.query.filter(
                and_(
                    self.User.last_active >= current_date,
                    self.User.last_active < next_date
                )
            ).count()
            
            # New users for the day
            new_count = self.User.query.filter(
                and_(
                    self.User.created_at >= current_date,
                    self.User.created_at < next_date
                )
            ).count()
            
            activity_data.append({
                'date': current_date.isoformat(),
                'active_users': active_count,
                'new_users': new_count
            })
        
        return activity_data

    def _get_engagement_analytics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get detailed engagement analytics"""
        
        total_users = self.User.query.count()
        
        # Daily active users (active today)
        today = datetime.utcnow().date()
        daily_active = self.User.query.filter(
            func.date(self.User.last_active) == today
        ).count()
        
        # Weekly active users (active in last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        weekly_active = self.User.query.filter(
            self.User.last_active >= week_ago
        ).count()
        
        # Monthly active users (active in last 30 days)
        month_ago = datetime.utcnow() - timedelta(days=30)
        monthly_active = self.User.query.filter(
            self.User.last_active >= month_ago
        ).count()
        
        # Average session time (mock data for now - would need session tracking)
        avg_session_time = 1245  # 20 minutes 45 seconds
        
        return {
            'total_users': total_users,
            'daily_active': daily_active,
            'weekly_active': weekly_active,
            'monthly_active': monthly_active,
            'avg_session_time': avg_session_time
        }

    def _get_user_interaction_counts(self, user_id: int) -> Dict[str, int]:
        """Get interaction counts for a user"""
        counts = {
            'interaction_count': 0,
            'ratings_count': 0,
            'favorites_count': 0,
            'watchlist_count': 0
        }
        
        if self.UserInteraction:
            # Total interactions
            counts['interaction_count'] = self.UserInteraction.query.filter_by(
                user_id=user_id
            ).count()
            
            # Ratings
            counts['ratings_count'] = self.UserInteraction.query.filter_by(
                user_id=user_id,
                interaction_type='rating'
            ).count()
            
            # Favorites
            counts['favorites_count'] = self.UserInteraction.query.filter_by(
                user_id=user_id,
                interaction_type='favorite'
            ).count()
            
            # Watchlist
            counts['watchlist_count'] = self.UserInteraction.query.filter_by(
                user_id=user_id,
                interaction_type='watchlist'
            ).count()
        
        # Add review count if Review model exists
        if self.Review:
            counts['reviews_count'] = self.Review.query.filter_by(
                user_id=user_id
            ).count()
        
        return counts

    def _parse_period(self, period: str) -> int:
        """Parse period string to days"""
        if period == '7d':
            return 7
        elif period == '30d':
            return 30
        elif period == '90d':
            return 90
        else:
            return 30  # Default

    def _validate_user_update(self, user, data: Dict[str, Any]) -> Optional[str]:
        """Validate user update data"""
        
        # Check username uniqueness
        if 'username' in data and data['username'] != user.username:
            existing = self.User.query.filter_by(username=data['username']).first()
            if existing:
                return 'Username already exists'
        
        # Check email uniqueness
        if 'email' in data and data['email'] != user.email:
            existing = self.User.query.filter_by(email=data['email']).first()
            if existing:
                return 'Email already exists'
        
        # Validate email format
        if 'email' in data:
            import re
            if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', data['email']):
                return 'Invalid email format'
        
        return None

    def _update_user_status(self, user, status: str) -> Optional[str]:
        """Update user status based on string value"""
        
        if status == 'active':
            if user.is_suspended or user.is_banned:
                user.is_suspended = False
                user.is_banned = False
                user.suspended_at = None
                user.suspended_by = None
                return 'activated'
                
        elif status == 'suspended':
            if not user.is_suspended:
                user.is_suspended = True
                user.suspended_at = datetime.utcnow()
                return 'suspended'
                
        elif status == 'banned':
            if not user.is_banned:
                user.is_banned = True
                user.suspended_at = datetime.utcnow()
                return 'banned'
        
        return None

    def _clear_user_cache(self, user_id: int):
        """Clear user-related cache entries"""
        if self.cache:
            try:
                # Clear specific user cache keys
                cache_keys = [
                    f'user:{user_id}',
                    f'user:profile:{user_id}',
                    f'user:stats:{user_id}'
                ]
                
                for key in cache_keys:
                    self.cache.delete(key)
                    
            except Exception as e:
                logger.warning(f"Failed to clear user cache: {e}")

    def _send_status_change_email(self, user, action: str):
        """Send email notification for status changes"""
        try:
            if action == 'suspend':
                subject = "Your CineBrain account has been suspended"
                message = """Your CineBrain account has been temporarily suspended. 
                            If you believe this is an error, please contact support."""
            elif action == 'activate':
                subject = "Your CineBrain account has been reactivated"
                message = "Good news! Your CineBrain account has been reactivated and you can now access all features."
            else:
                return
            
            # Use email service if available
            if hasattr(self.email_service, 'send_email'):
                self.email_service.send_email(
                    to=user.email,
                    subject=subject,
                    body=message,
                    is_html=False
                )
                
        except Exception as e:
            logger.warning(f"Failed to send status change email: {e}")

    def _export_users_csv(self, users: List[Any]) -> Tuple[io.StringIO, str, str]:
        """Export users as CSV"""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            'id', 'username', 'email', 'full_name', 'is_admin', 
            'is_suspended', 'is_banned', 'location', 'created_at', 
            'last_active', 'preferred_languages', 'preferred_genres'
        ])
        
        writer.writeheader()
        
        for user in users:
            writer.writerow({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': getattr(user, 'full_name', ''),
                'is_admin': 'Yes' if user.is_admin else 'No',
                'is_suspended': 'Yes' if getattr(user, 'is_suspended', False) else 'No',
                'is_banned': 'Yes' if getattr(user, 'is_banned', False) else 'No',
                'location': user.location or '',
                'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S') if user.created_at else '',
                'last_active': user.last_active.strftime('%Y-%m-%d %H:%M:%S') if user.last_active else '',
                'preferred_languages': ', '.join(json.loads(user.preferred_languages or '[]')),
                'preferred_genres': ', '.join(json.loads(user.preferred_genres or '[]'))
            })
        
        output.seek(0)
        filename = f"cinebrain-users-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
        
        return output, filename, 'text/csv'

    def _export_users_json(self, users: List[Any]) -> Tuple[str, str, str]:
        """Export users as JSON"""
        user_data = []
        
        for user in users:
            user_data.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': getattr(user, 'full_name', None),
                'is_admin': user.is_admin,
                'is_suspended': getattr(user, 'is_suspended', False),
                'is_banned': getattr(user, 'is_banned', False),
                'location': user.location,
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'last_active': user.last_active.isoformat() if user.last_active else None,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]')
            })
        
        json_data = json.dumps(user_data, indent=2)
        filename = f"cinebrain-users-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        
        return json_data, filename, 'application/json'


def init_user_service(app, db, models, services):
    """Initialize user management service"""
    return AdminUserService(app, db, models, services)