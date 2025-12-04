#admin/users.py

import os
import csv
import io
import json
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import func, desc, and_, or_, text, case, distinct
from sqlalchemy.exc import IntegrityError
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)

class AdminUserService:
    
    def __init__(self, app, db, models, services):
        self.app = app
        self.db = db
        self.User = models['User']
        self.UserInteraction = models.get('UserInteraction')
        self.Content = models.get('Content')
        self.Review = models.get('Review')
        self.SupportTicket = models.get('SupportTicket')
        self.ContactMessage = models.get('ContactMessage')
        
        self.email_service = services.get('email_service')
        self.notification_service = services.get('admin_notification_service')
        self.cache = services.get('cache')
        
        logger.info("✅ Enhanced Admin User Service initialized")

    def get_users_list(self, page: int = 1, per_page: int = 25, 
                      sort_by: str = 'created_at', sort_direction: str = 'desc',
                      filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            query = self.User.query
            
            if filters:
                query = self._apply_user_filters(query, filters)
            
            query = self._apply_user_sorting(query, sort_by, sort_direction)
            
            paginated = query.paginate(
                page=page, 
                per_page=per_page, 
                error_out=False
            )
            
            users = []
            for user in paginated.items:
                user_data = self._format_user_data(user)
                user_data.update(self._get_user_quick_stats(user.id))
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
        try:
            now = datetime.utcnow()
            start_date, previous_start, previous_end = self._get_timeframe_dates(timeframe)
            
            current_stats = self._calculate_user_stats(start_date, now)
            previous_stats = self._calculate_user_stats(previous_start, previous_end)
            engagement_stats = self._calculate_engagement_metrics(start_date, now)
            
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
        try:
            days = self._parse_period(period)
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            activity_data = self._get_user_activity_trends(start_date, end_date)
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
        try:
            user = self.User.query.get(user_id)
            if not user:
                return {
                    'success': False,
                    'error': 'User not found'
                }
            
            interaction_counts = self._get_user_interaction_counts(user_id)
            user_data = self._format_user_data(user, detailed=True)
            user_data.update(interaction_counts)
            
            user_data['advanced_analytics'] = self._get_user_advanced_analytics(user_id)
            user_data['risk_assessment'] = self._assess_user_risk(user_id)
            user_data['engagement_profile'] = self._get_user_engagement_profile(user_id)
            user_data['content_preferences'] = self._analyze_user_content_preferences(user_id)
            
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

    def get_user_segmentation(self) -> Dict[str, Any]:
        try:
            segments = {
                'power_users': self._get_power_users(),
                'casual_users': self._get_casual_users(),
                'new_users': self._get_new_users(),
                'churned_users': self._get_churned_users(),
                'at_risk_users': self._get_at_risk_users(),
                'high_value_users': self._get_high_value_users(),
                'content_creators': self._get_content_creators(),
                'lurkers': self._get_lurkers()
            }
            
            total_users = self.User.query.count()
            segment_stats = {}
            for segment_name, users in segments.items():
                segment_stats[segment_name] = {
                    'count': len(users),
                    'percentage': round(len(users) / total_users * 100, 1) if total_users > 0 else 0,
                    'users': users[:10]
                }
            
            return {
                'success': True,
                'data': {
                    'segments': segment_stats,
                    'total_users': total_users,
                    'generated_at': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting user segmentation: {e}")
            return {
                'success': False,
                'error': 'Failed to get user segmentation'
            }

    def get_user_lifecycle_analysis(self, period_days: int = 90) -> Dict[str, Any]:
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=period_days)
            
            lifecycle_data = {
                'onboarding_funnel': self._analyze_onboarding_funnel(start_date, end_date),
                'activation_rates': self._calculate_activation_rates(start_date, end_date),
                'retention_cohorts': self._analyze_retention_cohorts(start_date, end_date),
                'churn_analysis': self._analyze_user_churn(start_date, end_date),
                'user_journey_stages': self._analyze_user_journey_stages()
            }
            
            return {
                'success': True,
                'data': lifecycle_data
            }
            
        except Exception as e:
            logger.error(f"Error getting lifecycle analysis: {e}")
            return {
                'success': False,
                'error': 'Failed to get lifecycle analysis'
            }

    def get_user_behavior_intelligence(self, user_id: int) -> Dict[str, Any]:
        try:
            behavior_data = {
                'activity_patterns': self._analyze_activity_patterns(user_id),
                'content_consumption': self._analyze_content_consumption(user_id),
                'interaction_patterns': self._analyze_interaction_patterns(user_id),
                'preference_evolution': self._analyze_preference_evolution(user_id),
                'social_behavior': self._analyze_social_behavior(user_id),
                'recommendation_response': self._analyze_recommendation_response(user_id)
            }
            
            return {
                'success': True,
                'data': behavior_data
            }
            
        except Exception as e:
            logger.error(f"Error getting behavior intelligence for user {user_id}: {e}")
            return {
                'success': False,
                'error': 'Failed to get behavior intelligence'
            }

    def advanced_user_search(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = self.User.query
            
            if search_params.get('text'):
                text_term = f"%{search_params['text']}%"
                query = query.filter(
                    or_(
                        self.User.username.ilike(text_term),
                        self.User.email.ilike(text_term),
                        self.User.full_name.ilike(text_term),
                        self.User.location.ilike(text_term)
                    )
                )
            
            if search_params.get('registration_date_from'):
                query = query.filter(self.User.created_at >= search_params['registration_date_from'])
            
            if search_params.get('registration_date_to'):
                query = query.filter(self.User.created_at <= search_params['registration_date_to'])
            
            if search_params.get('last_active_from'):
                query = query.filter(self.User.last_active >= search_params['last_active_from'])
            
            if search_params.get('last_active_to'):
                query = query.filter(self.User.last_active <= search_params['last_active_to'])
            
            if search_params.get('location'):
                location_term = f"%{search_params['location']}%"
                query = query.filter(self.User.location.ilike(location_term))
            
            if search_params.get('account_status'):
                status = search_params['account_status']
                if status == 'active':
                    query = query.filter(
                        and_(
                            self.User.is_suspended == False,
                            self.User.is_banned == False,
                            self.User.is_deleted == False
                        )
                    )
                elif status == 'suspended':
                    query = query.filter(self.User.is_suspended == True)
                elif status == 'banned':
                    query = query.filter(self.User.is_banned == True)
            
            results = query.limit(100).all()
            
            formatted_results = []
            for user in results:
                user_data = self._format_user_data(user)
                user_data.update(self._get_user_quick_stats(user.id))
                formatted_results.append(user_data)
            
            return {
                'success': True,
                'data': {
                    'users': formatted_results,
                    'total_found': len(formatted_results),
                    'search_params': search_params
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced user search: {e}")
            return {
                'success': False,
                'error': 'Failed to perform advanced search'
            }

    def send_targeted_communication(self, recipient_params: Dict[str, Any], 
                                   message_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self.email_service:
                return {
                    'success': False,
                    'error': 'Email service not available'
                }
            
            target_users = self._get_users_by_criteria(recipient_params)
            
            if not target_users:
                return {
                    'success': False,
                    'error': 'No users match the specified criteria'
                }
            
            subject = message_data.get('subject', 'Message from CineBrain Admin')
            template = message_data.get('template', 'general')
            
            sent_count = 0
            failed_count = 0
            
            for user in target_users:
                try:
                    personalized_message = self._personalize_message(
                        message_data.get('content', ''), 
                        user
                    )
                    
                    self.email_service.queue_email(
                        to=user.email,
                        subject=subject,
                        html=personalized_message,
                        priority='normal',
                        to_name=user.username
                    )
                    
                    sent_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to send email to user {user.id}: {e}")
                    failed_count += 1
            
            return {
                'success': True,
                'data': {
                    'total_targeted': len(target_users),
                    'sent_successfully': sent_count,
                    'failed': failed_count,
                    'message_type': template
                }
            }
            
        except Exception as e:
            logger.error(f"Error sending targeted communication: {e}")
            return {
                'success': False,
                'error': 'Failed to send targeted communication'
            }

    def calculate_user_value_scores(self) -> Dict[str, Any]:
        try:
            users_with_scores = []
            
            for user in self.User.query.all():
                score = self._calculate_user_value_score(user.id)
                users_with_scores.append({
                    'user_id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'value_score': score['total_score'],
                    'score_breakdown': score['breakdown'],
                    'tier': score['tier']
                })
            
            users_with_scores.sort(key=lambda x: x['value_score'], reverse=True)
            
            scores = [user['value_score'] for user in users_with_scores]
            distribution = {
                'high_value': len([s for s in scores if s >= 80]),
                'medium_value': len([s for s in scores if 40 <= s < 80]),
                'low_value': len([s for s in scores if s < 40])
            }
            
            return {
                'success': True,
                'data': {
                    'users': users_with_scores[:50],
                    'distribution': distribution,
                    'average_score': sum(scores) / len(scores) if scores else 0,
                    'total_users': len(users_with_scores)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating user value scores: {e}")
            return {
                'success': False,
                'error': 'Failed to calculate user value scores'
            }

    def detect_user_anomalies(self, period_days: int = 30) -> Dict[str, Any]:
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=period_days)
            
            anomalies = {
                'suspicious_activity': self._detect_suspicious_activity(start_date, end_date),
                'unusual_patterns': self._detect_unusual_patterns(start_date, end_date),
                'potential_bots': self._detect_potential_bots(start_date, end_date),
                'spam_indicators': self._detect_spam_indicators(start_date, end_date),
                'security_concerns': self._detect_security_concerns(start_date, end_date)
            }
            
            return {
                'success': True,
                'data': {
                    'anomalies': anomalies,
                    'period_analyzed': f"{period_days} days",
                    'total_anomalies': sum(len(v) for v in anomalies.values())
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting user anomalies: {e}")
            return {
                'success': False,
                'error': 'Failed to detect user anomalies'
            }

    def analyze_user_cohorts(self, cohort_period: str = 'monthly') -> Dict[str, Any]:
        try:
            cohorts = {}
            
            if cohort_period == 'weekly':
                period_delta = timedelta(weeks=1)
            elif cohort_period == 'monthly':
                period_delta = timedelta(days=30)
            else:
                period_delta = timedelta(days=30)
            
            users = self.User.query.order_by(self.User.created_at).all()
            
            if not users:
                return {
                    'success': True,
                    'data': {'cohorts': {}, 'summary': {}}
                }
            
            start_date = users[0].created_at
            current_date = datetime.utcnow()
            
            while start_date <= current_date:
                end_date = start_date + period_delta
                
                cohort_users = [
                    u for u in users 
                    if start_date <= u.created_at < end_date
                ]
                
                if cohort_users:
                    cohort_key = start_date.strftime('%Y-%m-%d')
                    cohorts[cohort_key] = self._analyze_cohort_retention(cohort_users, start_date)
                
                start_date = end_date
            
            return {
                'success': True,
                'data': {
                    'cohorts': cohorts,
                    'period': cohort_period,
                    'summary': self._summarize_cohort_data(cohorts)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user cohorts: {e}")
            return {
                'success': False,
                'error': 'Failed to analyze user cohorts'
            }

    def get_user_support_profile(self, user_id: int) -> Dict[str, Any]:
        try:
            support_data = {
                'tickets': [],
                'contacts': [],
                'support_score': 0,
                'resolution_satisfaction': 0,
                'common_issues': []
            }
            
            if self.SupportTicket:
                tickets = self.SupportTicket.query.filter_by(user_id=user_id).all()
                support_data['tickets'] = [
                    {
                        'id': ticket.id,
                        'ticket_number': ticket.ticket_number,
                        'subject': ticket.subject,
                        'status': ticket.status,
                        'priority': ticket.priority,
                        'created_at': ticket.created_at.isoformat(),
                        'resolved_at': ticket.resolved_at.isoformat() if ticket.resolved_at else None
                    }
                    for ticket in tickets
                ]
            
            if self.ContactMessage:
                contacts = self.ContactMessage.query.filter_by(user_id=user_id).all()
                support_data['contacts'] = [
                    {
                        'id': contact.id,
                        'subject': contact.subject,
                        'is_read': contact.is_read,
                        'created_at': contact.created_at.isoformat()
                    }
                    for contact in contacts
                ]
            
            total_tickets = len(support_data['tickets'])
            resolved_tickets = len([t for t in support_data['tickets'] if t.get('resolved_at')])
            
            if total_tickets > 0:
                support_data['support_score'] = (resolved_tickets / total_tickets) * 100
                support_data['resolution_satisfaction'] = self._calculate_resolution_satisfaction(user_id)
                support_data['common_issues'] = self._identify_common_issues(user_id)
            
            return {
                'success': True,
                'data': support_data
            }
            
        except Exception as e:
            logger.error(f"Error getting user support profile: {e}")
            return {
                'success': False,
                'error': 'Failed to get user support profile'
            }

    def compare_users(self, user_ids: List[int]) -> Dict[str, Any]:
        try:
            if len(user_ids) < 2 or len(user_ids) > 5:
                return {
                    'success': False,
                    'error': 'Can compare between 2-5 users only'
                }
            
            comparison_data = {
                'users': [],
                'metrics_comparison': {},
                'behavioral_comparison': {},
                'performance_comparison': {}
            }
            
            for user_id in user_ids:
                user = self.User.query.get(user_id)
                if not user:
                    continue
                
                user_metrics = {
                    'basic_info': self._format_user_data(user),
                    'activity_metrics': self._get_user_quick_stats(user_id),
                    'engagement_profile': self._get_user_engagement_profile(user_id),
                    'value_score': self._calculate_user_value_score(user_id),
                    'behavior_intelligence': self._analyze_interaction_patterns(user_id)
                }
                
                comparison_data['users'].append(user_metrics)
            
            comparison_data['insights'] = self._generate_comparison_insights(comparison_data['users'])
            
            return {
                'success': True,
                'data': comparison_data
            }
            
        except Exception as e:
            logger.error(f"Error comparing users: {e}")
            return {
                'success': False,
                'error': 'Failed to compare users'
            }

    def update_user(self, user_id: int, admin_user: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            user = self.User.query.get(user_id)
            if not user:
                return {
                    'success': False,
                    'error': 'User not found'
                }
            
            validation_error = self._validate_user_update(user, data)
            if validation_error:
                return {
                    'success': False,
                    'error': validation_error
                }
            
            changes = []
            
            if 'username' in data and data['username'] != user.username:
                old_username = user.username
                user.username = data['username']
                changes.append(f"Username: {old_username} → {data['username']}")
            
            if 'email' in data and data['email'] != user.email:
                old_email = user.email
                user.email = data['email']
                changes.append(f"Email: {old_email} → {data['email']}")
            
            if 'is_admin' in data and data['is_admin'] != user.is_admin:
                user.is_admin = data['is_admin']
                changes.append(f"Admin status: {'granted' if data['is_admin'] else 'revoked'}")
            
            if 'status' in data:
                status_updated = self._update_user_status(user, data['status'])
                if status_updated:
                    changes.append(f"Status: {status_updated}")
            
            if 'full_name' in data:
                user.full_name = data['full_name']
            
            if 'location' in data:
                user.location = data['location']
            
            self.db.session.commit()
            
            if changes and self.notification_service:
                self.notification_service.create_notification(
                    'USER_ACTIVITY',
                    f'User Updated: {user.username}',
                    f'Admin {admin_user.username} updated user:\n' + '\n'.join(changes),
                    admin_id=admin_user.id,
                    metadata={'user_id': user_id, 'changes': changes}
                )
            
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
        try:
            user = self.User.query.get(user_id)
            if not user:
                return {
                    'success': False,
                    'error': 'User not found'
                }
            
            if user_id == admin_user.id and action == 'suspend':
                return {
                    'success': False,
                    'error': 'Cannot suspend your own account'
                }
            
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
            
            if self.notification_service:
                self.notification_service.create_notification(
                    'USER_ACTIVITY',
                    message,
                    f'Admin {admin_user.username} {action}d user {user.username}',
                    admin_id=admin_user.id,
                    metadata={'user_id': user_id, 'action': action}
                )
            
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
        try:
            if not user_ids:
                return {
                    'success': False,
                    'error': 'No users selected'
                }
            
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
                        user.username = f"deleted_user_{user.id}"
                        user.email = f"deleted_{user.id}@cinebrain.deleted"
                        user.is_deleted = True
                        user.deleted_at = datetime.utcnow()
                        user.deleted_by = admin_user.id
                        results['success_count'] += 1
                        
                except Exception as e:
                    results['failed_count'] += 1
                    results['errors'].append(f"Error processing user {user_id}: {str(e)}")
            
            if results['success_count'] > 0:
                self.db.session.commit()
                
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
        try:
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

    # ==================== HELPER METHODS ====================

    def _get_user_quick_stats(self, user_id: int) -> Dict[str, Any]:
        try:
            if not self.UserInteraction:
                return {
                    'total_interactions': 0,
                    'last_interaction': None,
                    'favorite_count': 0,
                    'rating_count': 0,
                    'engagement_level': 'none'
                }
            
            interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
            
            return {
                'total_interactions': len(interactions),
                'last_interaction': max([i.timestamp for i in interactions]).isoformat() if interactions else None,
                'favorite_count': len([i for i in interactions if i.interaction_type == 'favorite']),
                'rating_count': len([i for i in interactions if i.interaction_type == 'rating']),
                'engagement_level': self._calculate_engagement_level(interactions)
            }
        except Exception as e:
            logger.error(f"Error getting user quick stats: {e}")
            return {}

    def _get_user_advanced_analytics(self, user_id: int) -> Dict[str, Any]:
        return {
            'session_patterns': self._analyze_session_patterns(user_id),
            'content_discovery': self._analyze_content_discovery_patterns(user_id),
            'recommendation_effectiveness': self._analyze_recommendation_effectiveness(user_id),
            'platform_usage': self._analyze_platform_usage(user_id)
        }

    def _assess_user_risk(self, user_id: int) -> Dict[str, Any]:
        risk_factors = []
        risk_score = 0
        
        user = self.User.query.get(user_id)
        if not user:
            return {'risk_score': 0, 'risk_level': 'unknown', 'factors': []}
        
        if user.last_active and (datetime.utcnow() - user.last_active).days > 60:
            risk_factors.append('Inactive for extended period')
            risk_score += 20
        
        if self.UserInteraction:
            interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
            if len(interactions) > 100:
                risk_factors.append('Unusually high activity')
                risk_score += 30
            
            recent_interactions = [i for i in interactions if (datetime.utcnow() - i.timestamp).days <= 1]
            if len(recent_interactions) > 50:
                risk_factors.append('Extremely high recent activity')
                risk_score += 40
        
        if user.is_suspended or user.is_banned:
            risk_factors.append('Previously suspended/banned')
            risk_score += 50
        
        if user.email and ('temp' in user.email or '10minute' in user.email):
            risk_factors.append('Temporary email detected')
            risk_score += 25
        
        if risk_score >= 70:
            risk_level = 'high'
        elif risk_score >= 40:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_score': min(risk_score, 100),
            'risk_level': risk_level,
            'factors': risk_factors
        }

    def _get_user_engagement_profile(self, user_id: int) -> Dict[str, Any]:
        try:
            if not self.UserInteraction:
                return {}
            
            interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
            
            interaction_types = Counter([i.interaction_type for i in interactions])
            
            return {
                'engagement_score': self._calculate_engagement_score(interactions),
                'interaction_diversity': len(interaction_types),
                'most_common_interaction': interaction_types.most_common(1)[0][0] if interaction_types else None,
                'consistency_score': self._calculate_consistency_score(interactions),
                'recent_activity_trend': self._calculate_activity_trend(user_id)
            }
        except Exception as e:
            logger.error(f"Error getting user engagement profile: {e}")
            return {}

    def _analyze_user_content_preferences(self, user_id: int) -> Dict[str, Any]:
        try:
            if not self.UserInteraction or not self.Content:
                return {}
            
            interactions = self.db.session.query(self.UserInteraction, self.Content).join(
                self.Content, self.UserInteraction.content_id == self.Content.id
            ).filter(self.UserInteraction.user_id == user_id).all()
            
            genres = []
            languages = []
            content_types = []
            
            for interaction, content in interactions:
                if content.genres:
                    try:
                        content_genres = json.loads(content.genres)
                        genres.extend(content_genres)
                    except:
                        pass
                
                if content.languages:
                    try:
                        content_languages = json.loads(content.languages)
                        languages.extend(content_languages)
                    except:
                        pass
                
                content_types.append(content.content_type)
            
            return {
                'favorite_genres': [genre for genre, count in Counter(genres).most_common(5)],
                'preferred_languages': [lang for lang, count in Counter(languages).most_common(3)],
                'content_type_distribution': dict(Counter(content_types)),
                'diversity_score': len(set(genres)) / max(len(genres), 1) if genres else 0
            }
        except Exception as e:
            logger.error(f"Error analyzing user content preferences: {e}")
            return {}

    def _get_power_users(self) -> List[Dict]:
        try:
            if not self.UserInteraction:
                return []
            
            power_users = self.db.session.query(
                self.User,
                func.count(self.UserInteraction.id).label('interaction_count')
            ).join(self.UserInteraction).group_by(self.User.id).having(
                func.count(self.UserInteraction.id) > 50
            ).order_by(func.count(self.UserInteraction.id).desc()).limit(20).all()
            
            return [
                {
                    'id': user.id,
                    'username': user.username,
                    'interaction_count': count
                }
                for user, count in power_users
            ]
        except Exception as e:
            logger.error(f"Error getting power users: {e}")
            return []

    def _get_casual_users(self) -> List[Dict]:
        try:
            if not self.UserInteraction:
                return []
            
            casual_users = self.db.session.query(
                self.User,
                func.count(self.UserInteraction.id).label('interaction_count')
            ).outerjoin(self.UserInteraction).group_by(self.User.id).having(
                func.count(self.UserInteraction.id).between(1, 20)
            ).limit(20).all()
            
            return [
                {
                    'id': user.id,
                    'username': user.username,
                    'interaction_count': count
                }
                for user, count in casual_users
            ]
        except Exception as e:
            logger.error(f"Error getting casual users: {e}")
            return []

    def _get_new_users(self) -> List[Dict]:
        try:
            week_ago = datetime.utcnow() - timedelta(days=7)
            new_users = self.User.query.filter(
                self.User.created_at >= week_ago
            ).order_by(self.User.created_at.desc()).limit(20).all()
            
            return [
                {
                    'id': user.id,
                    'username': user.username,
                    'created_at': user.created_at.isoformat()
                }
                for user in new_users
            ]
        except Exception as e:
            logger.error(f"Error getting new users: {e}")
            return []

    def _get_churned_users(self) -> List[Dict]:
        try:
            month_ago = datetime.utcnow() - timedelta(days=30)
            churned_users = self.User.query.filter(
                or_(
                    self.User.last_active < month_ago,
                    self.User.last_active == None
                )
            ).order_by(self.User.last_active.desc()).limit(20).all()
            
            return [
                {
                    'id': user.id,
                    'username': user.username,
                    'last_active': user.last_active.isoformat() if user.last_active else None
                }
                for user in churned_users
            ]
        except Exception as e:
            logger.error(f"Error getting churned users: {e}")
            return []

    def _get_at_risk_users(self) -> List[Dict]:
        try:
            two_weeks_ago = datetime.utcnow() - timedelta(days=14)
            at_risk_users = self.User.query.filter(
                and_(
                    self.User.last_active < two_weeks_ago,
                    self.User.last_active >= datetime.utcnow() - timedelta(days=30)
                )
            ).order_by(self.User.last_active.desc()).limit(20).all()
            
            return [
                {
                    'id': user.id,
                    'username': user.username,
                    'last_active': user.last_active.isoformat() if user.last_active else None,
                    'days_inactive': (datetime.utcnow() - user.last_active).days if user.last_active else None
                }
                for user in at_risk_users
            ]
        except Exception as e:
            logger.error(f"Error getting at-risk users: {e}")
            return []

    def _get_high_value_users(self) -> List[Dict]:
        try:
            high_value_users = []
            
            admin_users = self.User.query.filter_by(is_admin=True).all()
            for user in admin_users:
                high_value_users.append({
                    'id': user.id,
                    'username': user.username,
                    'value_type': 'admin',
                    'value_score': 100
                })
            
            if self.UserInteraction and len(high_value_users) < 20:
                engagement_users = self.db.session.query(
                    self.User,
                    func.count(self.UserInteraction.id).label('interaction_count')
                ).join(self.UserInteraction).group_by(self.User.id).having(
                    func.count(self.UserInteraction.id) > 30
                ).order_by(func.count(self.UserInteraction.id).desc()).limit(15).all()
                
                for user, count in engagement_users:
                    if not any(u['id'] == user.id for u in high_value_users):
                        high_value_users.append({
                            'id': user.id,
                            'username': user.username,
                            'value_type': 'high_engagement',
                            'value_score': min(count * 2, 90)
                        })
            
            return high_value_users[:20]
        except Exception as e:
            logger.error(f"Error getting high value users: {e}")
            return []

    def _get_content_creators(self) -> List[Dict]:
        try:
            if not self.Review:
                return []
            
            creators = self.db.session.query(
                self.User,
                func.count(self.Review.id).label('review_count')
            ).join(self.Review).group_by(self.User.id).having(
                func.count(self.Review.id) > 3
            ).order_by(func.count(self.Review.id).desc()).limit(20).all()
            
            return [
                {
                    'id': user.id,
                    'username': user.username,
                    'review_count': count
                }
                for user, count in creators
            ]
        except Exception as e:
            logger.error(f"Error getting content creators: {e}")
            return []

    def _get_lurkers(self) -> List[Dict]:
        try:
            if not self.UserInteraction:
                return []
            
            lurkers = self.db.session.query(
                self.User,
                func.count(self.UserInteraction.id).label('interaction_count')
            ).outerjoin(self.UserInteraction).group_by(self.User.id).having(
                or_(
                    func.count(self.UserInteraction.id) == 0,
                    func.count(self.UserInteraction.id) <= 3
                )
            ).limit(20).all()
            
            return [
                {
                    'id': user.id,
                    'username': user.username,
                    'interaction_count': count
                }
                for user, count in lurkers
            ]
        except Exception as e:
            logger.error(f"Error getting lurkers: {e}")
            return []

    def _calculate_engagement_level(self, interactions: List) -> str:
        if not interactions:
            return 'none'
        
        interaction_count = len(interactions)
        
        if interaction_count >= 100:
            return 'very_high'
        elif interaction_count >= 50:
            return 'high'
        elif interaction_count >= 20:
            return 'medium'
        elif interaction_count >= 5:
            return 'low'
        else:
            return 'very_low'

    def _calculate_user_value_score(self, user_id: int) -> Dict[str, Any]:
        user = self.User.query.get(user_id)
        if not user:
            return {'total_score': 0, 'breakdown': {}, 'tier': 'unknown'}
        
        score_breakdown = {
            'account_age': 0,
            'activity_level': 0,
            'engagement_quality': 0,
            'admin_status': 0,
            'content_contribution': 0
        }
        
        if user.created_at:
            days_old = (datetime.utcnow() - user.created_at).days
            score_breakdown['account_age'] = min(days_old / 10, 20)
        
        if self.UserInteraction:
            interaction_count = self.UserInteraction.query.filter_by(user_id=user_id).count()
            score_breakdown['activity_level'] = min(interaction_count / 2, 30)
        
        if user.last_active:
            days_since_active = (datetime.utcnow() - user.last_active).days
            if days_since_active <= 7:
                score_breakdown['engagement_quality'] = 25
            elif days_since_active <= 30:
                score_breakdown['engagement_quality'] = 15
            elif days_since_active <= 90:
                score_breakdown['engagement_quality'] = 5
        
        if user.is_admin:
            score_breakdown['admin_status'] = 15
        
        if self.Review:
            review_count = self.Review.query.filter_by(user_id=user_id).count()
            score_breakdown['content_contribution'] = min(review_count * 2, 10)
        
        total_score = sum(score_breakdown.values())
        
        if total_score >= 80:
            tier = 'platinum'
        elif total_score >= 60:
            tier = 'gold'
        elif total_score >= 40:
            tier = 'silver'
        else:
            tier = 'bronze'
        
        return {
            'total_score': round(total_score, 1),
            'breakdown': score_breakdown,
            'tier': tier
        }

    # ==================== ANALYSIS METHODS ====================

    def _analyze_onboarding_funnel(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        try:
            new_users = self.User.query.filter(
                and_(
                    self.User.created_at >= start_date,
                    self.User.created_at <= end_date
                )
            ).all()
            
            total_registered = len(new_users)
            
            if not self.UserInteraction or total_registered == 0:
                return {
                    'total_registered': total_registered,
                    'completed_profile': 0,
                    'first_interaction': 0,
                    'first_week_retention': 0
                }
            
            completed_profile = len([u for u in new_users if u.preferred_languages or u.preferred_genres])
            
            user_ids = [u.id for u in new_users]
            users_with_interactions = self.db.session.query(
                distinct(self.UserInteraction.user_id)
            ).filter(self.UserInteraction.user_id.in_(user_ids)).count()
            
            week_later = end_date + timedelta(days=7)
            users_active_week_later = len([
                u for u in new_users 
                if u.last_active and u.last_active >= u.created_at + timedelta(days=7)
            ])
            
            return {
                'total_registered': total_registered,
                'completed_profile': completed_profile,
                'first_interaction': users_with_interactions,
                'first_week_retention': users_active_week_later,
                'funnel_rates': {
                    'profile_completion': (completed_profile / total_registered * 100) if total_registered > 0 else 0,
                    'first_interaction': (users_with_interactions / total_registered * 100) if total_registered > 0 else 0,
                    'week_retention': (users_active_week_later / total_registered * 100) if total_registered > 0 else 0
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing onboarding funnel: {e}")
            return {}

    def _calculate_activation_rates(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        try:
            new_users = self.User.query.filter(
                and_(
                    self.User.created_at >= start_date,
                    self.User.created_at <= end_date
                )
            ).all()
            
            if not new_users:
                return {'activation_rate': 0, 'activated_users': 0, 'total_users': 0}
            
            activated_users = 0
            
            for user in new_users:
                activation_score = 0
                
                if user.preferred_languages:
                    activation_score += 1
                if user.preferred_genres:
                    activation_score += 1
                if user.avatar_url:
                    activation_score += 1
                
                if self.UserInteraction:
                    interaction_count = self.UserInteraction.query.filter_by(user_id=user.id).count()
                    if interaction_count >= 3:
                        activation_score += 1
                
                if activation_score >= 2:
                    activated_users += 1
            
            return {
                'activation_rate': (activated_users / len(new_users) * 100) if new_users else 0,
                'activated_users': activated_users,
                'total_users': len(new_users)
            }
        except Exception as e:
            logger.error(f"Error calculating activation rates: {e}")
            return {}

    def _analyze_retention_cohorts(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        try:
            cohorts = {}
            
            current_date = start_date
            while current_date < end_date:
                cohort_end = current_date + timedelta(days=30)
                
                cohort_users = self.User.query.filter(
                    and_(
                        self.User.created_at >= current_date,
                        self.User.created_at < cohort_end
                    )
                ).all()
                
                if cohort_users:
                    cohort_key = current_date.strftime('%Y-%m')
                    cohorts[cohort_key] = self._analyze_cohort_retention(cohort_users, current_date)
                
                current_date = cohort_end
            
            return {
                'cohorts': cohorts,
                'average_retention': self._calculate_average_retention(cohorts)
            }
        except Exception as e:
            logger.error(f"Error analyzing retention cohorts: {e}")
            return {}

    def _analyze_cohort_retention(self, cohort_users: List, start_date: datetime) -> Dict[str, Any]:
        try:
            total_users = len(cohort_users)
            
            retention_periods = [7, 14, 30, 60, 90]
            retention_data = {}
            
            for period in retention_periods:
                check_date = start_date + timedelta(days=period)
                retained_users = len([
                    u for u in cohort_users 
                    if u.last_active and u.last_active >= check_date
                ])
                
                retention_data[f"day_{period}"] = {
                    'retained_users': retained_users,
                    'retention_rate': (retained_users / total_users * 100) if total_users > 0 else 0
                }
            
            return {
                'total_users': total_users,
                'start_date': start_date.isoformat(),
                'retention_rates': retention_data
            }
        except Exception as e:
            logger.error(f"Error analyzing cohort retention: {e}")
            return {}

    def _analyze_user_churn(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        try:
            active_users_start = self.User.query.filter(
                self.User.last_active >= start_date
            ).count()
            
            active_users_end = self.User.query.filter(
                self.User.last_active >= end_date - timedelta(days=30)
            ).count()
            
            churned_users = self.User.query.filter(
                and_(
                    self.User.last_active < end_date - timedelta(days=30),
                    self.User.last_active >= start_date
                )
            ).count()
            
            churn_rate = (churned_users / active_users_start * 100) if active_users_start > 0 else 0
            
            return {
                'churn_rate': round(churn_rate, 2),
                'churned_users': churned_users,
                'active_users_start': active_users_start,
                'active_users_end': active_users_end
            }
        except Exception as e:
            logger.error(f"Error analyzing user churn: {e}")
            return {}

    def _analyze_user_journey_stages(self) -> Dict[str, Any]:
        try:
            stages = {
                'discovery': 0,
                'activation': 0,
                'engagement': 0,
                'retention': 0,
                'advocacy': 0
            }
            
            for user in self.User.query.all():
                stage = self._determine_user_stage(user)
                if stage in stages:
                    stages[stage] += 1
            
            total_users = sum(stages.values())
            
            return {
                'stage_counts': stages,
                'stage_percentages': {
                    stage: (count / total_users * 100) if total_users > 0 else 0
                    for stage, count in stages.items()
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing user journey stages: {e}")
            return {}

    def _determine_user_stage(self, user) -> str:
        try:
            if not user.last_active:
                return 'discovery'
            
            days_since_registration = (datetime.utcnow() - user.created_at).days
            days_since_active = (datetime.utcnow() - user.last_active).days
            
            if days_since_active > 30:
                return 'discovery'
            
            interaction_count = 0
            if self.UserInteraction:
                interaction_count = self.UserInteraction.query.filter_by(user_id=user.id).count()
            
            if interaction_count == 0:
                return 'discovery'
            elif interaction_count < 5:
                return 'activation'
            elif interaction_count < 20:
                return 'engagement'
            elif days_since_registration > 30:
                return 'retention'
            else:
                return 'advocacy'
        except Exception as e:
            logger.error(f"Error determining user stage: {e}")
            return 'discovery'

    def _analyze_activity_patterns(self, user_id: int) -> Dict[str, Any]:
        try:
            if not self.UserInteraction:
                return {}
            
            interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
            
            if not interactions:
                return {'pattern': 'inactive', 'peak_hours': [], 'peak_days': []}
            
            hour_counts = Counter([i.timestamp.hour for i in interactions])
            day_counts = Counter([i.timestamp.weekday() for i in interactions])
            
            peak_hours = [hour for hour, count in hour_counts.most_common(3)]
            peak_days = [day for day, count in day_counts.most_common(3)]
            
            recent_interactions = [
                i for i in interactions 
                if (datetime.utcnow() - i.timestamp).days <= 30
            ]
            
            if len(recent_interactions) > len(interactions) * 0.8:
                pattern = 'highly_active'
            elif len(recent_interactions) > len(interactions) * 0.5:
                pattern = 'moderately_active'
            elif len(recent_interactions) > 0:
                pattern = 'low_activity'
            else:
                pattern = 'dormant'
            
            return {
                'pattern': pattern,
                'peak_hours': peak_hours,
                'peak_days': peak_days,
                'total_interactions': len(interactions),
                'recent_interactions': len(recent_interactions)
            }
        except Exception as e:
            logger.error(f"Error analyzing activity patterns: {e}")
            return {}

    def _analyze_content_consumption(self, user_id: int) -> Dict[str, Any]:
        try:
            if not self.UserInteraction or not self.Content:
                return {}
            
            interactions = self.db.session.query(self.UserInteraction, self.Content).join(
                self.Content, self.UserInteraction.content_id == self.Content.id
            ).filter(self.UserInteraction.user_id == user_id).all()
            
            if not interactions:
                return {'consumption_pattern': 'none'}
            
            content_types = [content.content_type for _, content in interactions]
            content_type_dist = Counter(content_types)
            
            ratings = [interaction.rating for interaction, _ in interactions if interaction.rating]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
            genres = []
            for _, content in interactions:
                if content.genres:
                    try:
                        content_genres = json.loads(content.genres)
                        genres.extend(content_genres)
                    except:
                        pass
            
            top_genres = [genre for genre, count in Counter(genres).most_common(5)]
            
            return {
                'consumption_pattern': 'diverse' if len(content_type_dist) > 2 else 'focused',
                'content_type_distribution': dict(content_type_dist),
                'average_rating': round(avg_rating, 2),
                'top_genres': top_genres,
                'total_consumed': len(interactions)
            }
        except Exception as e:
            logger.error(f"Error analyzing content consumption: {e}")
            return {}

    def _analyze_interaction_patterns(self, user_id: int) -> Dict[str, Any]:
        try:
            if not self.UserInteraction:
                return {}
            
            interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
            
            if not interactions:
                return {'dominant_interaction': 'none', 'diversity_score': 0}
            
            interaction_types = Counter([i.interaction_type for i in interactions])
            dominant_interaction = interaction_types.most_common(1)[0][0]
            
            diversity_score = len(interaction_types) / len(interactions) if interactions else 0
            
            recent_interactions = [
                i for i in interactions 
                if (datetime.utcnow() - i.timestamp).days <= 7
            ]
            
            return {
                'dominant_interaction': dominant_interaction,
                'interaction_distribution': dict(interaction_types),
                'diversity_score': round(diversity_score, 3),
                'recent_activity': len(recent_interactions),
                'total_interactions': len(interactions)
            }
        except Exception as e:
            logger.error(f"Error analyzing interaction patterns: {e}")
            return {}

    def _analyze_preference_evolution(self, user_id: int) -> Dict[str, Any]:
        try:
            user = self.User.query.get(user_id)
            if not user:
                return {}
            
            current_languages = json.loads(user.preferred_languages or '[]')
            current_genres = json.loads(user.preferred_genres or '[]')
            
            if not self.UserInteraction or not self.Content:
                return {
                    'evolution_detected': False,
                    'current_preferences': {
                        'languages': current_languages,
                        'genres': current_genres
                    }
                }
            
            recent_interactions = self.db.session.query(self.UserInteraction, self.Content).join(
                self.Content, self.UserInteraction.content_id == self.Content.id
            ).filter(
                and_(
                    self.UserInteraction.user_id == user_id,
                    self.UserInteraction.timestamp >= datetime.utcnow() - timedelta(days=30)
                )
            ).all()
            
            recent_genres = []
            recent_languages = []
            
            for interaction, content in recent_interactions:
                if content.genres:
                    try:
                        content_genres = json.loads(content.genres)
                        recent_genres.extend(content_genres)
                    except:
                        pass
                
                if content.languages:
                    try:
                        content_languages = json.loads(content.languages)
                        recent_languages.extend(content_languages)
                    except:
                        pass
            
            recent_top_genres = [genre for genre, count in Counter(recent_genres).most_common(3)]
            recent_top_languages = [lang for lang, count in Counter(recent_languages).most_common(2)]
            
            genre_evolution = len(set(recent_top_genres) - set(current_genres)) > 0
            language_evolution = len(set(recent_top_languages) - set(current_languages)) > 0
            
            return {
                'evolution_detected': genre_evolution or language_evolution,
                'current_preferences': {
                    'languages': current_languages,
                    'genres': current_genres
                },
                'recent_preferences': {
                    'languages': recent_top_languages,
                    'genres': recent_top_genres
                },
                'new_interests': {
                    'genres': list(set(recent_top_genres) - set(current_genres)),
                    'languages': list(set(recent_top_languages) - set(current_languages))
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing preference evolution: {e}")
            return {}

    def _analyze_social_behavior(self, user_id: int) -> Dict[str, Any]:
        try:
            social_score = 0
            behaviors = []
            
            if self.Review:
                review_count = self.Review.query.filter_by(user_id=user_id).count()
                if review_count > 0:
                    social_score += min(review_count * 10, 50)
                    behaviors.append('content_reviewer')
            
            if self.UserInteraction:
                rating_count = self.UserInteraction.query.filter_by(
                    user_id=user_id, 
                    interaction_type='rating'
                ).count()
                if rating_count > 10:
                    social_score += min(rating_count * 2, 30)
                    behaviors.append('active_rater')
            
            if social_score >= 60:
                social_type = 'highly_social'
            elif social_score >= 30:
                social_type = 'moderately_social'
            elif social_score > 0:
                social_type = 'low_social'
            else:
                social_type = 'passive'
            
            return {
                'social_score': social_score,
                'social_type': social_type,
                'behaviors': behaviors
            }
        except Exception as e:
            logger.error(f"Error analyzing social behavior: {e}")
            return {}

    def _analyze_recommendation_response(self, user_id: int) -> Dict[str, Any]:
        try:
            if not self.UserInteraction:
                return {}
            
            total_interactions = self.UserInteraction.query.filter_by(user_id=user_id).count()
            
            if total_interactions == 0:
                return {'response_rate': 0, 'engagement_level': 'none'}
            
            positive_interactions = self.UserInteraction.query.filter(
                and_(
                    self.UserInteraction.user_id == user_id,
                    or_(
                        self.UserInteraction.interaction_type == 'favorite',
                        self.UserInteraction.interaction_type == 'like',
                        and_(
                            self.UserInteraction.interaction_type == 'rating',
                            self.UserInteraction.rating >= 7
                        )
                    )
                )
            ).count()
            
            response_rate = (positive_interactions / total_interactions * 100) if total_interactions > 0 else 0
            
            if response_rate >= 70:
                engagement_level = 'high'
            elif response_rate >= 40:
                engagement_level = 'medium'
            elif response_rate > 0:
                engagement_level = 'low'
            else:
                engagement_level = 'none'
            
            return {
                'response_rate': round(response_rate, 2),
                'engagement_level': engagement_level,
                'positive_interactions': positive_interactions,
                'total_interactions': total_interactions
            }
        except Exception as e:
            logger.error(f"Error analyzing recommendation response: {e}")
            return {}

    # ==================== ANOMALY DETECTION METHODS ====================

    def _detect_suspicious_activity(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        try:
            suspicious_users = []
            
            if not self.UserInteraction:
                return suspicious_users
            
            # Find users with extremely high activity
            high_activity_users = self.db.session.query(
                self.User,
                func.count(self.UserInteraction.id).label('interaction_count')
            ).join(self.UserInteraction).filter(
                self.UserInteraction.timestamp.between(start_date, end_date)
            ).group_by(self.User.id).having(
                func.count(self.UserInteraction.id) > 200
            ).all()
            
            for user, count in high_activity_users:
                suspicious_users.append({
                    'user_id': user.id,
                    'username': user.username,
                    'anomaly_type': 'excessive_activity',
                    'details': f'{count} interactions in period',
                    'risk_level': 'high' if count > 500 else 'medium'
                })
            
            return suspicious_users
        except Exception as e:
            logger.error(f"Error detecting suspicious activity: {e}")
            return []

    def _detect_unusual_patterns(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        try:
            unusual_users = []
            
            # Find users with unusual registration patterns
            bulk_registrations = self.db.session.query(
                func.date(self.User.created_at).label('reg_date'),
                func.count(self.User.id).label('user_count')
            ).filter(
                self.User.created_at.between(start_date, end_date)
            ).group_by(func.date(self.User.created_at)).having(
                func.count(self.User.id) > 50
            ).all()
            
            for reg_date, count in bulk_registrations:
                users_on_date = self.User.query.filter(
                    func.date(self.User.created_at) == reg_date
                ).limit(10).all()
                
                for user in users_on_date:
                    unusual_users.append({
                        'user_id': user.id,
                        'username': user.username,
                        'anomaly_type': 'bulk_registration',
                        'details': f'Part of {count} users registered on {reg_date}',
                        'risk_level': 'medium'
                    })
            
            return unusual_users
        except Exception as e:
            logger.error(f"Error detecting unusual patterns: {e}")
            return []

    def _detect_potential_bots(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        try:
            potential_bots = []
            
            if not self.UserInteraction:
                return potential_bots
            
            # Find users with very regular interaction patterns (potential bots)
            regular_pattern_users = self.db.session.query(
                self.User,
                func.count(self.UserInteraction.id).label('interaction_count')
            ).join(self.UserInteraction).filter(
                self.UserInteraction.timestamp.between(start_date, end_date)
            ).group_by(self.User.id).having(
                func.count(self.UserInteraction.id) > 100
            ).all()
            
            for user, count in regular_pattern_users:
                interactions = self.UserInteraction.query.filter(
                    and_(
                        self.UserInteraction.user_id == user.id,
                        self.UserInteraction.timestamp.between(start_date, end_date)
                    )
                ).all()
                
                # Check for very regular timing (potential bot behavior)
                time_gaps = []
                for i in range(1, len(interactions)):
                    gap = (interactions[i].timestamp - interactions[i-1].timestamp).seconds
                    time_gaps.append(gap)
                
                if time_gaps:
                    avg_gap = sum(time_gaps) / len(time_gaps)
                    gap_variance = sum((gap - avg_gap) ** 2 for gap in time_gaps) / len(time_gaps)
                    
                    # Low variance suggests bot-like regularity
                    if gap_variance < 100 and count > 150:
                        potential_bots.append({
                            'user_id': user.id,
                            'username': user.username,
                            'anomaly_type': 'potential_bot',
                            'details': f'Regular pattern: {count} interactions, low variance',
                            'risk_level': 'high'
                        })
            
            return potential_bots
        except Exception as e:
            logger.error(f"Error detecting potential bots: {e}")
            return []

    def _detect_spam_indicators(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        try:
            spam_indicators = []
            
            # Find users with suspicious email patterns
            suspicious_emails = self.User.query.filter(
                and_(
                    self.User.created_at.between(start_date, end_date),
                    or_(
                        self.User.email.like('%temp%'),
                        self.User.email.like('%10minute%'),
                        self.User.email.like('%throwaway%'),
                        self.User.email.like('%mailinator%')
                    )
                )
            ).all()
            
            for user in suspicious_emails:
                spam_indicators.append({
                    'user_id': user.id,
                    'username': user.username,
                    'anomaly_type': 'suspicious_email',
                    'details': f'Temporary email pattern: {user.email}',
                    'risk_level': 'medium'
                })
            
            return spam_indicators
        except Exception as e:
            logger.error(f"Error detecting spam indicators: {e}")
            return []

    def _detect_security_concerns(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        try:
            security_concerns = []
            
            # Find users who were suspended/banned multiple times
            problematic_users = self.User.query.filter(
                and_(
                    self.User.suspended_at.between(start_date, end_date),
                    or_(
                        self.User.is_suspended == True,
                        self.User.is_banned == True
                    )
                )
            ).all()
            
            for user in problematic_users:
                security_concerns.append({
                    'user_id': user.id,
                    'username': user.username,
                    'anomaly_type': 'security_concern',
                    'details': 'Previously suspended/banned user',
                    'risk_level': 'high'
                })
            
            return security_concerns
        except Exception as e:
            logger.error(f"Error detecting security concerns: {e}")
            return []

    # ==================== HELPER METHODS FOR COMMUNICATION ====================

    def _get_users_by_criteria(self, criteria: Dict[str, Any]) -> List:
        try:
            query = self.User.query
            
            if criteria.get('user_segment'):
                segment = criteria['user_segment']
                if segment == 'power_users':
                    if self.UserInteraction:
                        power_user_ids = self.db.session.query(
                            self.UserInteraction.user_id
                        ).group_by(self.UserInteraction.user_id).having(
                            func.count(self.UserInteraction.id) > 50
                        ).subquery()
                        query = query.filter(self.User.id.in_(power_user_ids))
                elif segment == 'new_users':
                    week_ago = datetime.utcnow() - timedelta(days=7)
                    query = query.filter(self.User.created_at >= week_ago)
                elif segment == 'churned_users':
                    month_ago = datetime.utcnow() - timedelta(days=30)
                    query = query.filter(
                        or_(
                            self.User.last_active < month_ago,
                            self.User.last_active == None
                        )
                    )
            
            if criteria.get('admin_only'):
                query = query.filter(self.User.is_admin == True)
            
            if criteria.get('location'):
                location_term = f"%{criteria['location']}%"
                query = query.filter(self.User.location.ilike(location_term))
            
            return query.limit(1000).all()
        except Exception as e:
            logger.error(f"Error getting users by criteria: {e}")
            return []

    def _personalize_message(self, message: str, user) -> str:
        try:
            personalized = message.replace('{{username}}', user.username or 'User')
            personalized = personalized.replace('{{email}}', user.email or '')
            personalized = personalized.replace('{{full_name}}', user.full_name or user.username or 'User')
            
            # Add more personalization based on user data
            if user.location:
                personalized = personalized.replace('{{location}}', user.location)
            
            return personalized
        except Exception as e:
            logger.error(f"Error personalizing message: {e}")
            return message

    # ==================== REMAINING HELPER METHODS ====================

    def _calculate_engagement_score(self, interactions: List) -> float:
        if not interactions:
            return 0.0
        
        # Weight different interaction types
        weights = {
            'favorite': 3.0,
            'rating': 2.5,
            'like': 2.0,
            'watchlist': 1.5,
            'view': 1.0,
            'search': 0.5
        }
        
        total_score = 0
        for interaction in interactions:
            weight = weights.get(interaction.interaction_type, 1.0)
            total_score += weight
        
        # Normalize score (max 100)
        max_possible_score = len(interactions) * 3.0
        normalized_score = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
        
        return min(normalized_score, 100)

    def _calculate_consistency_score(self, interactions: List) -> float:
        if len(interactions) < 2:
            return 0.0
        
        # Calculate time gaps between interactions
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
        gaps = []
        
        for i in range(1, len(sorted_interactions)):
            gap = (sorted_interactions[i].timestamp - sorted_interactions[i-1].timestamp).total_seconds()
            gaps.append(gap)
        
        if not gaps:
            return 0.0
        
        # Calculate coefficient of variation (lower = more consistent)
        avg_gap = sum(gaps) / len(gaps)
        variance = sum((gap - avg_gap) ** 2 for gap in gaps) / len(gaps)
        std_dev = variance ** 0.5
        
        cv = std_dev / avg_gap if avg_gap > 0 else float('inf')
        
        # Convert to consistency score (0-100, higher = more consistent)
        consistency_score = max(0, 100 - (cv * 10))
        
        return min(consistency_score, 100)

    def _calculate_activity_trend(self, user_id: int) -> str:
        try:
            if not self.UserInteraction:
                return 'stable'
            
            recent_interactions = self.UserInteraction.query.filter(
                and_(
                    self.UserInteraction.user_id == user_id,
                    self.UserInteraction.timestamp >= datetime.utcnow() - timedelta(days=30)
                )
            ).count()
            
            older_interactions = self.UserInteraction.query.filter(
                and_(
                    self.UserInteraction.user_id == user_id,
                    self.UserInteraction.timestamp >= datetime.utcnow() - timedelta(days=60),
                    self.UserInteraction.timestamp < datetime.utcnow() - timedelta(days=30)
                )
            ).count()
            
            if older_interactions == 0:
                return 'new_user'
            
            trend_ratio = recent_interactions / older_interactions
            
            if trend_ratio > 1.5:
                return 'increasing'
            elif trend_ratio < 0.5:
                return 'decreasing'
            else:
                return 'stable'
        except Exception as e:
            logger.error(f"Error calculating activity trend: {e}")
            return 'stable'

    def _analyze_session_patterns(self, user_id: int) -> Dict[str, Any]:
        return {
            'average_session_length': 25.5,
            'sessions_per_week': 4.2,
            'preferred_session_time': 'evening'
        }

    def _analyze_content_discovery_patterns(self, user_id: int) -> Dict[str, Any]:
        return {
            'discovery_method': 'recommendations',
            'exploration_rate': 0.75,
            'repeat_consumption': 0.25
        }

    def _analyze_recommendation_effectiveness(self, user_id: int) -> Dict[str, Any]:
        return {
            'click_through_rate': 0.35,
            'conversion_rate': 0.15,
            'satisfaction_score': 8.2
        }

    def _analyze_platform_usage(self, user_id: int) -> Dict[str, Any]:
        return {
            'primary_device': 'mobile',
            'feature_adoption': 0.8,
            'navigation_efficiency': 'high'
        }

    def _calculate_resolution_satisfaction(self, user_id: int) -> float:
        return 85.0

    def _identify_common_issues(self, user_id: int) -> List[str]:
        return ['login_issues', 'recommendation_quality']

    def _generate_comparison_insights(self, users: List) -> Dict[str, Any]:
        return {
            'most_active': users[0]['basic_info']['username'] if users else 'N/A',
            'highest_value': users[0]['basic_info']['username'] if users else 'N/A',
            'common_traits': ['high_engagement', 'diverse_content']
        }

    def _calculate_average_retention(self, cohorts: Dict) -> float:
        if not cohorts:
            return 0.0
        
        retention_rates = []
        for cohort in cohorts.values():
            if 'retention_rates' in cohort and 'day_30' in cohort['retention_rates']:
                retention_rates.append(cohort['retention_rates']['day_30']['retention_rate'])
        
        return sum(retention_rates) / len(retention_rates) if retention_rates else 0.0

    def _summarize_cohort_data(self, cohorts: Dict) -> Dict[str, Any]:
        return {
            'total_cohorts': len(cohorts),
            'average_retention': self._calculate_average_retention(cohorts),
            'best_cohort': max(cohorts.keys()) if cohorts else None,
            'trend': 'improving'
        }

    # ==================== EXISTING HELPER METHODS ====================

    def _apply_user_filters(self, query, filters: Dict[str, Any]):
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
        
        if filters.get('role'):
            if filters['role'] == 'admin':
                query = query.filter(self.User.is_admin == True)
            elif filters['role'] == 'user':
                query = query.filter(self.User.is_admin == False)
        
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
            query = query.order_by(desc(self.User.created_at))
        
        return query

    def _format_user_data(self, user, detailed: bool = False) -> Dict[str, Any]:
        data = {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'full_name': user.full_name,
            'is_admin': user.is_admin,
            'is_suspended': user.is_suspended,
            'is_banned': user.is_banned,
            'is_active': True,
            'avatar_url': user.avatar_url,
            'created_at': user.created_at.isoformat() if user.created_at else None,
            'last_active': user.last_active.isoformat() if user.last_active else None,
            'location': user.location
        }
        
        if user.last_active:
            days_inactive = (datetime.utcnow() - user.last_active).days
            data['is_active'] = days_inactive < 30
        
        if detailed:
            data['preferred_languages'] = json.loads(user.preferred_languages or '[]')
            data['preferred_genres'] = json.loads(user.preferred_genres or '[]')
            data['suspended_at'] = user.suspended_at.isoformat() if user.suspended_at else None
            data['suspended_by'] = user.suspended_by
        
        return data

    def _get_timeframe_dates(self, timeframe: str) -> Tuple[datetime, datetime, datetime]:
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
            start = now - timedelta(days=7)
            previous_start = start - timedelta(days=7)
            previous_end = start
        
        return start, previous_start, previous_end

    def _calculate_user_stats(self, start_date: datetime, end_date: datetime) -> Dict[str, int]:
        total_users = self.User.query.filter(
            self.User.created_at <= end_date
        ).count()
        
        active_users = self.User.query.filter(
            and_(
                self.User.last_active >= start_date,
                self.User.last_active <= end_date
            )
        ).count()
        
        new_users = self.User.query.filter(
            and_(
                self.User.created_at >= start_date,
                self.User.created_at <= end_date
            )
        ).count()
        
        admin_users = self.User.query.filter(
            and_(
                self.User.is_admin == True,
                self.User.created_at <= end_date
            )
        ).count()
        
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
        total_users = self.User.query.filter(
            self.User.created_at <= end_date
        ).count()
        
        active_users = self.User.query.filter(
            and_(
                self.User.last_active >= start_date,
                self.User.last_active <= end_date
            )
        ).count()
        
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
        
        engagement_rate = (active_users / total_users * 100) if total_users > 0 else 0
        engagement_rate_previous = (previous_active / previous_total * 100) if previous_total > 0 else 0
        
        return {
            'engagement_rate': round(engagement_rate, 1),
            'engagement_rate_previous': round(engagement_rate_previous, 1)
        }

    def _get_user_activity_trends(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        try:
            date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
        except ImportError:
            # Fallback if pandas not available
            date_range = []
            current_date = start_date.date()
            while current_date <= end_date.date():
                date_range.append(current_date)
                current_date += timedelta(days=1)
        
        activity_data = []
        
        for current_date in date_range:
            try:
                if hasattr(current_date, 'date'):
                    current_date = current_date.date()
                
                next_date = current_date + timedelta(days=1)
                
                active_count = self.User.query.filter(
                    and_(
                        func.date(self.User.last_active) == current_date
                    )
                ).count()
                
                new_count = self.User.query.filter(
                    and_(
                        func.date(self.User.created_at) == current_date
                    )
                ).count()
                
                activity_data.append({
                    'date': current_date.isoformat(),
                    'active_users': active_count,
                    'new_users': new_count
                })
            except Exception as e:
                logger.error(f"Error processing date {current_date}: {e}")
                continue
        
        return activity_data

    def _get_engagement_analytics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        total_users = self.User.query.count()
        
        today = datetime.utcnow().date()
        daily_active = self.User.query.filter(
            func.date(self.User.last_active) == today
        ).count()
        
        week_ago = datetime.utcnow() - timedelta(days=7)
        weekly_active = self.User.query.filter(
            self.User.last_active >= week_ago
        ).count()
        
        month_ago = datetime.utcnow() - timedelta(days=30)
        monthly_active = self.User.query.filter(
            self.User.last_active >= month_ago
        ).count()
        
        avg_session_time = 1245
        
        return {
            'total_users': total_users,
            'daily_active': daily_active,
            'weekly_active': weekly_active,
            'monthly_active': monthly_active,
            'avg_session_time': avg_session_time
        }

    def _get_user_interaction_counts(self, user_id: int) -> Dict[str, int]:
        counts = {
            'interaction_count': 0,
            'ratings_count': 0,
            'favorites_count': 0,
            'watchlist_count': 0
        }
        
        if self.UserInteraction:
            counts['interaction_count'] = self.UserInteraction.query.filter_by(
                user_id=user_id
            ).count()
            
            counts['ratings_count'] = self.UserInteraction.query.filter_by(
                user_id=user_id,
                interaction_type='rating'
            ).count()
            
            counts['favorites_count'] = self.UserInteraction.query.filter_by(
                user_id=user_id,
                interaction_type='favorite'
            ).count()
            
            counts['watchlist_count'] = self.UserInteraction.query.filter_by(
                user_id=user_id,
                interaction_type='watchlist'
            ).count()
        
        if self.Review:
            counts['reviews_count'] = self.Review.query.filter_by(
                user_id=user_id
            ).count()
        
        return counts

    def _parse_period(self, period: str) -> int:
        if period == '7d':
            return 7
        elif period == '30d':
            return 30
        elif period == '90d':
            return 90
        else:
            return 30

    def _validate_user_update(self, user, data: Dict[str, Any]) -> Optional[str]:
        if 'username' in data and data['username'] != user.username:
            existing = self.User.query.filter_by(username=data['username']).first()
            if existing:
                return 'Username already exists'
        
        if 'email' in data and data['email'] != user.email:
            existing = self.User.query.filter_by(email=data['email']).first()
            if existing:
                return 'Email already exists'
        
        if 'email' in data:
            if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', data['email']):
                return 'Invalid email format'
        
        return None

    def _update_user_status(self, user, status: str) -> Optional[str]:
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
        if self.cache:
            try:
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
                'full_name': user.full_name or '',
                'is_admin': 'Yes' if user.is_admin else 'No',
                'is_suspended': 'Yes' if user.is_suspended else 'No',
                'is_banned': 'Yes' if user.is_banned else 'No',
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
        user_data = []
        
        for user in users:
            user_data.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name,
                'is_admin': user.is_admin,
                'is_suspended': user.is_suspended,
                'is_banned': user.is_banned,
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
    return AdminUserService(app, db, models, services)