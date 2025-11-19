# admin/dashboard.py

from datetime import datetime, timedelta
from sqlalchemy import func, desc, and_, or_
from collections import defaultdict
import json
import logging
import os

logger = logging.getLogger(__name__)

class AdminDashboard:
    def __init__(self, app, db, models, services):
        self.app = app
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.AdminRecommendation = models['AdminRecommendation']
        
        # Support models (optional)
        self.SupportTicket = models.get('SupportTicket')
        self.SupportCategory = models.get('SupportCategory')
        self.TicketActivity = models.get('TicketActivity')
        self.ContactMessage = models.get('ContactMessage')
        self.IssueReport = models.get('IssueReport')
        self.Feedback = models.get('Feedback')
        
        self.cache = services.get('cache')
        self.redis_client = services.get('redis_client')
        
        # Service references
        self.TMDBService = services.get('TMDBService')
        self.JikanService = services.get('JikanService')
    
    def get_overview(self):
        """Get admin dashboard overview"""
        try:
            overview_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'general_stats': self._get_general_stats(),
                'recent_activity': self._get_recent_activity(),
                'quick_actions': self._get_quick_actions(),
                'alerts': self._get_alerts()
            }
            
            # Add support stats if available
            if self.SupportTicket:
                overview_data['support_overview'] = self._get_support_overview()
            
            return overview_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard overview: {e}")
            return {'error': 'Failed to load dashboard overview'}
    
    def _get_general_stats(self):
        """Get general system statistics"""
        try:
            total_users = self.User.query.count()
            total_content = self.Content.query.count()
            total_interactions = self.UserInteraction.query.count()
            
            # Active users in last 7 days
            week_ago = datetime.utcnow() - timedelta(days=7)
            active_users = self.User.query.filter(
                self.User.last_active >= week_ago
            ).count()
            
            # New users in last 30 days
            month_ago = datetime.utcnow() - timedelta(days=30)
            new_users = self.User.query.filter(
                self.User.created_at >= month_ago
            ).count()
            
            # Popular content types
            content_types = self.db.session.query(
                self.Content.content_type,
                func.count(self.Content.id).label('count')
            ).group_by(self.Content.content_type).all()
            
            return {
                'total_users': total_users,
                'total_content': total_content,
                'total_interactions': total_interactions,
                'active_users_week': active_users,
                'new_users_month': new_users,
                'content_distribution': [
                    {'type': ct.content_type, 'count': ct.count}
                    for ct in content_types
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting general stats: {e}")
            return {}
    
    def _get_recent_activity(self):
        """Get recent system activity"""
        try:
            # Recent content added
            recent_content = self.Content.query.order_by(
                self.Content.created_at.desc()
            ).limit(5).all()
            
            # Recent user registrations
            recent_users = self.User.query.order_by(
                self.User.created_at.desc()
            ).limit(5).all()
            
            # Recent admin recommendations
            recent_recommendations = self.AdminRecommendation.query.filter_by(
                is_active=True
            ).order_by(self.AdminRecommendation.created_at.desc()).limit(5).all()
            
            return {
                'recent_content': [
                    {
                        'id': content.id,
                        'title': content.title,
                        'type': content.content_type,
                        'rating': content.rating,
                        'created_at': content.created_at.isoformat()
                    }
                    for content in recent_content
                ],
                'recent_users': [
                    {
                        'id': user.id,
                        'username': user.username,
                        'email': user.email,
                        'created_at': user.created_at.isoformat()
                    }
                    for user in recent_users
                ],
                'recent_recommendations': [
                    {
                        'id': rec.id,
                        'content_title': self.Content.query.get(rec.content_id).title if self.Content.query.get(rec.content_id) else 'Unknown',
                        'type': rec.recommendation_type,
                        'admin_name': self.User.query.get(rec.admin_id).username if self.User.query.get(rec.admin_id) else 'Unknown',
                        'created_at': rec.created_at.isoformat()
                    }
                    for rec in recent_recommendations
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting recent activity: {e}")
            return {}
    
    def _get_quick_actions(self):
        """Get quick actions for admin"""
        return [
            {
                'title': 'Add Content',
                'description': 'Search and add new movies, shows, or anime',
                'url': '/admin/content/search',
                'icon': '🎬'
            },
            {
                'title': 'Create Recommendation',
                'description': 'Share your picks with users',
                'url': '/admin/recommendations/create',
                'icon': '⭐'
            },
            {
                'title': 'View Analytics',
                'description': 'Check detailed platform analytics',
                'url': '/admin/analytics',
                'icon': '📊'
            },
            {
                'title': 'Support Center',
                'description': 'Manage tickets and user feedback',
                'url': '/admin/support',
                'icon': '🎧'
            }
        ]
    
    def _get_alerts(self):
        """Get system alerts and notifications with improved error handling"""
        alerts = []
        
        try:
            # Check for urgent tickets if support system is available
            if self.SupportTicket:
                try:
                    urgent_tickets = self.SupportTicket.query.filter(
                        and_(
                            self.SupportTicket.priority == 'urgent',
                            self.SupportTicket.status.in_(['open', 'in_progress'])
                        )
                    ).count()
                    
                    if urgent_tickets > 0:
                        alerts.append({
                            'type': 'urgent',
                            'title': f'{urgent_tickets} Urgent Tickets',
                            'message': f'{urgent_tickets} urgent support tickets need attention',
                            'action_url': '/admin/support/tickets?priority=urgent'
                        })
                    
                except Exception as e:
                    logger.error(f"Error checking urgent tickets: {e}")
                    try:
                        self.db.session.rollback()
                    except Exception:
                        pass
                
                try:
                    # Check for SLA breaches - separate transaction
                    sla_breached = self.SupportTicket.query.filter(
                        and_(
                            self.SupportTicket.sla_breached == True,
                            self.SupportTicket.status.in_(['open', 'in_progress'])
                        )
                    ).count()
                    
                    if sla_breached > 0:
                        alerts.append({
                            'type': 'warning',
                            'title': f'{sla_breached} SLA Breaches',
                            'message': f'{sla_breached} tickets have exceeded their SLA deadline',
                            'action_url': '/admin/support/tickets?sla_breached=true'
                        })
                        
                except Exception as e:
                    logger.error(f"Error checking SLA breaches: {e}")
                    try:
                        self.db.session.rollback()
                    except Exception:
                        pass
            
            # Check for unread feedback
            if self.Feedback:
                try:
                    unread_feedback = self.Feedback.query.filter_by(is_read=False).count()
                    if unread_feedback > 5:
                        alerts.append({
                            'type': 'info',
                            'title': f'{unread_feedback} Unread Feedback',
                            'message': f'{unread_feedback} feedback messages are waiting for review',
                            'action_url': '/admin/support/feedback?unread_only=true'
                        })
                except Exception as e:
                    logger.error(f"Error checking feedback: {e}")
                    try:
                        self.db.session.rollback()
                    except Exception:
                        pass
            
            # Check system health
            try:
                if not self._check_external_apis():
                    alerts.append({
                        'type': 'error',
                        'title': 'External API Issues',
                        'message': 'Some external APIs are not responding properly',
                        'action_url': '/admin/system-health'
                    })
            except Exception as e:
                logger.error(f"Error checking external APIs: {e}")
        
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            alerts.append({
                'type': 'error',
                'title': 'Alert System Error',
                'message': 'Unable to check system alerts',
                'action_url': '/admin/system-health'
            })
        
        return alerts
    
    def _get_support_overview(self):
        """Get support system overview"""
        try:
            if not self.SupportTicket:
                return {'status': 'not_available'}
            
            today = datetime.utcnow().date()
            
            total_tickets = self.SupportTicket.query.count()
            today_tickets = self.SupportTicket.query.filter(
                func.date(self.SupportTicket.created_at) == today
            ).count()
            today_resolved = self.SupportTicket.query.filter(
                func.date(self.SupportTicket.resolved_at) == today
            ).count()
            
            # FIXED: Use string values instead of enum objects
            open_tickets = self.SupportTicket.query.filter(
                self.SupportTicket.status.in_(['open', 'in_progress', 'waiting_for_user'])  # String values
            ).count()
            
            urgent_tickets = self.SupportTicket.query.filter(
                and_(
                    self.SupportTicket.priority == 'urgent',  # String value
                    self.SupportTicket.status.in_(['open', 'in_progress'])  # String values
                )
            ).count()
            
            # Calculate average response time
            avg_response_time = self.db.session.query(
                func.avg(
                    func.extract('epoch', self.SupportTicket.first_response_at - self.SupportTicket.created_at) / 3600
                )
            ).filter(self.SupportTicket.first_response_at.isnot(None)).scalar() or 0
            
            return {
                'total_tickets': total_tickets,
                'open_tickets': open_tickets,
                'urgent_tickets': urgent_tickets,
                'today_created': today_tickets,
                'today_resolved': today_resolved,
                'avg_response_time_hours': round(avg_response_time, 2),
                'resolution_rate': round((today_resolved / max(today_tickets, 1)) * 100, 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting support overview: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_analytics(self):
        """Get detailed analytics data"""
        try:
            return {
                'user_analytics': self._get_user_analytics(),
                'content_analytics': self._get_content_analytics(),
                'interaction_analytics': self._get_interaction_analytics(),
                'performance_metrics': self._get_performance_metrics(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {'error': 'Failed to load analytics'}
    
    def _get_user_analytics(self):
        """Get user-related analytics"""
        try:
            # User growth over time
            user_growth = self.db.session.query(
                func.date(self.User.created_at).label('date'),
                func.count(self.User.id).label('count')
            ).group_by(func.date(self.User.created_at)).order_by('date').limit(30).all()
            
            # Active users trend
            active_users_trend = []
            for i in range(7):
                date = datetime.utcnow().date() - timedelta(days=i)
                count = self.User.query.filter(
                    func.date(self.User.last_active) == date
                ).count()
                active_users_trend.append({
                    'date': date.isoformat(),
                    'count': count
                })
            
            return {
                'user_growth': [
                    {'date': ug.date.isoformat(), 'count': ug.count}
                    for ug in user_growth
                ],
                'active_users_trend': active_users_trend,
                'total_users': self.User.query.count(),
                'active_today': self.User.query.filter(
                    func.date(self.User.last_active) == datetime.utcnow().date()
                ).count()
            }
            
        except Exception as e:
            logger.error(f"Error getting user analytics: {e}")
            return {}
    
    def _get_content_analytics(self):
        """Get content-related analytics"""
        try:
            # Popular content
            popular_content = self.db.session.query(
                self.Content.id, 
                self.Content.title, 
                func.count(self.UserInteraction.id).label('interaction_count')
            ).join(self.UserInteraction).group_by(
                self.Content.id, self.Content.title
            ).order_by(desc('interaction_count')).limit(10).all()
            
            # Genre popularity
            all_interactions = self.UserInteraction.query.join(self.Content).all()
            genre_counts = defaultdict(int)
            for interaction in all_interactions:
                content = self.Content.query.get(interaction.content_id)
                if content and content.genres:
                    try:
                        genres = json.loads(content.genres)
                        for genre in genres:
                            genre_counts[genre] += 1
                    except:
                        pass
            
            popular_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Content type distribution
            content_types = self.db.session.query(
                self.Content.content_type,
                func.count(self.Content.id).label('count')
            ).group_by(self.Content.content_type).all()
            
            return {
                'popular_content': [
                    {'title': pc.title, 'interactions': pc.interaction_count}
                    for pc in popular_content
                ],
                'popular_genres': [
                    {'genre': genre, 'count': count}
                    for genre, count in popular_genres
                ],
                'content_distribution': [
                    {'type': ct.content_type, 'count': ct.count}
                    for ct in content_types
                ],
                'total_content': self.Content.query.count()
            }
            
        except Exception as e:
            logger.error(f"Error getting content analytics: {e}")
            return {}
    
    def _get_interaction_analytics(self):
        """Get user interaction analytics"""
        try:
            # Interaction types breakdown
            interaction_types = self.db.session.query(
                self.UserInteraction.interaction_type,
                func.count(self.UserInteraction.id).label('count')
            ).group_by(self.UserInteraction.interaction_type).all()
            
            # Daily interactions trend
            interaction_trend = []
            for i in range(7):
                date = datetime.utcnow().date() - timedelta(days=i)
                count = self.UserInteraction.query.filter(
                    func.date(self.UserInteraction.timestamp) == date
                ).count()
                interaction_trend.append({
                    'date': date.isoformat(),
                    'count': count
                })
            
            return {
                'interaction_types': [
                    {'type': it.interaction_type, 'count': it.count}
                    for it in interaction_types
                ],
                'daily_trend': interaction_trend,
                'total_interactions': self.UserInteraction.query.count()
            }
            
        except Exception as e:
            logger.error(f"Error getting interaction analytics: {e}")
            return {}
    
    def _get_performance_metrics(self):
        """Get system performance metrics"""
        try:
            # Cache statistics
            cache_stats = self.get_cache_stats()
            
            # Database statistics
            db_stats = {
                'total_tables': len(self.db.metadata.tables),
                'total_records': (
                    self.User.query.count() +
                    self.Content.query.count() +
                    self.UserInteraction.query.count()
                )
            }
            
            if self.SupportTicket:
                db_stats['total_records'] += self.SupportTicket.query.count()
            
            return {
                'cache_stats': cache_stats,
                'database_stats': db_stats,
                'external_apis': self._get_api_status()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def _check_external_apis(self):
        """Check if external APIs are working"""
        try:
            # This is a simple check - in production you might want more sophisticated testing
            if self.TMDBService and self.JikanService:
                return True
            return False
        except:
            return False
    
    def _get_api_status(self):
        """Get external API status"""
        return {
            'tmdb': 'configured' if self.TMDBService else 'not_configured',
            'jikan': 'configured' if self.JikanService else 'not_configured'
        }
    
    def get_cache_stats(self):
        """Get cache statistics"""
        try:
            cache_info = {
                'type': self.app.config.get('CACHE_TYPE', 'unknown'),
                'default_timeout': self.app.config.get('CACHE_DEFAULT_TIMEOUT', 0),
            }
            
            if self.app.config.get('CACHE_TYPE') == 'redis':
                try:
                    import redis
                    REDIS_URL = self.app.config.get('CACHE_REDIS_URL')
                    if REDIS_URL:
                        r = redis.from_url(REDIS_URL)
                        redis_info = r.info()
                        cache_info['redis'] = {
                            'used_memory': redis_info.get('used_memory_human', 'N/A'),
                            'connected_clients': redis_info.get('connected_clients', 0),
                            'total_commands_processed': redis_info.get('total_commands_processed', 0),
                            'uptime_in_seconds': redis_info.get('uptime_in_seconds', 0)
                        }
                except:
                    cache_info['redis'] = {'status': 'Unable to connect'}
            
            return cache_info
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}
    
    def get_system_health(self):
        """Get comprehensive system health status"""
        try:
            health_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'healthy',
                'components': {},
                'configuration': {
                    'telegram_bot': 'configured' if os.environ.get('TELEGRAM_BOT_TOKEN') else 'not_configured',
                    'telegram_channel': 'configured' if os.environ.get('TELEGRAM_CHANNEL_ID') else 'not_configured',
                    'telegram_admin_chat': 'configured' if os.environ.get('TELEGRAM_ADMIN_CHAT_ID') else 'not_configured',
                    'redis': 'configured' if os.environ.get('REDIS_URL') else 'not_configured'
                }
            }
            
            # Database health
            try:
                health_data['components']['database'] = {
                    'status': 'healthy',
                    'total_users': self.User.query.count(),
                    'total_content': self.Content.query.count(),
                    'total_interactions': self.UserInteraction.query.count()
                }
            except Exception as e:
                health_data['components']['database'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_data['status'] = 'degraded'
            
            # Cache health
            try:
                if self.cache:
                    self.cache.set('health_check', 'ok', timeout=10)
                    if self.cache.get('health_check') == 'ok':
                        health_data['components']['cache'] = {'status': 'healthy'}
                    else:
                        health_data['components']['cache'] = {'status': 'degraded'}
                        health_data['status'] = 'degraded'
                else:
                    health_data['components']['cache'] = {'status': 'not_configured'}
            except Exception as e:
                health_data['components']['cache'] = {'status': 'unhealthy', 'error': str(e)}
                health_data['status'] = 'degraded'
            
            # External APIs
            health_data['components']['external_apis'] = self._get_api_status()
            
            # Support system health
            if self.SupportTicket:
                try:
                    health_data['components']['support_system'] = {
                        'status': 'healthy',
                        'total_tickets': self.SupportTicket.query.count(),
                        'open_tickets': self.SupportTicket.query.filter(
                            self.SupportTicket.status.in_(['open', 'in_progress'])
                        ).count(),
                        'total_feedback': self.Feedback.query.count() if self.Feedback else 0
                    }
                except Exception as e:
                    health_data['components']['support_system'] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    health_data['status'] = 'degraded'
            else:
                health_data['components']['support_system'] = {
                    'status': 'not_available'
                }
            
            return health_data
            
        except Exception as e:
            logger.error(f"System health check error: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    # Support-specific dashboard methods
    def get_support_dashboard(self):
        """Get comprehensive support dashboard data"""
        try:
            if not self.SupportTicket:
                return {'error': 'Support system not available'}
            
            return {
                'ticket_stats': self._get_ticket_stats(),
                'metrics': self._get_support_metrics(),
                'category_breakdown': self._get_category_breakdown(),
                'priority_breakdown': self._get_priority_breakdown(),
                'recent_tickets': self._get_recent_tickets(),
                'feedback_stats': self._get_feedback_stats(),
                'recent_feedback': self._get_recent_feedback(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Support dashboard error: {e}")
            return {'error': 'Failed to load support dashboard'}
    
    def _get_ticket_stats(self):
        """Get ticket statistics"""
        try:
            today = datetime.utcnow().date()
            
            total_tickets = self.SupportTicket.query.count()
            today_tickets = self.SupportTicket.query.filter(
                func.date(self.SupportTicket.created_at) == today
            ).count()
            today_resolved = self.SupportTicket.query.filter(
                func.date(self.SupportTicket.resolved_at) == today
            ).count()
            
            # FIXED: Use string values instead of enum objects
            open_tickets = self.SupportTicket.query.filter(
                self.SupportTicket.status.in_(['open', 'in_progress', 'waiting_for_user'])  # String values
            ).count()
            
            urgent_tickets = self.SupportTicket.query.filter(
                and_(
                    self.SupportTicket.priority == 'urgent',  # String value
                    self.SupportTicket.status.in_(['open', 'in_progress'])  # String values
                )
            ).count()
            
            sla_breached = self.SupportTicket.query.filter(
                and_(
                    self.SupportTicket.sla_breached == True,
                    self.SupportTicket.status.in_(['open', 'in_progress'])  # String values
                )
            ).count()
            
            return {
                'total': total_tickets,
                'open': open_tickets,
                'urgent': urgent_tickets,
                'sla_breached': sla_breached,
                'today_created': today_tickets,
                'today_resolved': today_resolved
            }
            
        except Exception as e:
            logger.error(f"Error getting ticket stats: {e}")
            return {}
    
    def _get_support_metrics(self):
        """Get support performance metrics"""
        try:
            # Calculate average response time
            avg_response_time = self.db.session.query(
                func.avg(
                    func.extract('epoch', self.SupportTicket.first_response_at - self.SupportTicket.created_at) / 3600
                )
            ).filter(self.SupportTicket.first_response_at.isnot(None)).scalar() or 0
            
            today_tickets = self.SupportTicket.query.filter(
                func.date(self.SupportTicket.created_at) == datetime.utcnow().date()
            ).count()
            
            today_resolved = self.SupportTicket.query.filter(
                func.date(self.SupportTicket.resolved_at) == datetime.utcnow().date()
            ).count()
            
            return {
                'avg_response_time_hours': round(avg_response_time, 2),
                'resolution_rate': round((today_resolved / max(today_tickets, 1)) * 100, 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting support metrics: {e}")
            return {}
    
    def _get_category_breakdown(self):
        """Get ticket category breakdown"""
        try:
            if not self.SupportCategory:
                return []
            
            category_stats = self.db.session.query(
                self.SupportCategory.name,
                func.count(self.SupportTicket.id).label('count')
            ).join(self.SupportTicket).group_by(self.SupportCategory.name).all()
            
            return [
                {'category': stat.name, 'count': stat.count} 
                for stat in category_stats
            ]
            
        except Exception as e:
            logger.error(f"Error getting category breakdown: {e}")
            return []
    
    def _get_priority_breakdown(self):
        """Get ticket priority breakdown"""
        try:
            # FIXED: Use string values instead of enum objects
            priority_stats = self.db.session.query(
                self.SupportTicket.priority,
                func.count(self.SupportTicket.id).label('count')
            ).filter(
                self.SupportTicket.status.in_(['open', 'in_progress'])  # String values
            ).group_by(self.SupportTicket.priority).all()
            
            return [
                {'priority': stat.priority, 'count': stat.count}  # Already a string
                for stat in priority_stats
            ]
            
        except Exception as e:
            logger.error(f"Error getting priority breakdown: {e}")
            return []
    
    def _get_recent_tickets(self):
        """Get recent tickets for dashboard"""
        try:
            recent_tickets = self.SupportTicket.query.order_by(
                self.SupportTicket.created_at.desc()
            ).limit(10).all()
            
            tickets_data = []
            for ticket in recent_tickets:
                category = None
                if self.SupportCategory and ticket.category_id:
                    category = self.SupportCategory.query.get(ticket.category_id)
                
                tickets_data.append({
                    'id': ticket.id,
                    'ticket_number': ticket.ticket_number,
                    'subject': ticket.subject,
                    'user_name': ticket.user_name,
                    'priority': ticket.priority,  # Already a string
                    'status': ticket.status,  # Already a string
                    'category': category.name if category else 'Unknown',
                    'created_at': ticket.created_at.isoformat(),
                    'is_sla_breached': ticket.sla_breached
                })
            
            return tickets_data
            
        except Exception as e:
            logger.error(f"Error getting recent tickets: {e}")
            return []
    
    def _get_feedback_stats(self):
        """Get feedback statistics"""
        try:
            if not self.Feedback:
                return {'total': 0, 'unread': 0}
            
            total_feedback = self.Feedback.query.count()
            unread_feedback = self.Feedback.query.filter_by(is_read=False).count()
            
            return {
                'total': total_feedback,
                'unread': unread_feedback
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {'total': 0, 'unread': 0}
    
    def _get_recent_feedback(self):
        """Get recent feedback for dashboard"""
        try:
            if not self.Feedback:
                return []
            
            recent_feedback = self.Feedback.query.order_by(
                self.Feedback.created_at.desc()
            ).limit(5).all()
            
            feedback_data = []
            for feedback in recent_feedback:
                feedback_data.append({
                    'id': feedback.id,
                    'subject': feedback.subject,
                    'user_name': feedback.user_name,
                    'feedback_type': feedback.feedback_type if isinstance(feedback.feedback_type, str) else 'general',  # Handle as string
                    'rating': feedback.rating,
                    'is_read': feedback.is_read,
                    'created_at': feedback.created_at.isoformat()
                })
            
            return feedback_data
            
        except Exception as e:
            logger.error(f"Error getting recent feedback: {e}")
            return []
    
    # Additional dashboard methods for other admin functions
    def get_support_tickets(self, page, per_page, status, priority, category_id, search):
        """Get filtered support tickets"""
        try:
            if not self.SupportTicket:
                return {'error': 'Support system not available'}
            
            # Implementation would go here - similar to the original function
            # but moved to dashboard service for better organization
            
            return {'tickets': [], 'pagination': {}}
            
        except Exception as e:
            logger.error(f"Error getting support tickets: {e}")
            return {'error': 'Failed to get support tickets'}
    
    def get_feedback_list(self, page, per_page, feedback_type, is_read, search):
        """Get filtered feedback list"""
        try:
            if not self.Feedback:
                return {'error': 'Feedback system not available'}
            
            # Implementation would go here
            return {'feedback': [], 'pagination': {}}
            
        except Exception as e:
            logger.error(f"Error getting feedback list: {e}")
            return {'error': 'Failed to get feedback list'}
    
    def get_users_management(self, page, per_page, search):
        """Get users for management interface"""
        try:
            # Implementation would go here
            return {'users': [], 'pagination': {}}
            
        except Exception as e:
            logger.error(f"Error getting users management: {e}")
            return {'error': 'Failed to get users'}
    
    def get_content_management(self, page, per_page, content_type, search):
        """Get content for management interface"""
        try:
            # Implementation would go here
            return {'content': [], 'pagination': {}}
            
        except Exception as e:
            logger.error(f"Error getting content management: {e}")
            return {'error': 'Failed to get content'}

def init_dashboard_service(app, db, models, services):
    """Initialize dashboard service"""
    return AdminDashboard(app, db, models, services)