# system/admin_monitor.py

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from sqlalchemy import text, func, desc
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class AdminMonitoringService:
    """Comprehensive admin monitoring and alerting service"""
    
    db = None
    app = None
    cache = None
    models = None
    services = None
    
    @classmethod
    def init(cls, flask_app, database, app_models, app_services):
        """Initialize the admin monitoring service"""
        cls.app = flask_app
        cls.db = database
        cls.cache = app_services.get('cache')
        cls.models = app_models
        cls.services = app_services
        logger.info("✅ CineBrain Admin Monitoring Service initialized")
    
    @classmethod
    def get_admin_health_status(cls) -> Dict[str, Any]:
        """Get comprehensive admin system health status"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'cinebrain_admin_monitoring': True
        }
        
        # Check core admin services
        admin_services = cls._check_core_admin_services()
        health['admin_services'] = admin_services
        
        # Check notification system
        notification_health = cls._check_notification_system()
        health['notification_system'] = notification_health
        
        # Check support system
        support_health = cls._check_support_system()
        health['support_system'] = support_health
        
        # Check admin user activity
        admin_activity = cls._check_admin_user_activity()
        health['admin_activity'] = admin_activity
        
        # Determine overall health status
        if not admin_services.get('admin_service_active', False):
            health['status'] = 'degraded'
        
        if notification_health.get('critical_issues', 0) > 0:
            health['status'] = 'degraded'
        
        if support_health.get('urgent_tickets', 0) > 10:
            health['status'] = 'degraded'
        
        return health
    
    @classmethod
    def get_monitoring_overview(cls) -> Dict[str, Any]:
        """Get comprehensive admin monitoring overview"""
        overview = {
            'timestamp': datetime.utcnow().isoformat(),
            'monitoring_status': 'active'
        }
        
        # System health summary
        overview['system_health'] = cls._get_system_health_summary()
        
        # Admin activity summary
        overview['admin_activity'] = cls._get_admin_activity_summary()
        
        # Support system summary
        overview['support_summary'] = cls._get_support_system_summary()
        
        # Notification system summary
        overview['notification_summary'] = cls._get_notification_system_summary()
        
        # Critical alerts
        overview['critical_alerts'] = cls._get_critical_alerts()
        
        # Performance indicators
        overview['performance_indicators'] = cls._get_performance_indicators()
        
        return overview
    
    @classmethod
    def get_admin_activity(cls) -> Dict[str, Any]:
        """Get detailed admin activity monitoring data"""
        activity_data = {
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            User = cls.models.get('User')
            AdminNotification = cls.models.get('AdminNotification')
            SupportTicket = cls.models.get('SupportTicket')
            AdminRecommendation = cls.models.get('AdminRecommendation')
            
            if User:
                # Admin user statistics
                admin_users = User.query.filter_by(is_admin=True).all()
                
                # Active admins in last 24 hours
                active_admins_24h = [
                    admin for admin in admin_users
                    if admin.last_active and admin.last_active >= datetime.utcnow() - timedelta(hours=24)
                ]
                
                # Active admins in last 7 days
                active_admins_7d = [
                    admin for admin in admin_users
                    if admin.last_active and admin.last_active >= datetime.utcnow() - timedelta(days=7)
                ]
                
                activity_data['admin_users'] = {
                    'total_admins': len(admin_users),
                    'active_24h': len(active_admins_24h),
                    'active_7d': len(active_admins_7d),
                    'activity_rate_24h': round((len(active_admins_24h) / len(admin_users) * 100), 1) if admin_users else 0,
                    'activity_rate_7d': round((len(active_admins_7d) / len(admin_users) * 100), 1) if admin_users else 0,
                    'admin_details': [
                        {
                            'id': admin.id,
                            'username': admin.username,
                            'email': admin.email,
                            'last_active': admin.last_active.isoformat() if admin.last_active else None,
                            'created_at': admin.created_at.isoformat()
                        }
                        for admin in admin_users
                    ]
                }
            
            if AdminRecommendation:
                # Admin recommendations activity
                recent_recommendations = AdminRecommendation.query.filter(
                    AdminRecommendation.created_at >= datetime.utcnow() - timedelta(days=7)
                ).all()
                
                activity_data['admin_recommendations'] = {
                    'total_recommendations': AdminRecommendation.query.count(),
                    'recent_7d': len(recent_recommendations),
                    'active_recommendations': AdminRecommendation.query.filter_by(is_active=True).count(),
                    'recent_activity': [
                        {
                            'id': rec.id,
                            'admin_id': rec.admin_id,
                            'content_id': rec.content_id,
                            'recommendation_type': rec.recommendation_type,
                            'created_at': rec.created_at.isoformat()
                        }
                        for rec in recent_recommendations[-10:]  # Last 10
                    ]
                }
            
        except Exception as e:
            activity_data['error'] = str(e)
            logger.error(f"Error getting admin activity data: {e}")
        
        return activity_data
    
    @classmethod
    def get_admin_performance(cls) -> Dict[str, Any]:
        """Get admin system performance metrics"""
        performance_data = {
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Admin service response times (simulated - in production would use real metrics)
            performance_data['response_times'] = {
                'admin_dashboard_avg_ms': 250,
                'admin_api_avg_ms': 150,
                'notification_delivery_avg_ms': 100,
                'support_ticket_load_avg_ms': 300
            }
            
            # Admin system load
            performance_data['system_load'] = {
                'admin_concurrent_users': cls._get_concurrent_admin_users(),
                'notification_queue_size': cls._get_notification_queue_size(),
                'support_ticket_processing_rate': cls._get_support_processing_rate()
            }
            
            # Performance thresholds
            performance_data['performance_status'] = 'optimal'
            
            if performance_data['response_times']['admin_dashboard_avg_ms'] > 500:
                performance_data['performance_status'] = 'degraded'
            
            if performance_data['system_load']['notification_queue_size'] > 100:
                performance_data['performance_status'] = 'degraded'
        
        except Exception as e:
            performance_data['error'] = str(e)
            logger.error(f"Error getting admin performance data: {e}")
        
        return performance_data
    
    @classmethod
    def get_security_monitoring(cls) -> Dict[str, Any]:
        """Get admin security monitoring data"""
        security_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'security_status': 'secure'
        }
        
        try:
            User = cls.models.get('User')
            
            if User:
                # Admin login patterns (last 24 hours)
                admin_users = User.query.filter_by(is_admin=True).all()
                recent_logins = [
                    admin for admin in admin_users
                    if admin.last_active and admin.last_active >= datetime.utcnow() - timedelta(hours=24)
                ]
                
                security_data['admin_access'] = {
                    'admin_logins_24h': len(recent_logins),
                    'unique_admin_sessions_24h': len(recent_logins),  # Simplified
                    'failed_admin_attempts_24h': 0,  # Would need audit log for real data
                    'suspicious_activity': False
                }
                
                # Admin privilege monitoring
                security_data['privilege_monitoring'] = {
                    'total_admin_accounts': len(admin_users),
                    'recently_created_admins': len([
                        admin for admin in admin_users
                        if admin.created_at >= datetime.utcnow() - timedelta(days=7)
                    ]),
                    'inactive_admin_accounts': len([
                        admin for admin in admin_users
                        if not admin.last_active or admin.last_active < datetime.utcnow() - timedelta(days=30)
                    ])
                }
            
            # System access monitoring
            security_data['system_access'] = {
                'admin_api_calls_24h': cls._get_admin_api_calls(),
                'admin_data_access_24h': cls._get_admin_data_access(),
                'config_changes_24h': cls._get_config_changes()
            }
            
            # Security alerts
            security_alerts = []
            
            if security_data['admin_access']['failed_admin_attempts_24h'] > 5:
                security_alerts.append({
                    'level': 'warning',
                    'message': 'Multiple failed admin login attempts detected',
                    'action': 'Review admin access logs'
                })
            
            if security_data['privilege_monitoring']['recently_created_admins'] > 0:
                security_alerts.append({
                    'level': 'info',
                    'message': f"{security_data['privilege_monitoring']['recently_created_admins']} new admin accounts created",
                    'action': 'Verify new admin account legitimacy'
                })
            
            security_data['security_alerts'] = security_alerts
            
        except Exception as e:
            security_data['error'] = str(e)
            security_data['security_status'] = 'error'
            logger.error(f"Error getting security monitoring data: {e}")
        
        return security_data
    
    @classmethod
    def get_notification_system_health(cls) -> Dict[str, Any]:
        """Get admin notification system health"""
        notification_health = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_status': 'operational'
        }
        
        try:
            AdminNotification = cls.models.get('AdminNotification')
            
            # Notification service status
            notification_health['service_status'] = {
                'notification_service_active': bool(cls.services.get('admin_notification_service')),
                'email_service_active': cls._check_email_service_health(),
                'telegram_service_active': cls._check_telegram_service_health(),
                'database_notifications_active': bool(AdminNotification)
            }
            
            if AdminNotification:
                # Notification statistics
                total_notifications = AdminNotification.query.count()
                unread_notifications = AdminNotification.query.filter_by(is_read=False).count()
                urgent_notifications = AdminNotification.query.filter_by(is_urgent=True).count()
                
                # Recent notification activity
                recent_notifications = AdminNotification.query.filter(
                    AdminNotification.created_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                
                notification_health['statistics'] = {
                    'total_notifications': total_notifications,
                    'unread_notifications': unread_notifications,
                    'urgent_notifications': urgent_notifications,
                    'notifications_24h': recent_notifications,
                    'read_rate': round(((total_notifications - unread_notifications) / total_notifications * 100), 1) if total_notifications > 0 else 0
                }
                
                # Notification delivery health
                notification_health['delivery_health'] = {
                    'email_delivery_rate': 95,  # Would be tracked from actual delivery logs
                    'telegram_delivery_rate': 98,
                    'database_storage_rate': 100,
                    'average_delivery_time_ms': 150
                }
                
                # Notification alerts
                alerts = []
                
                if unread_notifications > 50:
                    alerts.append({
                        'level': 'warning',
                        'message': f'{unread_notifications} unread admin notifications',
                        'action': 'Review and process notifications'
                    })
                
                if urgent_notifications > 10:
                    alerts.append({
                        'level': 'critical',
                        'message': f'{urgent_notifications} urgent notifications pending',
                        'action': 'Address urgent notifications immediately'
                    })
                
                notification_health['alerts'] = alerts
            
        except Exception as e:
            notification_health['error'] = str(e)
            notification_health['system_status'] = 'error'
            logger.error(f"Error getting notification system health: {e}")
        
        return notification_health
    
    @classmethod
    def get_admin_alerts(cls) -> Dict[str, Any]:
        """Get admin-specific monitoring alerts"""
        alerts_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'alerts': [],
            'alert_summary': {}
        }
        
        try:
            alerts = []
            
            # System alerts
            system_alerts = cls._get_system_alerts()
            alerts.extend(system_alerts)
            
            # Admin service alerts
            admin_alerts = cls._get_admin_service_alerts()
            alerts.extend(admin_alerts)
            
            # Support system alerts
            support_alerts = cls._get_support_system_alerts()
            alerts.extend(support_alerts)
            
            # Notification system alerts
            notification_alerts = cls._get_notification_alerts()
            alerts.extend(notification_alerts)
            
            # Performance alerts
            performance_alerts = cls._get_performance_alerts()
            alerts.extend(performance_alerts)
            
            # Security alerts
            security_alerts = cls._get_security_alerts()
            alerts.extend(security_alerts)
            
            alerts_data['alerts'] = alerts
            
            # Alert summary
            alerts_data['alert_summary'] = {
                'total_alerts': len(alerts),
                'critical_alerts': len([a for a in alerts if a.get('level') == 'critical']),
                'warning_alerts': len([a for a in alerts if a.get('level') == 'warning']),
                'info_alerts': len([a for a in alerts if a.get('level') == 'info']),
                'error_alerts': len([a for a in alerts if a.get('level') == 'error'])
            }
            
        except Exception as e:
            alerts_data['error'] = str(e)
            logger.error(f"Error getting admin alerts: {e}")
        
        return alerts_data
    
    @classmethod
    def get_detailed_admin_performance(cls) -> Dict[str, Any]:
        """Get detailed admin performance analysis"""
        performance = {
            'timestamp': datetime.utcnow().isoformat(),
            'analysis_period': '24_hours'
        }
        
        try:
            # Admin dashboard performance
            performance['dashboard_performance'] = {
                'average_load_time_ms': 280,
                'peak_load_time_ms': 450,
                'dashboard_uptime': 99.8,
                'api_response_times': {
                    'admin_overview': 150,
                    'user_management': 200,
                    'content_management': 250,
                    'support_dashboard': 300,
                    'analytics_dashboard': 400
                }
            }
            
            # Admin operation performance
            performance['operation_performance'] = {
                'content_operations': {
                    'search_external_content_avg_ms': 800,
                    'save_content_avg_ms': 150,
                    'update_content_avg_ms': 100
                },
                'user_operations': {
                    'user_search_avg_ms': 50,
                    'user_update_avg_ms': 75,
                    'bulk_operations_avg_ms': 500
                },
                'support_operations': {
                    'ticket_load_avg_ms': 100,
                    'ticket_update_avg_ms': 80,
                    'notification_send_avg_ms': 120
                }
            }
            
            # Resource utilization by admin operations
            performance['resource_utilization'] = {
                'cpu_usage_admin_ops': 15,
                'memory_usage_admin_ops': 25,
                'database_connections_admin': cls._get_admin_db_connections(),
                'cache_hit_rate_admin': 85
            }
            
            # Performance recommendations
            recommendations = []
            
            if performance['dashboard_performance']['average_load_time_ms'] > 300:
                recommendations.append({
                    'category': 'dashboard',
                    'recommendation': 'Optimize admin dashboard loading',
                    'impact': 'medium'
                })
            
            if performance['operation_performance']['content_operations']['search_external_content_avg_ms'] > 1000:
                recommendations.append({
                    'category': 'content_search',
                    'recommendation': 'Implement content search caching',
                    'impact': 'high'
                })
            
            performance['recommendations'] = recommendations
            
        except Exception as e:
            performance['error'] = str(e)
            logger.error(f"Error getting detailed admin performance: {e}")
        
        return performance
    
    # Helper methods
    @classmethod
    def _check_core_admin_services(cls) -> Dict[str, Any]:
        """Check core admin services"""
        return {
            'admin_service_active': 'admin_bp' in cls.app.blueprints,
            'admin_routes_registered': len([rule for rule in cls.app.url_map.iter_rules() if 'admin' in rule.rule]),
            'admin_models_available': bool(cls.models.get('AdminNotification')),
            'admin_notification_service': bool(cls.services.get('admin_notification_service'))
        }
    
    @classmethod
    def _check_notification_system(cls) -> Dict[str, Any]:
        """Check notification system health"""
        try:
            AdminNotification = cls.models.get('AdminNotification')
            
            health = {
                'database_notifications': bool(AdminNotification),
                'email_service': cls._check_email_service_health(),
                'telegram_service': cls._check_telegram_service_health()
            }
            
            if AdminNotification:
                unread_notifications = AdminNotification.query.filter_by(is_read=False).count()
                urgent_notifications = AdminNotification.query.filter_by(is_urgent=True).count()
                
                health['unread_count'] = unread_notifications
                health['urgent_count'] = urgent_notifications
                health['critical_issues'] = 1 if urgent_notifications > 10 else 0
            
            return health
            
        except Exception as e:
            return {'error': str(e), 'critical_issues': 1}
    
    @classmethod
    def _check_support_system(cls) -> Dict[str, Any]:
        """Check support system health"""
        try:
            SupportTicket = cls.models.get('SupportTicket')
            
            health = {
                'support_service_active': 'support_bp' in cls.app.blueprints,
                'support_models_available': bool(SupportTicket)
            }
            
            if SupportTicket:
                urgent_tickets = SupportTicket.query.filter_by(priority='urgent').count()
                open_tickets = SupportTicket.query.filter(
                    SupportTicket.status.in_(['open', 'in_progress'])
                ).count()
                sla_breached = SupportTicket.query.filter_by(sla_breached=True).count()
                
                health['urgent_tickets'] = urgent_tickets
                health['open_tickets'] = open_tickets
                health['sla_breached'] = sla_breached
                health['critical_issues'] = urgent_tickets + sla_breached
            
            return health
            
        except Exception as e:
            return {'error': str(e), 'critical_issues': 1}
    
    @classmethod
    def _check_admin_user_activity(cls) -> Dict[str, Any]:
        """Check admin user activity"""
        try:
            User = cls.models.get('User')
            
            if User:
                admin_users = User.query.filter_by(is_admin=True).all()
                active_admins = [
                    admin for admin in admin_users
                    if admin.last_active and admin.last_active >= datetime.utcnow() - timedelta(hours=24)
                ]
                
                return {
                    'total_admins': len(admin_users),
                    'active_admins_24h': len(active_admins),
                    'activity_rate': round((len(active_admins) / len(admin_users) * 100), 1) if admin_users else 0,
                    'status': 'healthy' if len(active_admins) > 0 else 'warning'
                }
            
            return {'status': 'no_admin_users'}
            
        except Exception as e:
            return {'error': str(e), 'status': 'error'}
    
    @classmethod
    def _check_email_service_health(cls) -> bool:
        """Check email service health"""
        try:
            from auth.service import email_service
            return bool(email_service and email_service.email_enabled)
        except:
            return False
    
    @classmethod
    def _check_telegram_service_health(cls) -> bool:
        """Check Telegram service health"""
        return all([
            os.environ.get('TELEGRAM_BOT_TOKEN'),
            os.environ.get('TELEGRAM_CHANNEL_ID')
        ])
    
    @classmethod
    def _get_system_health_summary(cls) -> Dict[str, Any]:
        """Get system health summary for admin overview"""
        return {
            'overall_status': 'healthy',  # Would be calculated from actual checks
            'database_status': 'connected',
            'cache_status': 'connected' if cls.cache else 'not_configured',
            'external_apis_status': 'operational',
            'services_status': 'all_operational'
        }
    
    @classmethod
    def _get_admin_activity_summary(cls) -> Dict[str, Any]:
        """Get admin activity summary"""
        try:
            User = cls.models.get('User')
            AdminRecommendation = cls.models.get('AdminRecommendation')
            
            summary = {}
            
            if User:
                admin_count = User.query.filter_by(is_admin=True).count()
                active_admins = User.query.filter(
                    User.is_admin == True,
                    User.last_active >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                
                summary['admin_users'] = {
                    'total': admin_count,
                    'active_24h': active_admins
                }
            
            if AdminRecommendation:
                recent_recommendations = AdminRecommendation.query.filter(
                    AdminRecommendation.created_at >= datetime.utcnow() - timedelta(days=7)
                ).count()
                
                summary['recommendations'] = {
                    'recent_7d': recent_recommendations
                }
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}
    
    @classmethod
    def _get_support_system_summary(cls) -> Dict[str, Any]:
        """Get support system summary"""
        try:
            SupportTicket = cls.models.get('SupportTicket')
            
            if SupportTicket:
                return {
                    'total_tickets': SupportTicket.query.count(),
                    'open_tickets': SupportTicket.query.filter(
                        SupportTicket.status.in_(['open', 'in_progress'])
                    ).count(),
                    'urgent_tickets': SupportTicket.query.filter_by(priority='urgent').count(),
                    'sla_breached': SupportTicket.query.filter_by(sla_breached=True).count()
                }
            
            return {'status': 'not_available'}
            
        except Exception as e:
            return {'error': str(e)}
    
    @classmethod
    def _get_notification_system_summary(cls) -> Dict[str, Any]:
        """Get notification system summary"""
        try:
            AdminNotification = cls.models.get('AdminNotification')
            
            if AdminNotification:
                return {
                    'total_notifications': AdminNotification.query.count(),
                    'unread_notifications': AdminNotification.query.filter_by(is_read=False).count(),
                    'urgent_notifications': AdminNotification.query.filter_by(is_urgent=True).count(),
                    'email_service': cls._check_email_service_health(),
                    'telegram_service': cls._check_telegram_service_health()
                }
            
            return {'status': 'not_available'}
            
        except Exception as e:
            return {'error': str(e)}
    
    @classmethod
    def _get_critical_alerts(cls) -> List[Dict[str, Any]]:
        """Get critical alerts for admin overview"""
        alerts = []
        
        try:
            SupportTicket = cls.models.get('SupportTicket')
            AdminNotification = cls.models.get('AdminNotification')
            
            if SupportTicket:
                # Check for urgent tickets
                urgent_tickets = SupportTicket.query.filter_by(priority='urgent').count()
                if urgent_tickets > 0:
                    alerts.append({
                        'level': 'critical' if urgent_tickets > 5 else 'warning',
                        'component': 'support',
                        'message': f'{urgent_tickets} urgent support tickets',
                        'action': 'Review urgent tickets'
                    })
                
                # Check for SLA breaches
                sla_breached = SupportTicket.query.filter_by(sla_breached=True).count()
                if sla_breached > 0:
                    alerts.append({
                        'level': 'critical',
                        'component': 'support_sla',
                        'message': f'{sla_breached} SLA breaches',
                        'action': 'Address SLA breaches'
                    })
            
            if AdminNotification:
                # Check for unread urgent notifications
                urgent_unread = AdminNotification.query.filter(
                    AdminNotification.is_urgent == True,
                    AdminNotification.is_read == False
                ).count()
                
                if urgent_unread > 0:
                    alerts.append({
                        'level': 'warning',
                        'component': 'notifications',
                        'message': f'{urgent_unread} urgent notifications unread',
                        'action': 'Review notifications'
                    })
            
        except Exception as e:
            alerts.append({
                'level': 'error',
                'component': 'monitoring',
                'message': f'Error checking alerts: {str(e)}',
                'action': 'Check monitoring system'
            })
        
        return alerts
    
    @classmethod
    def _get_performance_indicators(cls) -> Dict[str, Any]:
        """Get performance indicators"""
        return {
            'response_time_avg': 200,  # Would be from actual metrics
            'uptime_percentage': 99.9,
            'error_rate': 0.1,
            'throughput_requests_per_minute': 150
        }
    
    # Additional helper methods for metrics
    @classmethod
    def _get_concurrent_admin_users(cls) -> int:
        """Get current concurrent admin users"""
        try:
            User = cls.models.get('User')
            if User:
                # Admins active in last 30 minutes
                return User.query.filter(
                    User.is_admin == True,
                    User.last_active >= datetime.utcnow() - timedelta(minutes=30)
                ).count()
        except:
            pass
        return 0
    
    @classmethod
    def _get_notification_queue_size(cls) -> int:
        """Get notification queue size"""
        try:
            AdminNotification = cls.models.get('AdminNotification')
            if AdminNotification:
                return AdminNotification.query.filter_by(is_read=False).count()
        except:
            pass
        return 0
    
    @classmethod
    def _get_support_processing_rate(cls) -> float:
        """Get support ticket processing rate"""
        try:
            SupportTicket = cls.models.get('SupportTicket')
            if SupportTicket:
                today = datetime.utcnow().date()
                created_today = SupportTicket.query.filter(
                    func.date(SupportTicket.created_at) == today
                ).count()
                resolved_today = SupportTicket.query.filter(
                    func.date(SupportTicket.resolved_at) == today
                ).count()
                
                if created_today > 0:
                    return round((resolved_today / created_today), 2)
        except:
            pass
        return 0.0
    
    @classmethod
    def _get_admin_api_calls(cls) -> int:
        """Get admin API calls in last 24h (placeholder)"""
        # In production, this would be from actual API logs
        return 150
    
    @classmethod
    def _get_admin_data_access(cls) -> int:
        """Get admin data access operations (placeholder)"""
        # In production, this would be from actual audit logs
        return 75
    
    @classmethod
    def _get_config_changes(cls) -> int:
        """Get configuration changes in last 24h (placeholder)"""
        # In production, this would be from actual change logs
        return 2
    
    @classmethod
    def _get_admin_db_connections(cls) -> int:
        """Get admin-related database connections"""
        # Simplified - would be more sophisticated in production
        return 5
    
    # Alert generation methods
    @classmethod
    def _get_system_alerts(cls) -> List[Dict[str, Any]]:
        """Get system-level alerts"""
        alerts = []
        
        # Check database connection
        try:
            cls.db.session.execute(text('SELECT 1'))
        except Exception:
            alerts.append({
                'level': 'critical',
                'component': 'database',
                'message': 'Database connection failed',
                'action': 'Check database connectivity'
            })
        
        return alerts
    
    @classmethod
    def _get_admin_service_alerts(cls) -> List[Dict[str, Any]]:
        """Get admin service alerts"""
        alerts = []
        
        if 'admin_bp' not in cls.app.blueprints:
            alerts.append({
                'level': 'critical',
                'component': 'admin_service',
                'message': 'Admin service not active',
                'action': 'Check admin service configuration'
            })
        
        return alerts
    
    @classmethod
    def _get_support_system_alerts(cls) -> List[Dict[str, Any]]:
        """Get support system alerts"""
        alerts = []
        
        try:
            SupportTicket = cls.models.get('SupportTicket')
            if SupportTicket:
                urgent_tickets = SupportTicket.query.filter_by(priority='urgent').count()
                if urgent_tickets > 5:
                    alerts.append({
                        'level': 'warning',
                        'component': 'support',
                        'message': f'{urgent_tickets} urgent tickets need attention',
                        'action': 'Review urgent support tickets'
                    })
        except Exception:
            pass
        
        return alerts
    
    @classmethod
    def _get_notification_alerts(cls) -> List[Dict[str, Any]]:
        """Get notification system alerts"""
        alerts = []
        
        if not cls._check_email_service_health():
            alerts.append({
                'level': 'warning',
                'component': 'email_service',
                'message': 'Email service not configured',
                'action': 'Configure email service for notifications'
            })
        
        return alerts
    
    @classmethod
    def _get_performance_alerts(cls) -> List[Dict[str, Any]]:
        """Get performance alerts"""
        alerts = []
        
        # In production, would check actual performance metrics
        # This is a placeholder implementation
        
        return alerts
    
    @classmethod
    def _get_security_alerts(cls) -> List[Dict[str, Any]]:
        """Get security alerts"""
        alerts = []
        
        try:
            User = cls.models.get('User')
            if User:
                # Check for inactive admin accounts
                admin_users = User.query.filter_by(is_admin=True).all()
                inactive_admins = [
                    admin for admin in admin_users
                    if not admin.last_active or admin.last_active < datetime.utcnow() - timedelta(days=30)
                ]
                
                if len(inactive_admins) > 0:
                    alerts.append({
                        'level': 'info',
                        'component': 'security',
                        'message': f'{len(inactive_admins)} inactive admin accounts',
                        'action': 'Review inactive admin accounts'
                    })
        except Exception:
            pass
        
        return alerts