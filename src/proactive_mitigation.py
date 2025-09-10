import logging
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import subprocess
import threading
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MitigationAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    RESTART_SERVICE = "restart_service"
    CLEAR_CACHE = "clear_cache"
    INCREASE_MEMORY = "increase_memory"
    THROTTLE_REQUESTS = "throttle_requests"
    FAILOVER = "failover"
    ALERT_OPERATIONS = "alert_operations"

class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ProactiveMitigationEngine:
    """
    Proactive mitigation engine that automatically responds to detected anomalies
    and system issues to reduce downtime and maintain system performance.
    """
    
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.mitigation_history = []
        self.active_mitigations = {}
        self.cooldown_periods = {}
        
        # Mitigation rules mapping
        self.mitigation_rules = {
            'high_cpu_usage': [MitigationAction.SCALE_UP, MitigationAction.THROTTLE_REQUESTS],
            'high_memory_usage': [MitigationAction.CLEAR_CACHE, MitigationAction.INCREASE_MEMORY, MitigationAction.RESTART_SERVICE],
            'high_response_time': [MitigationAction.SCALE_UP, MitigationAction.CLEAR_CACHE],
            'high_error_rate': [MitigationAction.RESTART_SERVICE, MitigationAction.FAILOVER],
            'disk_bottleneck': [MitigationAction.CLEAR_CACHE, MitigationAction.SCALE_UP],
            'network_congestion': [MitigationAction.THROTTLE_REQUESTS, MitigationAction.SCALE_UP],
            'service_unavailable': [MitigationAction.RESTART_SERVICE, MitigationAction.FAILOVER]
        }
        
        logger.info("ProactiveMitigationEngine initialized")
    
    def load_config(self, config_path):
        """Load configuration for mitigation engine"""
        default_config = {
            'thresholds': {
                'cpu_usage': {'warning': 70, 'critical': 90},
                'memory_usage': {'warning': 70, 'critical': 90},
                'response_time_ms': {'warning': 500, 'critical': 1000},
                'error_rate_percent': {'warning': 2, 'critical': 5},
                'disk_io_mbps': {'warning': 300, 'critical': 500},
                'network_io_mbps': {'warning': 150, 'critical': 200}
            },
            'cooldown_minutes': {
                'scale_up': 10,
                'scale_down': 15,
                'restart_service': 5,
                'clear_cache': 2,
                'increase_memory': 30,
                'throttle_requests': 5,
                'failover': 60
            },
            'auto_mitigation_enabled': True,
            'notification_endpoints': {
                'slack_webhook': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
                'email_api': 'https://api.sendgrid.com/v3/mail/send',
                'pagerduty_api': 'https://events.pagerduty.com/v2/enqueue'
            },
            'aws_config': {
                'region': 'us-west-2',
                'auto_scaling_group': 'monitoring-asg',
                'load_balancer': 'monitoring-lb'
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    default_config.update(custom_config)
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def analyze_anomaly(self, anomaly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze anomaly data and determine appropriate mitigation actions"""
        logger.info(f"Analyzing anomaly: {anomaly_data.get('server_id', 'unknown')}")
        
        analysis = {
            'server_id': anomaly_data.get('server_id'),
            'timestamp': anomaly_data.get('timestamp', datetime.now().isoformat()),
            'anomaly_types': [],
            'severity': SeverityLevel.LOW,
            'recommended_actions': [],
            'urgency_score': 0
        }
        
        # Analyze each metric against thresholds
        metrics = anomaly_data.get('metrics', {})
        thresholds = self.config['thresholds']
        
        for metric, value in metrics.items():
            if metric in thresholds:
                threshold_config = thresholds[metric]
                
                if value >= threshold_config.get('critical', float('inf')):
                    analysis['anomaly_types'].append(f'critical_{metric}')
                    analysis['urgency_score'] += 10
                    if analysis['severity'].value < SeverityLevel.CRITICAL.value:
                        analysis['severity'] = SeverityLevel.CRITICAL
                elif value >= threshold_config.get('warning', float('inf')):
                    analysis['anomaly_types'].append(f'high_{metric}')
                    analysis['urgency_score'] += 5
                    if analysis['severity'].value < SeverityLevel.HIGH.value:
                        analysis['severity'] = SeverityLevel.HIGH
        
        # Determine recommended actions based on anomaly types
        for anomaly_type in analysis['anomaly_types']:
            # Map anomaly types to mitigation rules
            rule_key = anomaly_type.replace('critical_', 'high_').replace('_percent', '_rate')
            if rule_key in self.mitigation_rules:
                for action in self.mitigation_rules[rule_key]:
                    if action not in analysis['recommended_actions']:
                        analysis['recommended_actions'].append(action)
        
        logger.info(f"Analysis complete: {len(analysis['recommended_actions'])} actions recommended")
        return analysis
    
    def check_cooldown(self, action: MitigationAction, server_id: str) -> bool:
        """Check if action is in cooldown period"""
        cooldown_key = f"{action.value}_{server_id}"
        
        if cooldown_key in self.cooldown_periods:
            last_execution = self.cooldown_periods[cooldown_key]
            cooldown_minutes = self.config['cooldown_minutes'].get(action.value, 5)
            
            if datetime.now() - last_execution < timedelta(minutes=cooldown_minutes):
                logger.info(f"Action {action.value} for {server_id} is in cooldown")
                return True
        
        return False
    
    def execute_mitigation_action(self, action: MitigationAction, server_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific mitigation action"""
        logger.info(f"Executing mitigation action: {action.value} for {server_id}")
        
        result = {
            'action': action.value,
            'server_id': server_id,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'message': '',
            'details': {}
        }
        
        try:
            if action == MitigationAction.SCALE_UP:
                result = self._scale_up_infrastructure(server_id, context)
            elif action == MitigationAction.SCALE_DOWN:
                result = self._scale_down_infrastructure(server_id, context)
            elif action == MitigationAction.RESTART_SERVICE:
                result = self._restart_service(server_id, context)
            elif action == MitigationAction.CLEAR_CACHE:
                result = self._clear_cache(server_id, context)
            elif action == MitigationAction.INCREASE_MEMORY:
                result = self._increase_memory(server_id, context)
            elif action == MitigationAction.THROTTLE_REQUESTS:
                result = self._throttle_requests(server_id, context)
            elif action == MitigationAction.FAILOVER:
                result = self._initiate_failover(server_id, context)
            elif action == MitigationAction.ALERT_OPERATIONS:
                result = self._alert_operations_team(server_id, context)
            
            if result['success']:
                # Update cooldown period
                cooldown_key = f"{action.value}_{server_id}"
                self.cooldown_periods[cooldown_key] = datetime.now()
                
                # Record in history
                self.mitigation_history.append(result)
                
                logger.info(f"Mitigation action {action.value} completed successfully")
            else:
                logger.error(f"Mitigation action {action.value} failed: {result['message']}")
        
        except Exception as e:
            result['success'] = False
            result['message'] = f"Exception during mitigation: {str(e)}"
            logger.error(f"Exception executing {action.value}: {str(e)}")
        
        return result
    
    def _scale_up_infrastructure(self, server_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Scale up infrastructure (simulated)"""
        # In a real implementation, this would interact with AWS Auto Scaling, Kubernetes, etc.
        logger.info(f"Scaling up infrastructure for {server_id}")
        
        # Simulate AWS Auto Scaling Group scaling
        try:
            # Example AWS CLI command (would be replaced with boto3 in production)
            # subprocess.run(['aws', 'autoscaling', 'set-desired-capacity', 
            #                '--auto-scaling-group-name', self.config['aws_config']['auto_scaling_group'],
            #                '--desired-capacity', '3'], check=True)
            
            # Simulated success
            time.sleep(2)  # Simulate processing time
            
            return {
                'action': 'scale_up',
                'server_id': server_id,
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'message': 'Infrastructure scaled up successfully',
                'details': {
                    'previous_capacity': 2,
                    'new_capacity': 3,
                    'estimated_completion': '5 minutes'
                }
            }
        except Exception as e:
            return {
                'action': 'scale_up',
                'server_id': server_id,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'message': f'Failed to scale up: {str(e)}',
                'details': {}
            }
    
    def _scale_down_infrastructure(self, server_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Scale down infrastructure (simulated)"""
        logger.info(f"Scaling down infrastructure for {server_id}")
        
        time.sleep(1)
        return {
            'action': 'scale_down',
            'server_id': server_id,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'message': 'Infrastructure scaled down successfully',
            'details': {'previous_capacity': 3, 'new_capacity': 2}
        }
    
    def _restart_service(self, server_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Restart service (simulated)"""
        logger.info(f"Restarting service on {server_id}")
        
        # In production, this would use SSH, Ansible, or container orchestration
        time.sleep(3)
        return {
            'action': 'restart_service',
            'server_id': server_id,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'message': 'Service restarted successfully',
            'details': {'service_name': 'monitoring-app', 'restart_time': '30 seconds'}
        }
    
    def _clear_cache(self, server_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Clear cache (simulated)"""
        logger.info(f"Clearing cache on {server_id}")
        
        time.sleep(1)
        return {
            'action': 'clear_cache',
            'server_id': server_id,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'message': 'Cache cleared successfully',
            'details': {'cache_type': 'redis', 'cleared_size': '2.5 GB'}
        }
    
    def _increase_memory(self, server_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Increase memory allocation (simulated)"""
        logger.info(f"Increasing memory allocation for {server_id}")
        
        time.sleep(2)
        return {
            'action': 'increase_memory',
            'server_id': server_id,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'message': 'Memory allocation increased successfully',
            'details': {'previous_memory': '8 GB', 'new_memory': '12 GB'}
        }
    
    def _throttle_requests(self, server_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Throttle incoming requests (simulated)"""
        logger.info(f"Throttling requests for {server_id}")
        
        time.sleep(1)
        return {
            'action': 'throttle_requests',
            'server_id': server_id,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'message': 'Request throttling enabled successfully',
            'details': {'throttle_rate': '100 req/min', 'duration': '10 minutes'}
        }
    
    def _initiate_failover(self, server_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Initiate failover to backup systems (simulated)"""
        logger.info(f"Initiating failover for {server_id}")
        
        time.sleep(5)
        return {
            'action': 'failover',
            'server_id': server_id,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'message': 'Failover completed successfully',
            'details': {'backup_server': 'server_backup_1', 'failover_time': '45 seconds'}
        }
    
    def _alert_operations_team(self, server_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Alert operations team (simulated)"""
        logger.info(f"Alerting operations team about {server_id}")
        
        # In production, this would send actual notifications
        alert_message = {
            'server_id': server_id,
            'severity': context.get('severity', 'unknown'),
            'anomaly_types': context.get('anomaly_types', []),
            'timestamp': datetime.now().isoformat(),
            'message': f"Critical anomaly detected on {server_id}. Immediate attention required."
        }
        
        # Simulate sending notifications
        logger.info(f"Sending alert: {json.dumps(alert_message, indent=2)}")
        
        return {
            'action': 'alert_operations',
            'server_id': server_id,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'message': 'Operations team alerted successfully',
            'details': {'channels': ['slack', 'email', 'pagerduty']}
        }
    
    def process_anomaly(self, anomaly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to process an anomaly and execute mitigations"""
        logger.info(f"Processing anomaly for server: {anomaly_data.get('server_id', 'unknown')}")
        
        # Analyze the anomaly
        analysis = self.analyze_anomaly(anomaly_data)
        
        # Check if auto-mitigation is enabled
        if not self.config.get('auto_mitigation_enabled', True):
            logger.info("Auto-mitigation is disabled. Only alerting operations team.")
            analysis['recommended_actions'] = [MitigationAction.ALERT_OPERATIONS]
        
        # Execute mitigation actions
        mitigation_results = []
        
        for action in analysis['recommended_actions']:
            # Check cooldown period
            if self.check_cooldown(action, analysis['server_id']):
                continue
            
            # Execute the action
            result = self.execute_mitigation_action(action, analysis['server_id'], analysis)
            mitigation_results.append(result)
            
            # For critical issues, execute actions sequentially and check if issue is resolved
            if analysis['severity'] == SeverityLevel.CRITICAL:
                time.sleep(2)  # Wait between actions for critical issues
        
        # Compile final result
        final_result = {
            'anomaly_analysis': analysis,
            'mitigation_results': mitigation_results,
            'total_actions_executed': len(mitigation_results),
            'processing_timestamp': datetime.now().isoformat(),
            'estimated_downtime_reduction': self._calculate_downtime_reduction(mitigation_results)
        }
        
        logger.info(f"Anomaly processing complete. {len(mitigation_results)} actions executed.")
        return final_result
    
    def _calculate_downtime_reduction(self, mitigation_results: List[Dict[str, Any]]) -> str:
        """Calculate estimated downtime reduction based on executed actions"""
        # Simplified calculation - in production, this would be based on historical data
        successful_actions = [r for r in mitigation_results if r['success']]
        
        if not successful_actions:
            return "0 minutes"
        
        # Estimate based on action types
        reduction_minutes = 0
        for result in successful_actions:
            action = result['action']
            if action in ['scale_up', 'failover']:
                reduction_minutes += 15
            elif action in ['restart_service', 'clear_cache']:
                reduction_minutes += 10
            elif action in ['throttle_requests', 'increase_memory']:
                reduction_minutes += 5
        
        return f"{min(reduction_minutes, 35)} minutes"  # Cap at 35 minutes (from 10 to 6.5 hours)
    
    def get_mitigation_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get mitigation history for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_history = [
            record for record in self.mitigation_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_time
        ]
        
        return recent_history
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get a summary of system health and mitigation effectiveness"""
        recent_history = self.get_mitigation_history(24)
        
        summary = {
            'total_mitigations_24h': len(recent_history),
            'successful_mitigations': len([r for r in recent_history if r['success']]),
            'most_common_actions': {},
            'estimated_total_downtime_prevented': "0 minutes",
            'system_stability_score': 0.0
        }
        
        if recent_history:
            # Calculate most common actions
            action_counts = {}
            for record in recent_history:
                action = record['action']
                action_counts[action] = action_counts.get(action, 0) + 1
            
            summary['most_common_actions'] = dict(sorted(action_counts.items(), key=lambda x: x[1], reverse=True))
            
            # Calculate system stability score (0-100)
            success_rate = summary['successful_mitigations'] / summary['total_mitigations_24h']
            summary['system_stability_score'] = round(success_rate * 100, 1)
            
            # Estimate total downtime prevented
            total_prevention = summary['successful_mitigations'] * 15  # Average 15 minutes per successful mitigation
            summary['estimated_total_downtime_prevented'] = f"{total_prevention} minutes"
        
        return summary

def main():
    """Main function for testing proactive mitigation"""
    mitigation_engine = ProactiveMitigationEngine()
    
    # Simulate an anomaly
    sample_anomaly = {
        'server_id': 'server_1',
        'timestamp': datetime.now().isoformat(),
        'anomaly_severity': 'high',
        'metrics': {
            'cpu_usage': 95.0,
            'memory_usage': 88.0,
            'response_time_ms': 1200.0,
            'error_rate_percent': 6.5
        }
    }
    
    print("=== Proactive Mitigation Engine Test ===")
    print(f"Processing sample anomaly: {sample_anomaly['server_id']}")
    
    # Process the anomaly
    result = mitigation_engine.process_anomaly(sample_anomaly)
    
    print("\n=== Processing Results ===")
    print(f"Anomaly severity: {result['anomaly_analysis']['severity'].value}")
    print(f"Actions executed: {result['total_actions_executed']}")
    print(f"Estimated downtime reduction: {result['estimated_downtime_reduction']}")
    
    print("\n=== Mitigation Actions ===")
    for i, action_result in enumerate(result['mitigation_results'], 1):
        status = "✓" if action_result['success'] else "✗"
        print(f"{i}. {status} {action_result['action']}: {action_result['message']}")
    
    # Get system health summary
    health_summary = mitigation_engine.get_system_health_summary()
    print("\n=== System Health Summary ===")
    print(f"Total mitigations (24h): {health_summary['total_mitigations_24h']}")
    print(f"Success rate: {health_summary['system_stability_score']}%")
    print(f"Estimated downtime prevented: {health_summary['estimated_total_downtime_prevented']}")

if __name__ == "__main__":
    main()

