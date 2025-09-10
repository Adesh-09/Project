from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import json
import threading
import time
from datetime import datetime, timedelta
from anomaly_detection import SparkAnomalyDetector
from proactive_mitigation import ProactiveMitigationEngine
import redis
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global instances
anomaly_detector = None
mitigation_engine = None
redis_client = None

def initialize_services():
    """Initialize all services"""
    global anomaly_detector, mitigation_engine, redis_client
    
    try:
        # Initialize Spark Anomaly Detector
        anomaly_detector = SparkAnomalyDetector()
        logger.info("Spark Anomaly Detector initialized")
        
        # Initialize Proactive Mitigation Engine
        mitigation_engine = ProactiveMitigationEngine()
        logger.info("Proactive Mitigation Engine initialized")
        
        # Initialize Redis client
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()  # Test connection
        logger.info("Redis client initialized")
        
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_client.ping()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'anomaly_detector': 'running',
                'mitigation_engine': 'running',
                'redis': 'connected'
            }
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/anomalies/detect', methods=['POST'])
def detect_anomalies():
    """Endpoint to trigger anomaly detection"""
    try:
        data = request.get_json()
        data_source = data.get('data_source', 'sample')
        
        # Run anomaly detection
        results_df, metrics, alerts = anomaly_detector.run_batch_detection(data_source)
        
        # Store results in Redis for caching
        cache_key = f"anomaly_results:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        redis_client.setex(cache_key, 3600, json.dumps(metrics))  # Cache for 1 hour
        
        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'alerts_count': len(alerts),
            'cache_key': cache_key,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/anomalies/results/<cache_key>', methods=['GET'])
def get_anomaly_results(cache_key):
    """Get cached anomaly detection results"""
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return jsonify({
                'status': 'success',
                'data': json.loads(cached_data),
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'status': 'not_found',
                'message': 'Results not found or expired',
                'timestamp': datetime.now().isoformat()
            }), 404
            
    except Exception as e:
        logger.error(f"Error retrieving results: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/mitigation/process', methods=['POST'])
def process_mitigation():
    """Endpoint to process anomaly and trigger mitigation"""
    try:
        anomaly_data = request.get_json()
        
        if not anomaly_data:
            return jsonify({
                'status': 'error',
                'error': 'No anomaly data provided',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Process the anomaly
        result = mitigation_engine.process_anomaly(anomaly_data)
        
        # Store mitigation result in Redis
        cache_key = f"mitigation_result:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        redis_client.setex(cache_key, 7200, json.dumps(result, default=str))  # Cache for 2 hours
        
        return jsonify({
            'status': 'success',
            'result': result,
            'cache_key': cache_key,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in mitigation processing: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/mitigation/history', methods=['GET'])
def get_mitigation_history():
    """Get mitigation history"""
    try:
        hours = request.args.get('hours', 24, type=int)
        history = mitigation_engine.get_mitigation_history(hours)
        
        return jsonify({
            'status': 'success',
            'history': history,
            'count': len(history),
            'hours': hours,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving mitigation history: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/system/health', methods=['GET'])
def get_system_health():
    """Get system health summary"""
    try:
        health_summary = mitigation_engine.get_system_health_summary()
        
        # Add additional system metrics
        health_summary.update({
            'uptime': get_system_uptime(),
            'active_alerts': get_active_alerts_count(),
            'last_anomaly_check': get_last_anomaly_check(),
            'system_load': get_system_load()
        })
        
        return jsonify({
            'status': 'success',
            'health_summary': health_summary,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving system health: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/metrics/prometheus', methods=['GET'])
def get_prometheus_metrics():
    """Get metrics in Prometheus format"""
    try:
        # Get system health summary
        health_summary = mitigation_engine.get_system_health_summary()
        
        # Format as Prometheus metrics
        metrics = []
        metrics.append(f"# HELP system_mitigations_total Total number of mitigations in the last 24 hours")
        metrics.append(f"# TYPE system_mitigations_total counter")
        metrics.append(f"system_mitigations_total {health_summary['total_mitigations_24h']}")
        
        metrics.append(f"# HELP system_stability_score System stability score (0-100)")
        metrics.append(f"# TYPE system_stability_score gauge")
        metrics.append(f"system_stability_score {health_summary['system_stability_score']}")
        
        metrics.append(f"# HELP system_successful_mitigations_total Successful mitigations in the last 24 hours")
        metrics.append(f"# TYPE system_successful_mitigations_total counter")
        metrics.append(f"system_successful_mitigations_total {health_summary['successful_mitigations']}")
        
        return '\n'.join(metrics), 200, {'Content-Type': 'text/plain'}
        
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {str(e)}")
        return f"# Error generating metrics: {str(e)}", 500, {'Content-Type': 'text/plain'}

@app.route('/api/dashboard/data', methods=['GET'])
def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        # Get recent anomaly results from Redis
        anomaly_keys = redis_client.keys("anomaly_results:*")
        recent_anomalies = []
        
        for key in sorted(anomaly_keys, reverse=True)[:10]:  # Get last 10 results
            data = redis_client.get(key)
            if data:
                recent_anomalies.append(json.loads(data))
        
        # Get mitigation history
        mitigation_history = mitigation_engine.get_mitigation_history(24)
        
        # Get system health
        health_summary = mitigation_engine.get_system_health_summary()
        
        dashboard_data = {
            'recent_anomalies': recent_anomalies,
            'mitigation_history': mitigation_history[-20:],  # Last 20 mitigations
            'health_summary': health_summary,
            'system_status': {
                'uptime': get_system_uptime(),
                'active_alerts': get_active_alerts_count(),
                'last_update': datetime.now().isoformat()
            }
        }
        
        return jsonify({
            'status': 'success',
            'dashboard_data': dashboard_data,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving dashboard data: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def get_system_uptime():
    """Get system uptime (simulated)"""
    # In production, this would get actual system uptime
    return "2 days, 14 hours, 32 minutes"

def get_active_alerts_count():
    """Get count of active alerts (simulated)"""
    # In production, this would query Prometheus/Alertmanager
    return 3

def get_last_anomaly_check():
    """Get timestamp of last anomaly check"""
    return datetime.now().isoformat()

def get_system_load():
    """Get system load metrics (simulated)"""
    return {
        'cpu_avg': 45.2,
        'memory_avg': 62.8,
        'disk_usage': 78.5
    }

def background_anomaly_detection():
    """Background task for continuous anomaly detection"""
    logger.info("Starting background anomaly detection")
    
    while True:
        try:
            # Run anomaly detection every 5 minutes
            logger.info("Running scheduled anomaly detection")
            results_df, metrics, alerts = anomaly_detector.run_batch_detection()
            
            # Store results in Redis
            cache_key = f"scheduled_anomaly_results:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            redis_client.setex(cache_key, 3600, json.dumps(metrics))
            
            # Process any high-severity anomalies automatically
            if alerts:
                for alert in alerts:
                    if alert.get('severity') == 'high':
                        logger.info(f"Processing high-severity alert: {alert['description']}")
                        # Convert alert to anomaly data format
                        anomaly_data = {
                            'server_id': alert['server_id'],
                            'timestamp': alert['timestamp'],
                            'metrics': alert['metrics']
                        }
                        mitigation_engine.process_anomaly(anomaly_data)
            
            logger.info(f"Scheduled anomaly detection completed. {len(alerts)} alerts generated.")
            
        except Exception as e:
            logger.error(f"Error in background anomaly detection: {str(e)}")
        
        # Wait 5 minutes before next run
        time.sleep(300)

if __name__ == '__main__':
    try:
        # Initialize services
        initialize_services()
        
        # Start background anomaly detection in a separate thread
        background_thread = threading.Thread(target=background_anomaly_detection, daemon=True)
        background_thread.start()
        
        # Start Flask app
        logger.info("Starting monitoring API server")
        app.run(host='0.0.0.0', port=8000, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start monitoring API: {str(e)}")
        if anomaly_detector:
            anomaly_detector.stop()
    finally:
        if anomaly_detector:
            anomaly_detector.stop()

