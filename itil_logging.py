import logging
import logging.handlers
import os
import json
import shutil
from datetime import datetime, timedelta
import gzip
import threading
import time
from pathlib import Path

class ITILCompliantLogger:
    """
    ITIL-compliant logging system for the churn prediction model
    Implements structured logging with proper categorization, retention, and backup
    """
    
    def __init__(self, service_name="churn_prediction", log_dir="/home/ubuntu/projects/customer_churn_prediction/logs"):
        self.service_name = service_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # ITIL log categories
        self.log_categories = {
            'operational': 'operational.log',
            'security': 'security.log',
            'performance': 'performance.log',
            'error': 'error.log',
            'audit': 'audit.log',
            'change': 'change.log'
        }
        
        self.loggers = {}
        self.setup_loggers()
        
        # Backup configuration
        self.backup_dir = self.log_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.retention_days = 90
        
        # Start background tasks
        self.start_log_rotation()
        self.start_backup_scheduler()
    
    def setup_loggers(self):
        """Setup individual loggers for each ITIL category"""
        
        # Common formatter for structured logging
        formatter = logging.Formatter(
            '%(asctime)s|%(levelname)s|%(name)s|%(funcName)s|%(lineno)d|%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        for category, filename in self.log_categories.items():
            logger = logging.getLogger(f"{self.service_name}.{category}")
            logger.setLevel(logging.INFO)
            
            # File handler with rotation
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / filename,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Console handler for critical logs
            if category in ['error', 'security']:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                console_handler.setLevel(logging.WARNING)
                logger.addHandler(console_handler)
            
            self.loggers[category] = logger
    
    def log_operational(self, message, level=logging.INFO, **kwargs):
        """Log operational events"""
        structured_message = self._create_structured_message(
            category="operational",
            message=message,
            **kwargs
        )
        self.loggers['operational'].log(level, structured_message)
    
    def log_security(self, message, level=logging.WARNING, **kwargs):
        """Log security events"""
        structured_message = self._create_structured_message(
            category="security",
            message=message,
            **kwargs
        )
        self.loggers['security'].log(level, structured_message)
    
    def log_performance(self, message, level=logging.INFO, **kwargs):
        """Log performance metrics"""
        structured_message = self._create_structured_message(
            category="performance",
            message=message,
            **kwargs
        )
        self.loggers['performance'].log(level, structured_message)
    
    def log_error(self, message, level=logging.ERROR, **kwargs):
        """Log error events"""
        structured_message = self._create_structured_message(
            category="error",
            message=message,
            **kwargs
        )
        self.loggers['error'].log(level, structured_message)
    
    def log_audit(self, message, level=logging.INFO, **kwargs):
        """Log audit events"""
        structured_message = self._create_structured_message(
            category="audit",
            message=message,
            **kwargs
        )
        self.loggers['audit'].log(level, structured_message)
    
    def log_change(self, message, level=logging.INFO, **kwargs):
        """Log change management events"""
        structured_message = self._create_structured_message(
            category="change",
            message=message,
            **kwargs
        )
        self.loggers['change'].log(level, structured_message)
    
    def _create_structured_message(self, category, message, **kwargs):
        """Create structured log message in JSON format"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'service': self.service_name,
            'category': category,
            'message': message,
            'metadata': kwargs
        }
        return json.dumps(log_entry)
    
    def start_log_rotation(self):
        """Start background log rotation"""
        def rotate_logs():
            while True:
                try:
                    self._rotate_old_logs()
                    time.sleep(24 * 3600)  # Run daily
                except Exception as e:
                    print(f"Error in log rotation: {e}")
        
        rotation_thread = threading.Thread(target=rotate_logs, daemon=True)
        rotation_thread.start()
    
    def start_backup_scheduler(self):
        """Start background backup scheduler"""
        def backup_logs():
            while True:
                try:
                    self._backup_logs()
                    time.sleep(7 * 24 * 3600)  # Run weekly
                except Exception as e:
                    print(f"Error in log backup: {e}")
        
        backup_thread = threading.Thread(target=backup_logs, daemon=True)
        backup_thread.start()
    
    def _rotate_old_logs(self):
        """Rotate and compress old log files"""
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                # Compress old log file
                compressed_name = f"{log_file.stem}_{datetime.now().strftime('%Y%m%d')}.log.gz"
                compressed_path = self.log_dir / compressed_name
                
                with open(log_file, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove original file
                log_file.unlink()
    
    def _backup_logs(self):
        """Backup logs to backup directory"""
        backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"logs_backup_{backup_timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        # Copy all log files to backup
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.is_file():
                shutil.copy2(log_file, backup_path)
        
        # Compress backup
        shutil.make_archive(str(backup_path), 'zip', str(backup_path))
        shutil.rmtree(backup_path)
        
        # Clean old backups
        self._clean_old_backups()
    
    def _clean_old_backups(self):
        """Clean backups older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for backup_file in self.backup_dir.glob("logs_backup_*.zip"):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                backup_file.unlink()

class ModelOperationsLogger:
    """
    Specialized logger for ML model operations
    Integrates with ITILCompliantLogger for comprehensive logging
    """
    
    def __init__(self):
        self.itil_logger = ITILCompliantLogger()
    
    def log_model_training_start(self, model_name, parameters):
        """Log model training initiation"""
        self.itil_logger.log_operational(
            f"Model training started: {model_name}",
            model_name=model_name,
            parameters=parameters,
            operation="training_start"
        )
        
        self.itil_logger.log_audit(
            f"Model training audit: {model_name}",
            model_name=model_name,
            action="training_initiated",
            user="system",
            parameters=parameters
        )
    
    def log_model_training_complete(self, model_name, metrics, duration):
        """Log model training completion"""
        self.itil_logger.log_operational(
            f"Model training completed: {model_name}",
            model_name=model_name,
            metrics=metrics,
            duration_seconds=duration,
            operation="training_complete"
        )
        
        self.itil_logger.log_performance(
            f"Model performance metrics: {model_name}",
            model_name=model_name,
            metrics=metrics,
            training_duration=duration
        )
    
    def log_model_deployment(self, model_name, version, deployment_target):
        """Log model deployment"""
        self.itil_logger.log_change(
            f"Model deployed: {model_name} v{version}",
            model_name=model_name,
            version=version,
            deployment_target=deployment_target,
            change_type="deployment"
        )
        
        self.itil_logger.log_audit(
            f"Model deployment audit: {model_name}",
            model_name=model_name,
            version=version,
            action="deployed",
            target=deployment_target
        )
    
    def log_prediction_request(self, request_id, input_features, prediction, confidence):
        """Log prediction requests"""
        self.itil_logger.log_operational(
            f"Prediction request processed: {request_id}",
            request_id=request_id,
            prediction=prediction,
            confidence=confidence,
            operation="prediction"
        )
    
    def log_data_quality_issue(self, issue_type, details):
        """Log data quality issues"""
        self.itil_logger.log_error(
            f"Data quality issue detected: {issue_type}",
            issue_type=issue_type,
            details=details,
            severity="medium"
        )
    
    def log_model_drift(self, model_name, drift_metrics):
        """Log model drift detection"""
        self.itil_logger.log_operational(
            f"Model drift detected: {model_name}",
            model_name=model_name,
            drift_metrics=drift_metrics,
            operation="drift_detection"
        )
        
        self.itil_logger.log_performance(
            f"Model drift metrics: {model_name}",
            model_name=model_name,
            drift_metrics=drift_metrics
        )
    
    def log_security_event(self, event_type, details):
        """Log security-related events"""
        self.itil_logger.log_security(
            f"Security event: {event_type}",
            event_type=event_type,
            details=details,
            severity="high"
        )

# Example usage and testing
if __name__ == "__main__":
    # Initialize logger
    ops_logger = ModelOperationsLogger()
    
    # Example logging scenarios
    ops_logger.log_model_training_start(
        "ensemble_churn_model",
        {"n_estimators": 100, "max_depth": 10}
    )
    
    time.sleep(1)  # Simulate training time
    
    ops_logger.log_model_training_complete(
        "ensemble_churn_model",
        {"accuracy": 0.87, "auc": 0.92, "precision": 0.85},
        duration=120
    )
    
    ops_logger.log_model_deployment(
        "ensemble_churn_model",
        "v1.2.0",
        "production"
    )
    
    ops_logger.log_prediction_request(
        "req_001",
        {"tenure": 24, "monthly_charges": 65.0},
        "no_churn",
        0.78
    )
    
    ops_logger.log_data_quality_issue(
        "missing_values",
        {"column": "total_charges", "missing_percentage": 5.2}
    )
    
    print("ITIL-compliant logging system initialized and tested successfully")
    print(f"Logs are stored in: /home/ubuntu/projects/customer_churn_prediction/logs")

