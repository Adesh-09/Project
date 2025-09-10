from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, stddev, abs as spark_abs, current_timestamp, window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import IsolationForest
from pyspark.streaming import StreamingContext
import logging
import json
from datetime import datetime
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SparkAnomalyDetector:
    """
    Spark-based anomaly detection system for real-time system monitoring
    """
    
    def __init__(self, app_name="SystemMonitoringAnomalyDetection"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # Anomaly detection models
        self.isolation_forest = None
        self.kmeans_model = None
        self.statistical_thresholds = {}
        
        # Prometheus integration
        self.prometheus_gateway = "http://localhost:9091"
        
        logger.info("SparkAnomalyDetector initialized")
    
    def create_sample_system_metrics(self):
        """Generate sample system metrics for demonstration"""
        import random
        from datetime import datetime, timedelta
        
        metrics = []
        base_time = datetime.now()
        
        for i in range(1000):
            timestamp = base_time - timedelta(minutes=i)
            
            # Normal system behavior with some noise
            cpu_usage = random.normalvariate(45, 10)  # Normal around 45%
            memory_usage = random.normalvariate(60, 15)  # Normal around 60%
            disk_io = random.normalvariate(100, 20)  # Normal around 100 MB/s
            network_io = random.normalvariate(50, 10)  # Normal around 50 MB/s
            response_time = random.normalvariate(200, 50)  # Normal around 200ms
            error_rate = random.normalvariate(0.5, 0.2)  # Normal around 0.5%
            
            # Inject some anomalies (5% of data)
            if random.random() < 0.05:
                anomaly_type = random.choice(['cpu_spike', 'memory_leak', 'disk_bottleneck', 'network_congestion', 'high_latency', 'error_burst'])
                
                if anomaly_type == 'cpu_spike':
                    cpu_usage = random.uniform(85, 100)
                elif anomaly_type == 'memory_leak':
                    memory_usage = random.uniform(85, 98)
                elif anomaly_type == 'disk_bottleneck':
                    disk_io = random.uniform(500, 1000)
                elif anomaly_type == 'network_congestion':
                    network_io = random.uniform(200, 500)
                elif anomaly_type == 'high_latency':
                    response_time = random.uniform(1000, 5000)
                elif anomaly_type == 'error_burst':
                    error_rate = random.uniform(5, 20)
            
            # Ensure values are within reasonable bounds
            cpu_usage = max(0, min(100, cpu_usage))
            memory_usage = max(0, min(100, memory_usage))
            disk_io = max(0, disk_io)
            network_io = max(0, network_io)
            response_time = max(0, response_time)
            error_rate = max(0, error_rate)
            
            metrics.append({
                'timestamp': timestamp,
                'server_id': f'server_{random.randint(1, 10)}',
                'cpu_usage': round(cpu_usage, 2),
                'memory_usage': round(memory_usage, 2),
                'disk_io_mbps': round(disk_io, 2),
                'network_io_mbps': round(network_io, 2),
                'response_time_ms': round(response_time, 2),
                'error_rate_percent': round(error_rate, 2),
                'active_connections': random.randint(50, 500),
                'queue_length': random.randint(0, 100)
            })
        
        return metrics
    
    def load_system_metrics(self, data_source="sample"):
        """Load system metrics from various sources"""
        if data_source == "sample":
            # Generate sample data for demonstration
            sample_data = self.create_sample_system_metrics()
            df = self.spark.createDataFrame(sample_data)
            logger.info(f"Loaded {df.count()} sample system metrics")
            return df
        else:
            # In production, this would load from Kafka, Kinesis, or other streaming sources
            schema = StructType([
                StructField("timestamp", TimestampType(), True),
                StructField("server_id", StringType(), True),
                StructField("cpu_usage", DoubleType(), True),
                StructField("memory_usage", DoubleType(), True),
                StructField("disk_io_mbps", DoubleType(), True),
                StructField("network_io_mbps", DoubleType(), True),
                StructField("response_time_ms", DoubleType(), True),
                StructField("error_rate_percent", DoubleType(), True),
                StructField("active_connections", IntegerType(), True),
                StructField("queue_length", IntegerType(), True)
            ])
            
            # Example: Load from CSV file
            df = self.spark.read.csv(data_source, header=True, schema=schema)
            return df
    
    def calculate_statistical_thresholds(self, df):
        """Calculate statistical thresholds for anomaly detection"""
        logger.info("Calculating statistical thresholds")
        
        numeric_columns = ['cpu_usage', 'memory_usage', 'disk_io_mbps', 
                          'network_io_mbps', 'response_time_ms', 'error_rate_percent',
                          'active_connections', 'queue_length']
        
        for col_name in numeric_columns:
            stats = df.select(
                avg(col(col_name)).alias('mean'),
                stddev(col(col_name)).alias('stddev')
            ).collect()[0]
            
            mean_val = stats['mean']
            stddev_val = stats['stddev'] if stats['stddev'] is not None else 0
            
            # Define thresholds as mean ± 3 * standard deviation
            self.statistical_thresholds[col_name] = {
                'mean': mean_val,
                'stddev': stddev_val,
                'upper_threshold': mean_val + 3 * stddev_val,
                'lower_threshold': max(0, mean_val - 3 * stddev_val)
            }
        
        logger.info(f"Statistical thresholds calculated for {len(numeric_columns)} metrics")
        return self.statistical_thresholds
    
    def detect_statistical_anomalies(self, df):
        """Detect anomalies using statistical methods"""
        logger.info("Detecting statistical anomalies")
        
        anomaly_conditions = []
        
        for col_name, thresholds in self.statistical_thresholds.items():
            upper_threshold = thresholds['upper_threshold']
            lower_threshold = thresholds['lower_threshold']
            
            anomaly_conditions.append(
                (col(col_name) > upper_threshold) | (col(col_name) < lower_threshold)
            )
        
        # Combine all conditions with OR
        combined_condition = anomaly_conditions[0]
        for condition in anomaly_conditions[1:]:
            combined_condition = combined_condition | condition
        
        # Add anomaly flag
        df_with_anomalies = df.withColumn("is_statistical_anomaly", when(combined_condition, 1).otherwise(0))
        
        anomaly_count = df_with_anomalies.filter(col("is_statistical_anomaly") == 1).count()
        total_count = df_with_anomalies.count()
        
        logger.info(f"Statistical anomalies detected: {anomaly_count}/{total_count} ({anomaly_count/total_count*100:.2f}%)")
        
        return df_with_anomalies
    
    def train_isolation_forest(self, df):
        """Train Isolation Forest model for anomaly detection"""
        logger.info("Training Isolation Forest model")
        
        # Prepare features
        feature_columns = ['cpu_usage', 'memory_usage', 'disk_io_mbps', 
                          'network_io_mbps', 'response_time_ms', 'error_rate_percent',
                          'active_connections', 'queue_length']
        
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        df_features = assembler.transform(df)
        
        # Scale features
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        scaler_model = scaler.fit(df_features)
        df_scaled = scaler_model.transform(df_features)
        
        # Train Isolation Forest
        # Note: PySpark doesn't have native Isolation Forest, so we'll use a simplified approach
        # In production, you might use scikit-learn's IsolationForest with Spark UDFs
        # or implement a custom Isolation Forest algorithm
        
        # For demonstration, we'll use KMeans as a proxy for anomaly detection
        kmeans = KMeans(featuresCol="scaled_features", predictionCol="cluster", k=5, seed=42)
        self.kmeans_model = kmeans.fit(df_scaled)
        
        # Calculate distances from cluster centers for anomaly scoring
        df_with_clusters = self.kmeans_model.transform(df_scaled)
        
        logger.info("Isolation Forest (KMeans proxy) model trained")
        return df_with_clusters, scaler_model
    
    def detect_ml_anomalies(self, df, scaler_model):
        """Detect anomalies using machine learning models"""
        logger.info("Detecting ML-based anomalies")
        
        # Prepare features
        feature_columns = ['cpu_usage', 'memory_usage', 'disk_io_mbps', 
                          'network_io_mbps', 'response_time_ms', 'error_rate_percent',
                          'active_connections', 'queue_length']
        
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        df_features = assembler.transform(df)
        
        # Scale features
        df_scaled = scaler_model.transform(df_features)
        
        # Apply KMeans model
        df_with_clusters = self.kmeans_model.transform(df_scaled)
        
        # Calculate anomaly scores based on distance to cluster centers
        # This is a simplified approach - in production, you'd use proper anomaly scoring
        df_with_anomaly_score = df_with_clusters.withColumn(
            "is_ml_anomaly", 
            when(col("cluster") == 4, 1).otherwise(0)  # Assume cluster 4 represents anomalies
        )
        
        anomaly_count = df_with_anomaly_score.filter(col("is_ml_anomaly") == 1).count()
        total_count = df_with_anomaly_score.count()
        
        logger.info(f"ML anomalies detected: {anomaly_count}/{total_count} ({anomaly_count/total_count*100:.2f}%)")
        
        return df_with_anomaly_score
    
    def combine_anomaly_detections(self, df):
        """Combine different anomaly detection methods"""
        logger.info("Combining anomaly detection results")
        
        # Create final anomaly flag
        df_final = df.withColumn(
            "is_anomaly",
            when((col("is_statistical_anomaly") == 1) | (col("is_ml_anomaly") == 1), 1).otherwise(0)
        )
        
        # Add anomaly severity
        df_final = df_final.withColumn(
            "anomaly_severity",
            when((col("is_statistical_anomaly") == 1) & (col("is_ml_anomaly") == 1), "high")
            .when((col("is_statistical_anomaly") == 1) | (col("is_ml_anomaly") == 1), "medium")
            .otherwise("low")
        )
        
        return df_final
    
    def send_to_prometheus(self, anomaly_metrics):
        """Send anomaly metrics to Prometheus Push Gateway"""
        try:
            # Prepare metrics for Prometheus
            metrics_data = {
                'job': 'anomaly_detection',
                'instance': 'spark_detector',
                'metrics': anomaly_metrics
            }
            
            # In a real implementation, you would use the Prometheus Python client
            # For demonstration, we'll just log the metrics
            logger.info(f"Sending metrics to Prometheus: {json.dumps(metrics_data, indent=2)}")
            
            # Example of what the actual implementation might look like:
            # from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
            # registry = CollectorRegistry()
            # anomaly_gauge = Gauge('system_anomalies_total', 'Total number of system anomalies', registry=registry)
            # anomaly_gauge.set(anomaly_metrics['total_anomalies'])
            # push_to_gateway(self.prometheus_gateway, job='anomaly_detection', registry=registry)
            
        except Exception as e:
            logger.error(f"Failed to send metrics to Prometheus: {str(e)}")
    
    def generate_alerts(self, anomalies_df):
        """Generate alerts for detected anomalies"""
        logger.info("Generating alerts for anomalies")
        
        # Get high severity anomalies
        high_severity_anomalies = anomalies_df.filter(col("anomaly_severity") == "high").collect()
        
        alerts = []
        for anomaly in high_severity_anomalies:
            alert = {
                'timestamp': anomaly['timestamp'].isoformat(),
                'server_id': anomaly['server_id'],
                'severity': anomaly['anomaly_severity'],
                'metrics': {
                    'cpu_usage': anomaly['cpu_usage'],
                    'memory_usage': anomaly['memory_usage'],
                    'disk_io_mbps': anomaly['disk_io_mbps'],
                    'network_io_mbps': anomaly['network_io_mbps'],
                    'response_time_ms': anomaly['response_time_ms'],
                    'error_rate_percent': anomaly['error_rate_percent']
                },
                'alert_type': 'system_anomaly',
                'description': f"High severity anomaly detected on {anomaly['server_id']}"
            }
            alerts.append(alert)
        
        logger.info(f"Generated {len(alerts)} high severity alerts")
        return alerts
    
    def run_batch_detection(self, data_source="sample"):
        """Run batch anomaly detection"""
        logger.info("Starting batch anomaly detection")
        
        try:
            # Load data
            df = self.load_system_metrics(data_source)
            
            # Calculate statistical thresholds
            self.calculate_statistical_thresholds(df)
            
            # Detect statistical anomalies
            df_stat_anomalies = self.detect_statistical_anomalies(df)
            
            # Train and apply ML models
            df_with_clusters, scaler_model = self.train_isolation_forest(df)
            df_ml_anomalies = self.detect_ml_anomalies(df_stat_anomalies, scaler_model)
            
            # Combine anomaly detections
            df_final = self.combine_anomaly_detections(df_ml_anomalies)
            
            # Generate summary metrics
            total_records = df_final.count()
            total_anomalies = df_final.filter(col("is_anomaly") == 1).count()
            high_severity_anomalies = df_final.filter(col("anomaly_severity") == "high").count()
            
            anomaly_metrics = {
                'total_records': total_records,
                'total_anomalies': total_anomalies,
                'high_severity_anomalies': high_severity_anomalies,
                'anomaly_rate': total_anomalies / total_records if total_records > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Send metrics to Prometheus
            self.send_to_prometheus(anomaly_metrics)
            
            # Generate alerts
            alerts = self.generate_alerts(df_final)
            
            # Save results
            output_path = "/home/ubuntu/projects/system_monitoring_pipeline/data/anomaly_results"
            df_final.write.mode("overwrite").parquet(output_path)
            
            logger.info(f"Batch anomaly detection completed. Results saved to {output_path}")
            logger.info(f"Summary: {total_anomalies}/{total_records} anomalies detected ({anomaly_metrics['anomaly_rate']*100:.2f}%)")
            
            return df_final, anomaly_metrics, alerts
            
        except Exception as e:
            logger.error(f"Error in batch anomaly detection: {str(e)}")
            raise
    
    def stop(self):
        """Stop the Spark session"""
        self.spark.stop()
        logger.info("SparkAnomalyDetector stopped")

def main():
    """Main function for running anomaly detection"""
    detector = SparkAnomalyDetector()
    
    try:
        # Run batch detection
        results_df, metrics, alerts = detector.run_batch_detection()
        
        # Display results
        print("\n=== Anomaly Detection Results ===")
        print(f"Total records processed: {metrics['total_records']}")
        print(f"Total anomalies detected: {metrics['total_anomalies']}")
        print(f"High severity anomalies: {metrics['high_severity_anomalies']}")
        print(f"Anomaly rate: {metrics['anomaly_rate']*100:.2f}%")
        
        if alerts:
            print(f"\n=== High Severity Alerts ({len(alerts)}) ===")
            for i, alert in enumerate(alerts[:5]):  # Show first 5 alerts
                print(f"Alert {i+1}: {alert['description']} at {alert['timestamp']}")
        
        # Show sample anomalies
        print("\n=== Sample Anomalies ===")
        anomalies_sample = results_df.filter(col("is_anomaly") == 1).limit(5)
        anomalies_sample.select("timestamp", "server_id", "cpu_usage", "memory_usage", 
                               "response_time_ms", "anomaly_severity").show(truncate=False)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
    finally:
        detector.stop()

if __name__ == "__main__":
    main()

