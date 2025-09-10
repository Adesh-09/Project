import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import logging
from datetime import datetime, timedelta
import schedule
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLflowAutoRetrainer:
    def __init__(self, experiment_name="churn_prediction_experiment"):
        self.client = MlflowClient()
        self.experiment_name = experiment_name
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = self.experiment.experiment_id
        
        self.performance_threshold = 0.85  # Minimum acceptable AUC
        self.retraining_interval_days = 7  # Retrain every 7 days
        
    def get_latest_model_performance(self):
        """Get the performance of the latest production model"""
        try:
            # Get the latest run from the experiment
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if not runs:
                logger.warning("No runs found in the experiment")
                return None
            
            latest_run = runs[0]
            ensemble_auc = latest_run.data.metrics.get('ensemble_auc', 0)
            
            logger.info(f"Latest model AUC: {ensemble_auc}")
            return ensemble_auc
            
        except Exception as e:
            logger.error(f"Error getting latest model performance: {str(e)}")
            return None
    
    def check_data_drift(self, new_data_path, reference_data_path):
        """Check for data drift between new data and reference data"""
        try:
            spark = SparkSession.builder \
                .appName("DataDriftDetection") \
                .getOrCreate()
            
            # Load datasets
            new_df = spark.read.parquet(new_data_path)
            ref_df = spark.read.parquet(reference_data_path)
            
            # Simple drift detection based on statistical measures
            # In production, you might use more sophisticated methods like KS test, PSI, etc.
            
            # Check for significant changes in key metrics
            new_stats = new_df.describe(['MonthlyCharges', 'TotalCharges', 'Tenure']).collect()
            ref_stats = ref_df.describe(['MonthlyCharges', 'TotalCharges', 'Tenure']).collect()
            
            drift_detected = False
            drift_threshold = 0.1  # 10% change threshold
            
            for i, col in enumerate(['MonthlyCharges', 'TotalCharges', 'Tenure']):
                new_mean = float(new_stats[1][i+1])  # Mean is at index 1
                ref_mean = float(ref_stats[1][i+1])
                
                if abs(new_mean - ref_mean) / ref_mean > drift_threshold:
                    drift_detected = True
                    logger.warning(f"Data drift detected in {col}: {ref_mean} -> {new_mean}")
            
            spark.stop()
            return drift_detected
            
        except Exception as e:
            logger.error(f"Error checking data drift: {str(e)}")
            return False
    
    def trigger_retraining(self, data_path):
        """Trigger model retraining"""
        logger.info("Triggering model retraining")
        
        try:
            # Import the training module
            from model_training import ChurnPredictionModel
            
            spark = SparkSession.builder \
                .appName("AutoRetraining") \
                .config("spark.sql.shuffle.partitions", "4") \
                .getOrCreate()
            
            with mlflow.start_run(experiment_id=self.experiment_id):
                # Log retraining trigger
                mlflow.log_param("retraining_trigger", "automated")
                mlflow.log_param("retraining_timestamp", datetime.now().isoformat())
                
                # Load and prepare data
                df = spark.read.parquet(data_path)
                model_trainer = ChurnPredictionModel(spark)
                
                # Prepare features
                feature_pipeline = model_trainer.prepare_features(df)
                feature_model = feature_pipeline.fit(df)
                prepared_df = feature_model.transform(df)
                
                # Split data
                train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=42)
                
                # Train models
                predictions_dict = model_trainer.train_individual_models(train_df, test_df)
                ensemble_predictions = model_trainer.create_ensemble_prediction(predictions_dict, test_df)
                
                # Evaluate models
                results = model_trainer.evaluate_models(predictions_dict, ensemble_predictions)
                
                # Log results
                model_trainer.log_to_mlflow(results, model_trainer.models)
                
                # Check if new model is better
                new_auc = results['ensemble']['auc']
                current_best_auc = self.get_latest_model_performance()
                
                if current_best_auc is None or new_auc > current_best_auc:
                    # Register new model as production
                    for model_name, model in model_trainer.models.items():
                        model_path = f"/home/ubuntu/projects/customer_churn_prediction/models/{model_name}_retrained"
                        model.write().overwrite().save(model_path)
                    
                    mlflow.log_param("model_promoted", "yes")
                    logger.info(f"New model promoted to production. AUC improved from {current_best_auc} to {new_auc}")
                else:
                    mlflow.log_param("model_promoted", "no")
                    logger.info(f"New model not promoted. AUC {new_auc} not better than current {current_best_auc}")
            
            spark.stop()
            return True
            
        except Exception as e:
            logger.error(f"Error during retraining: {str(e)}")
            return False
    
    def check_and_retrain(self):
        """Main function to check conditions and trigger retraining if needed"""
        logger.info("Checking retraining conditions")
        
        data_path = "/home/ubuntu/projects/customer_churn_prediction/data/processed_churn_data.parquet"
        reference_data_path = "/home/ubuntu/projects/customer_churn_prediction/data/reference_data.parquet"
        
        # Check if data files exist
        if not os.path.exists(data_path.replace("file://", "")):
            logger.warning(f"Data file not found: {data_path}")
            return
        
        # Get current model performance
        current_performance = self.get_latest_model_performance()
        
        # Check for performance degradation
        performance_degraded = current_performance is not None and current_performance < self.performance_threshold
        
        # Check for data drift (if reference data exists)
        data_drift = False
        if os.path.exists(reference_data_path.replace("file://", "")):
            data_drift = self.check_data_drift(data_path, reference_data_path)
        
        # Check if it's time for scheduled retraining
        last_run_time = self.get_last_retraining_time()
        scheduled_retrain = (datetime.now() - last_run_time).days >= self.retraining_interval_days
        
        # Trigger retraining if any condition is met
        if performance_degraded or data_drift or scheduled_retrain:
            reasons = []
            if performance_degraded:
                reasons.append(f"performance degraded (AUC: {current_performance})")
            if data_drift:
                reasons.append("data drift detected")
            if scheduled_retrain:
                reasons.append("scheduled retraining")
            
            logger.info(f"Retraining triggered due to: {', '.join(reasons)}")
            success = self.trigger_retraining(data_path)
            
            if success:
                logger.info("Retraining completed successfully")
            else:
                logger.error("Retraining failed")
        else:
            logger.info("No retraining needed at this time")
    
    def get_last_retraining_time(self):
        """Get the timestamp of the last retraining"""
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string="params.retraining_trigger = 'automated'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if runs:
                return datetime.fromtimestamp(runs[0].info.start_time / 1000)
            else:
                return datetime.now() - timedelta(days=self.retraining_interval_days + 1)
                
        except Exception as e:
            logger.error(f"Error getting last retraining time: {str(e)}")
            return datetime.now() - timedelta(days=self.retraining_interval_days + 1)
    
    def start_scheduler(self):
        """Start the automated retraining scheduler"""
        logger.info("Starting automated retraining scheduler")
        
        # Schedule daily checks
        schedule.every().day.at("02:00").do(self.check_and_retrain)
        
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour

def main():
    """Main function for manual execution"""
    retrainer = MLflowAutoRetrainer()
    retrainer.check_and_retrain()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--scheduler":
        # Run as scheduler
        retrainer = MLflowAutoRetrainer()
        retrainer.start_scheduler()
    else:
        # Run manual check
        main()

