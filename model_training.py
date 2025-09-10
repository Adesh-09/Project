from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import mlflow
import mlflow.spark
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChurnPredictionModel:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.models = {}
        self.ensemble_predictions = None
        
    def prepare_features(self, df):
        """Prepare features for model training"""
        logger.info("Starting feature preparation")
        
        # Categorical columns to be indexed and encoded
        categorical_cols = ['Gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                           'PaperlessBilling', 'PaymentMethod']
        
        # Numerical columns
        numerical_cols = ['SeniorCitizen', 'Tenure', 'MonthlyCharges', 'TotalCharges']
        
        # String indexing for categorical variables
        indexers = [StringIndexer(inputCol=col, outputCol=col+"_indexed", handleInvalid="keep") 
                   for col in categorical_cols]
        
        # One-hot encoding
        encoders = [OneHotEncoder(inputCol=col+"_indexed", outputCol=col+"_encoded") 
                   for col in categorical_cols]
        
        # Target variable indexing
        label_indexer = StringIndexer(inputCol="Churn", outputCol="label")
        
        # Assemble all features
        feature_cols = [col+"_encoded" for col in categorical_cols] + numerical_cols
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        
        # Create pipeline for feature preparation
        feature_pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler])
        
        logger.info("Feature preparation pipeline created")
        return feature_pipeline
    
    def train_individual_models(self, train_df, test_df):
        """Train individual models for ensemble"""
        logger.info("Training individual models")
        
        # Logistic Regression
        lr = LogisticRegression(featuresCol="features", labelCol="label")
        lr_model = lr.fit(train_df)
        lr_predictions = lr_model.transform(test_df)
        self.models['logistic_regression'] = lr_model
        
        # Random Forest
        rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100)
        rf_model = rf.fit(train_df)
        rf_predictions = rf_model.transform(test_df)
        self.models['random_forest'] = rf_model
        
        # Gradient Boosted Trees
        gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=100)
        gbt_model = gbt.fit(train_df)
        gbt_predictions = gbt_model.transform(test_df)
        self.models['gradient_boosting'] = gbt_model
        
        logger.info("Individual models trained successfully")
        return {
            'lr_predictions': lr_predictions,
            'rf_predictions': rf_predictions,
            'gbt_predictions': gbt_predictions
        }
    
    def create_ensemble_prediction(self, predictions_dict, test_df):
        """Create ensemble predictions using voting"""
        logger.info("Creating ensemble predictions")
        
        # Simple voting ensemble - majority vote
        from pyspark.sql.functions import col, when, avg
        
        # Extract predictions from each model
        lr_pred = predictions_dict['lr_predictions'].select("CustomerID", col("prediction").alias("lr_pred"))
        rf_pred = predictions_dict['rf_predictions'].select("CustomerID", col("prediction").alias("rf_pred"))
        gbt_pred = predictions_dict['gbt_predictions'].select("CustomerID", col("prediction").alias("gbt_pred"))
        
        # Join predictions
        ensemble_df = lr_pred.join(rf_pred, "CustomerID").join(gbt_pred, "CustomerID")
        
        # Calculate ensemble prediction (majority vote)
        ensemble_df = ensemble_df.withColumn(
            "ensemble_prediction",
            when((col("lr_pred") + col("rf_pred") + col("gbt_pred")) >= 2, 1.0).otherwise(0.0)
        )
        
        # Join with original test data
        final_predictions = test_df.join(ensemble_df.select("CustomerID", "ensemble_prediction"), "CustomerID")
        
        logger.info("Ensemble predictions created")
        return final_predictions
    
    def evaluate_models(self, predictions_dict, ensemble_predictions):
        """Evaluate all models and ensemble"""
        logger.info("Evaluating models")
        
        evaluator_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
        evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        
        results = {}
        
        # Evaluate individual models
        for model_name, predictions in predictions_dict.items():
            auc = evaluator_auc.evaluate(predictions)
            accuracy = evaluator_acc.evaluate(predictions)
            results[model_name] = {'auc': auc, 'accuracy': accuracy}
            logger.info(f"{model_name} - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
        
        # Evaluate ensemble
        ensemble_auc = evaluator_auc.evaluate(ensemble_predictions.withColumn("prediction", col("ensemble_prediction")))
        ensemble_acc = evaluator_acc.evaluate(ensemble_predictions.withColumn("prediction", col("ensemble_prediction")))
        results['ensemble'] = {'auc': ensemble_auc, 'accuracy': ensemble_acc}
        logger.info(f"Ensemble - AUC: {ensemble_auc:.4f}, Accuracy: {ensemble_acc:.4f}")
        
        return results
    
    def log_to_mlflow(self, results, models):
        """Log models and metrics to MLflow"""
        logger.info("Logging to MLflow")
        
        with mlflow.start_run():
            # Log metrics
            for model_name, metrics in results.items():
                mlflow.log_metric(f"{model_name}_auc", metrics['auc'])
                mlflow.log_metric(f"{model_name}_accuracy", metrics['accuracy'])
            
            # Log models
            for model_name, model in models.items():
                mlflow.spark.log_model(model, f"{model_name}_model")
            
            # Log ensemble improvement
            baseline_acc = max([results[model]['accuracy'] for model in ['lr_predictions', 'rf_predictions', 'gbt_predictions']])
            ensemble_improvement = (results['ensemble']['accuracy'] - baseline_acc) / baseline_acc * 100
            mlflow.log_metric("ensemble_improvement_percent", ensemble_improvement)
            
            logger.info(f"Ensemble improvement: {ensemble_improvement:.2f}%")

def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("ChurnPredictionModelTraining") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    
    # Initialize MLflow
    mlflow.set_experiment("churn_prediction_experiment")
    
    try:
        # Load processed data
        data_path = "/home/ubuntu/projects/customer_churn_prediction/data/processed_churn_data.parquet"
        df = spark.read.parquet(data_path)
        
        # Initialize model trainer
        model_trainer = ChurnPredictionModel(spark)
        
        # Prepare features
        feature_pipeline = model_trainer.prepare_features(df)
        feature_model = feature_pipeline.fit(df)
        prepared_df = feature_model.transform(df)
        
        # Split data
        train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=42)
        
        logger.info(f"Training set size: {train_df.count()}")
        logger.info(f"Test set size: {test_df.count()}")
        
        # Train individual models
        predictions_dict = model_trainer.train_individual_models(train_df, test_df)
        
        # Create ensemble predictions
        ensemble_predictions = model_trainer.create_ensemble_prediction(predictions_dict, test_df)
        
        # Evaluate models
        results = model_trainer.evaluate_models(predictions_dict, ensemble_predictions)
        
        # Log to MLflow
        model_trainer.log_to_mlflow(results, model_trainer.models)
        
        # Save models
        for model_name, model in model_trainer.models.items():
            model_path = f"/home/ubuntu/projects/customer_churn_prediction/models/{model_name}"
            model.write().overwrite().save(model_path)
            logger.info(f"Model {model_name} saved to {model_path}")
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main()

