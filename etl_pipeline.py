
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

def create_spark_session(app_name="ChurnPredictionETL"):
    """Creates and returns a SparkSession."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    return spark

def load_data(spark, input_path):
    """Loads data from a specified path."""
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    return df

def clean_data(df):
    """Performs data cleaning and handles missing values."""
    # Example: Fill missing 'TotalCharges' with 0 for new customers
    df = df.withColumn("TotalCharges", when(col("TotalCharges").isNull(), 0).otherwise(col("TotalCharges")))
    return df

def feature_engineering(df):
    """Performs feature engineering."""
    # Example: Create a 'SeniorCitizen' category
    df = df.withColumn("SeniorCitizen", when(col("SeniorCitizen") == 1, "Yes").otherwise("No"))
    return df

def transform_data(df):
    """Applies necessary transformations for model training."""
    # Example: One-hot encode categorical features (simplified for brevity)
    # In a real scenario, use StringIndexer and OneHotEncoderEstimator from MLlib
    return df

def save_processed_data(df, output_path):
    """Saves the processed data to a specified path."""
    df.write.mode("overwrite").parquet(output_path)

if __name__ == "__main__":
    spark = create_spark_session()
    
    # For demonstration, let's assume a dummy input path
    # In a real scenario, this would be a path to your actual data
    dummy_input_path = "/home/ubuntu/projects/customer_churn_prediction/data/telecom_churn.csv"
    processed_output_path = "/home/ubuntu/projects/customer_churn_prediction/data/processed_churn_data.parquet"

    # Create a dummy CSV file for testing
    dummy_csv_content = """
CustomerID,Gender,SeniorCitizen,Partner,Dependents,Tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
1,Male,0,Yes,No,24,Yes,No,DSL,Yes,Yes,No,Yes,No,No,One year,Yes,Mailed check,60.0,1440.0,No
2,Female,1,No,No,12,Yes,No,Fiber optic,No,Yes,No,No,Yes,No,Month-to-month,Yes,Electronic check,80.0,960.0,Yes
3,Male,0,No,No,36,Yes,Yes,No,No,No,No,No,No,No,Two year,No,Bank transfer (automatic),20.0,720.0,No
4,Female,0,Yes,Yes,48,Yes,Yes,DSL,Yes,No,Yes,Yes,No,Yes,One year,Yes,Credit card (automatic),90.0,4320.0,No
5,Male,1,No,No,6,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,70.0,420.0,Yes
"""
    with open(dummy_input_path, "w") as f:
        f.write(dummy_csv_content)

    df = load_data(spark, dummy_input_path)
    df_cleaned = clean_data(df)
    df_engineered = feature_engineering(df_cleaned)
    df_transformed = transform_data(df_engineered)
    save_processed_data(df_transformed, processed_output_path)

    print(f"ETL pipeline completed. Processed data saved to {processed_output_path}")
    spark.stop()


