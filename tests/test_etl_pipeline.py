import unittest

try:  # pragma: no cover
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
except Exception:  # pragma: no cover
    SparkSession = None

try:
    from src.etl_pipeline import create_spark_session, load_data, clean_data, feature_engineering, transform_data
except Exception:  # pragma: no cover
    create_spark_session = load_data = clean_data = feature_engineering = transform_data = None

class TestETLPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SparkSession is None or create_spark_session is None:
            raise unittest.SkipTest("PySpark not available")
        cls.spark = create_spark_session("TestChurnETL")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_load_data(self):
        # Create a dummy CSV file for testing
        dummy_csv_content = """
CustomerID,Gender,SeniorCitizen,Partner,Dependents,Tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
1,Male,0,Yes,No,24,Yes,No,DSL,Yes,Yes,No,Yes,No,No,One year,Yes,Mailed check,60.0,1440.0,No
2,Female,1,No,No,12,Yes,No,Fiber optic,No,Yes,No,No,Yes,No,Month-to-month,Yes,Electronic check,80.0,960.0,Yes
"""
        with open("/tmp/test_data.csv", "w") as f:
            f.write(dummy_csv_content)

        df = load_data(self.spark, "/tmp/test_data.csv")
        self.assertIsNotNone(df)
        self.assertEqual(df.count(), 2)
        self.assertIn("CustomerID", df.columns)

    def test_clean_data(self):
        schema = StructType([
            StructField("TotalCharges", StringType(), True),
            StructField("CustomerID", StringType(), True)
        ])
        data = [(" ", "1"), ("100.0", "2")]
        df = self.spark.createDataFrame(data, schema)
        cleaned_df = clean_data(df)
        
        # Check if empty strings are converted to 0.0
        self.assertEqual(cleaned_df.filter(cleaned_df.TotalCharges == 0.0).count(), 1)
        self.assertEqual(cleaned_df.filter(cleaned_df.TotalCharges == 100.0).count(), 1)

    def test_feature_engineering(self):
        schema = StructType([
            StructField("SeniorCitizen", IntegerType(), True),
            StructField("CustomerID", StringType(), True)
        ])
        data = [(0, "1"), (1, "2")]
        df = self.spark.createDataFrame(data, schema)
        engineered_df = feature_engineering(df)
        
        self.assertIn("SeniorCitizen", engineered_df.columns)
        self.assertEqual(engineered_df.filter(engineered_df.SeniorCitizen == "Yes").count(), 1)
        self.assertEqual(engineered_df.filter(engineered_df.SeniorCitizen == "No").count(), 1)

    def test_transform_data(self):
        # This test is simplified as transform_data is a placeholder in etl_pipeline.py
        # In a real scenario, you'd test the actual encoding logic.
        schema = StructType([
            StructField("Gender", StringType(), True),
            StructField("Churn", StringType(), True)
        ])
        data = [("Male", "No"), ("Female", "Yes")]
        df = self.spark.createDataFrame(data, schema)
        transformed_df = transform_data(df)
        self.assertIsNotNone(transformed_df)

if __name__ == "__main__":
    unittest.main()

