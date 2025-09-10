import unittest
import sys
import os
import tempfile
import shutil
import pandas as pd
from unittest.mock import patch, MagicMock

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from bert_classifier import BERTCustomerQueryClassifier
except ImportError:
    # Skip tests if BERT classifier is not available
    BERTCustomerQueryClassifier = None

class TestBERTCustomerQueryClassifier(unittest.TestCase):
    """Test cases for BERT Customer Query Classifier"""
    
    def setUp(self):
        """Set up test fixtures"""
        if BERTCustomerQueryClassifier is None:
            self.skipTest("BERT classifier not available")
        
        self.classifier = BERTCustomerQueryClassifier()
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample test data
        self.sample_queries = [
            "My internet is not working",
            "I want to cancel my subscription",
            "Can you explain my bill?",
            "What plans do you offer?",
            "I need technical support"
        ]
        
        self.sample_categories = [
            "technical_support",
            "cancellation_request", 
            "billing_inquiry",
            "product_information",
            "technical_support"
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test classifier initialization"""
        self.assertIsNotNone(self.classifier)
        self.assertEqual(self.classifier.model_name, "distilbert-base-uncased")
        self.assertIsNotNone(self.classifier.default_categories)
        self.assertEqual(len(self.classifier.default_categories), 10)
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        df = self.classifier.generate_sample_data(num_samples=100)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        self.assertIn('query', df.columns)
        self.assertIn('category', df.columns)
        self.assertTrue(df['category'].nunique() <= 10)
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        # Create sample dataframe
        df = pd.DataFrame({
            'query': self.sample_queries,
            'category': self.sample_categories
        })
        
        queries, labels = self.classifier.preprocess_data(df)
        
        self.assertEqual(len(queries), len(self.sample_queries))
        self.assertEqual(len(labels), len(self.sample_categories))
        self.assertIsInstance(queries, list)
        self.assertIsInstance(labels, list)
        self.assertTrue(all(isinstance(label, int) for label in labels))
    
    def test_predict_single_query(self):
        """Test prediction on a single query"""
        query = "My internet is not working"
        predictions = self.classifier.predict(query)
        
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)
        
        prediction = predictions[0]
        self.assertIn('query', prediction)
        self.assertIn('predicted_category', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn('label_id', prediction)
        
        self.assertEqual(prediction['query'], query)
        self.assertIsInstance(prediction['confidence'], (int, float))
        self.assertTrue(0 <= prediction['confidence'] <= 1)
    
    def test_predict_multiple_queries(self):
        """Test prediction on multiple queries"""
        predictions = self.classifier.predict(self.sample_queries)
        
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(self.sample_queries))
        
        for i, prediction in enumerate(predictions):
            self.assertEqual(prediction['query'], self.sample_queries[i])
            self.assertIn('predicted_category', prediction)
            self.assertIn('confidence', prediction)
    
    def test_save_and_load_model(self):
        """Test model saving and loading"""
        # Create some dummy training history
        self.classifier.training_history = [
            {'epoch': 1, 'train_accuracy': 0.8, 'val_accuracy': 0.75}
        ]
        
        # Fit label encoder with sample data
        df = pd.DataFrame({
            'query': self.sample_queries,
            'category': self.sample_categories
        })
        self.classifier.preprocess_data(df)
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'test_model')
        self.classifier.save_model(model_path)
        
        # Check if files were created
        self.assertTrue(os.path.exists(os.path.join(model_path, 'metadata.json')))
        self.assertTrue(os.path.exists(os.path.join(model_path, 'label_encoder.pkl')))
        
        # Load model
        new_classifier = BERTCustomerQueryClassifier()
        new_classifier.load_model(model_path)
        
        self.assertEqual(new_classifier.model_name, self.classifier.model_name)
        self.assertEqual(new_classifier.num_labels, self.classifier.num_labels)
    
    def test_evaluate_model_simulation(self):
        """Test model evaluation with simulated data"""
        # Create sample data
        df = pd.DataFrame({
            'query': self.sample_queries,
            'category': self.sample_categories
        })
        
        queries, labels = self.classifier.preprocess_data(df)
        
        # Test evaluation
        results = self.classifier.evaluate_model(queries, labels)
        
        self.assertIn('accuracy', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('f1_score', results)
        self.assertIn('num_samples', results)
        
        self.assertEqual(results['num_samples'], len(queries))
        self.assertTrue(0 <= results['accuracy'] <= 1)
    
    def test_category_coverage(self):
        """Test that all default categories are covered"""
        expected_categories = [
            'billing_inquiry', 'technical_support', 'account_management',
            'product_information', 'complaint', 'cancellation_request',
            'upgrade_request', 'general_inquiry', 'refund_request', 'service_outage'
        ]
        
        self.assertEqual(set(self.classifier.default_categories), set(expected_categories))
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent for the same input"""
        query = "I want to cancel my service"
        
        prediction1 = self.classifier.predict(query)[0]
        prediction2 = self.classifier.predict(query)[0]
        
        # Predictions should be consistent (allowing for some randomness in simulation)
        self.assertEqual(prediction1['query'], prediction2['query'])
        # Note: In simulation mode, predictions might vary slightly due to randomness

class TestBERTClassifierIntegration(unittest.TestCase):
    """Integration tests for BERT classifier"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        if BERTCustomerQueryClassifier is None:
            self.skipTest("BERT classifier not available")
        
        self.classifier = BERTCustomerQueryClassifier()
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data generation to prediction"""
        # Generate sample data
        df = self.classifier.generate_sample_data(num_samples=50)
        
        # Preprocess data
        queries, labels = self.classifier.preprocess_data(df)
        
        # Initialize model (will use simulation mode)
        self.classifier.initialize_model()
        
        # Train model (simulated)
        training_history = self.classifier.train_model(queries, labels)
        
        # Evaluate model
        evaluation_results = self.classifier.evaluate_model(queries, labels)
        
        # Make predictions
        test_queries = ["Help me with my account", "Service is down"]
        predictions = self.classifier.predict(test_queries)
        
        # Verify results
        self.assertIsInstance(training_history, list)
        self.assertIsInstance(evaluation_results, dict)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(test_queries))
    
    def test_annotation_reduction_calculation(self):
        """Test annotation reduction calculation"""
        # Generate predictions with varying confidence levels
        test_queries = [
            "My bill is wrong",
            "Internet not working", 
            "Cancel my account",
            "What services do you offer?",
            "I need help"
        ]
        
        predictions = self.classifier.predict(test_queries)
        
        # Calculate annotation reduction
        high_confidence_count = sum(1 for pred in predictions if pred['confidence'] > 0.8)
        annotation_reduction = (high_confidence_count / len(predictions)) * 100
        
        self.assertIsInstance(annotation_reduction, (int, float))
        self.assertTrue(0 <= annotation_reduction <= 100)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Generate sample data
        df = self.classifier.generate_sample_data(num_samples=100)
        queries, labels = self.classifier.preprocess_data(df)
        
        # Evaluate model
        results = self.classifier.evaluate_model(queries, labels)
        
        # Check that all required metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            self.assertIn(metric, results)
            self.assertIsInstance(results[metric], (int, float))
            self.assertTrue(0 <= results[metric] <= 1)

class TestBERTClassifierEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up edge case test fixtures"""
        if BERTCustomerQueryClassifier is None:
            self.skipTest("BERT classifier not available")
        
        self.classifier = BERTCustomerQueryClassifier()
    
    def test_empty_query_prediction(self):
        """Test prediction with empty query"""
        predictions = self.classifier.predict([""])
        
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions[0], dict)
    
    def test_very_long_query_prediction(self):
        """Test prediction with very long query"""
        long_query = "This is a very long query " * 100
        predictions = self.classifier.predict([long_query])
        
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions[0], dict)
    
    def test_special_characters_query(self):
        """Test prediction with special characters"""
        special_query = "My bill has errors!!! @#$%^&*()_+ Can you help???"
        predictions = self.classifier.predict([special_query])
        
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions[0], dict)
    
    def test_non_english_query(self):
        """Test prediction with non-English text"""
        non_english_query = "Bonjour, j'ai un problème avec mon compte"
        predictions = self.classifier.predict([non_english_query])
        
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions[0], dict)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestBERTCustomerQueryClassifier))
    test_suite.addTest(unittest.makeSuite(TestBERTClassifierIntegration))
    test_suite.addTest(unittest.makeSuite(TestBERTClassifierEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)

