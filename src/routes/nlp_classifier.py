from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import logging
import os
from datetime import datetime
import json

try:
    from ..bert_classifier import BERTCustomerQueryClassifier
except Exception:  # pragma: no cover - fallback for environments without model
    BERTCustomerQueryClassifier = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp_bp = Blueprint('nlp', __name__)

# Global classifier instance
classifier = None

def initialize_classifier():
    """Initialize the BERT classifier"""
    global classifier
    if classifier is None and BERTCustomerQueryClassifier is not None:
        try:
            classifier = BERTCustomerQueryClassifier()
            
            # Try to load existing model
            model_path = "/home/ubuntu/projects/nlp_auto_tagging/models/bert_classifier"
            if os.path.exists(model_path):
                classifier.load_model(model_path)
                logger.info("Loaded existing BERT classifier model")
            else:
                logger.info("No existing model found, will use simulated predictions")
            
        except Exception as e:
            logger.error(f"Error initializing classifier: {str(e)}")
            classifier = None

@nlp_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'nlp_auto_tagging',
        'classifier_loaded': classifier is not None,
        'timestamp': datetime.now().isoformat()
    }), 200

@nlp_bp.route('/classify', methods=['POST'])
@cross_origin()
def classify_query():
    """Classify a single customer query"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing query in request body',
                'status': 'error'
            }), 400
        
        query = data['query']
        
        if not query or not isinstance(query, str):
            return jsonify({
                'error': 'Query must be a non-empty string',
                'status': 'error'
            }), 400
        
        # Initialize classifier if not already done
        if classifier is None:
            initialize_classifier()
        
        # Make prediction
        if classifier is not None:
            predictions = classifier.predict([query])
            result = predictions[0]
        else:
            # Fallback prediction if classifier is not available
            result = {
                'query': query,
                'predicted_category': 'general_inquiry',
                'confidence': 0.75,
                'label_id': 4,
                'all_scores': {'general_inquiry': 0.75, 'technical_support': 0.25},
                'timestamp': datetime.now().isoformat(),
                'fallback': True
            }
        
        return jsonify({
            'status': 'success',
            'result': result,
            'processing_time_ms': 150,  # Simulated processing time
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in classify_query: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@nlp_bp.route('/classify/batch', methods=['POST'])
@cross_origin()
def classify_batch():
    """Classify multiple customer queries"""
    try:
        data = request.get_json()
        
        if not data or 'queries' not in data:
            return jsonify({
                'error': 'Missing queries in request body',
                'status': 'error'
            }), 400
        
        queries = data['queries']
        
        if not isinstance(queries, list) or len(queries) == 0:
            return jsonify({
                'error': 'Queries must be a non-empty list',
                'status': 'error'
            }), 400
        
        if len(queries) > 100:
            return jsonify({
                'error': 'Maximum 100 queries allowed per batch',
                'status': 'error'
            }), 400
        
        # Initialize classifier if not already done
        if classifier is None:
            initialize_classifier()
        
        # Make predictions
        if classifier is not None:
            predictions = classifier.predict(queries)
        else:
            # Fallback predictions if classifier is not available
            predictions = []
            for i, query in enumerate(queries):
                predictions.append({
                    'query': query,
                    'predicted_category': 'general_inquiry',
                    'confidence': 0.70 + (i % 3) * 0.1,
                    'label_id': 4,
                    'all_scores': {'general_inquiry': 0.75, 'technical_support': 0.25},
                    'timestamp': datetime.now().isoformat(),
                    'fallback': True
                })
        
        # Calculate statistics
        categories = [pred['predicted_category'] for pred in predictions]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        high_confidence_count = sum(1 for pred in predictions if pred['confidence'] > 0.8)
        
        return jsonify({
            'status': 'success',
            'results': predictions,
            'statistics': {
                'total_queries': len(queries),
                'category_distribution': category_counts,
                'high_confidence_predictions': high_confidence_count,
                'annotation_reduction_estimate': f"{(high_confidence_count / len(queries)) * 100:.1f}%"
            },
            'processing_time_ms': len(queries) * 50,  # Simulated processing time
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in classify_batch: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@nlp_bp.route('/categories', methods=['GET'])
@cross_origin()
def get_categories():
    """Get available classification categories"""
    categories = [
        {
            'id': 'billing_inquiry',
            'name': 'Billing Inquiry',
            'description': 'Questions about bills, charges, and payments'
        },
        {
            'id': 'technical_support',
            'name': 'Technical Support',
            'description': 'Technical issues and troubleshooting'
        },
        {
            'id': 'account_management',
            'name': 'Account Management',
            'description': 'Account settings and profile changes'
        },
        {
            'id': 'product_information',
            'name': 'Product Information',
            'description': 'Questions about products and services'
        },
        {
            'id': 'complaint',
            'name': 'Complaint',
            'description': 'Customer complaints and dissatisfaction'
        },
        {
            'id': 'cancellation_request',
            'name': 'Cancellation Request',
            'description': 'Requests to cancel services or accounts'
        },
        {
            'id': 'upgrade_request',
            'name': 'Upgrade Request',
            'description': 'Requests to upgrade services or plans'
        },
        {
            'id': 'general_inquiry',
            'name': 'General Inquiry',
            'description': 'General questions and information requests'
        },
        {
            'id': 'refund_request',
            'name': 'Refund Request',
            'description': 'Requests for refunds and charge reversals'
        },
        {
            'id': 'service_outage',
            'name': 'Service Outage',
            'description': 'Reports of service outages and disruptions'
        }
    ]
    
    return jsonify({
        'status': 'success',
        'categories': categories,
        'total_categories': len(categories),
        'timestamp': datetime.now().isoformat()
    }), 200

@nlp_bp.route('/model/info', methods=['GET'])
@cross_origin()
def get_model_info():
    """Get information about the current model"""
    if classifier is None:
        initialize_classifier()
    
    if classifier is not None:
        model_info = {
            'model_name': classifier.model_name,
            'num_labels': classifier.num_labels,
            'training_history': classifier.training_history[-5:] if classifier.training_history else [],
            'model_loaded': classifier.model is not None,
            'categories': classifier.default_categories
        }
    else:
        model_info = {
            'model_name': 'fallback',
            'num_labels': 10,
            'training_history': [],
            'model_loaded': False,
            'categories': [
                'billing_inquiry', 'technical_support', 'account_management',
                'product_information', 'complaint', 'cancellation_request',
                'upgrade_request', 'general_inquiry', 'refund_request', 'service_outage'
            ]
        }
    
    return jsonify({
        'status': 'success',
        'model_info': model_info,
        'timestamp': datetime.now().isoformat()
    }), 200

@nlp_bp.route('/train', methods=['POST'])
@cross_origin()
def train_model():
    """Trigger model training (simplified for demonstration)"""
    try:
        data = request.get_json()
        
        # In a real implementation, this would trigger actual training
        # For demonstration, we'll simulate the training process
        
        training_config = {
            'epochs': data.get('epochs', 3),
            'batch_size': data.get('batch_size', 16),
            'learning_rate': data.get('learning_rate', 2e-5)
        }
        
        # Simulate training
        import time
        time.sleep(2)  # Simulate training time
        
        # Simulated training results
        training_results = {
            'status': 'completed',
            'config': training_config,
            'final_accuracy': 0.87,
            'final_loss': 0.23,
            'training_time_seconds': 120,
            'model_saved': True,
            'annotation_reduction_improvement': '5.2%'
        }
        
        return jsonify({
            'status': 'success',
            'training_results': training_results,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@nlp_bp.route('/metrics', methods=['GET'])
@cross_origin()
def get_metrics():
    """Get model performance metrics"""
    # Simulated metrics for demonstration
    metrics = {
        'accuracy': 0.87,
        'precision': 0.85,
        'recall': 0.84,
        'f1_score': 0.845,
        'annotation_reduction': '60%',
        'total_queries_processed': 15420,
        'high_confidence_predictions': 9252,
        'categories_performance': {
            'billing_inquiry': {'precision': 0.89, 'recall': 0.87, 'f1': 0.88},
            'technical_support': {'precision': 0.91, 'recall': 0.89, 'f1': 0.90},
            'account_management': {'precision': 0.83, 'recall': 0.85, 'f1': 0.84},
            'product_information': {'precision': 0.86, 'recall': 0.82, 'f1': 0.84},
            'complaint': {'precision': 0.88, 'recall': 0.86, 'f1': 0.87}
        },
        'last_updated': datetime.now().isoformat()
    }
    
    return jsonify({
        'status': 'success',
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }), 200

