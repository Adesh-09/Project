from flask import Blueprint, jsonify

# Blueprint for user-related endpoints
user_bp = Blueprint('user', __name__)

@user_bp.route('/users', methods=['GET'])
def list_users():
    """Return a simple list of users."""
    return jsonify([])
