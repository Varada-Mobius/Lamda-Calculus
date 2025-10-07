"""
REST API for Concurrent λ-Calculus with π-Calculus Channels
===========================================================

A Flask-based REST API for processing concurrent λ+π systems.

Endpoints:
    POST /api/v1/execute - Execute concurrent system
    POST /api/v1/analyze - Analyze system without execution
    GET  /api/v1/patterns - List available patterns
    POST /api/v1/patterns/{pattern_name} - Create system from pattern
    GET  /api/v1/health - Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import traceback
from typing import Dict, Any
import time

# Import the main processing function
from main import (
    process_concurrent_system,
    parse_expression_dict,
    create_simple_communication,
    create_producer_consumer,
    create_request_response,
    ConcurrentEngine,
    Expression
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'concurrent-lambda-pi',
        'version': '1.0.0',
        'timestamp': time.time()
    }), 200


@app.route('/api/v1/execute', methods=['POST'])
def execute_system():
    """
    Execute a concurrent λ+π system.
    
    Request body:
    {
        "expression": {...},     // Expression specification (required)
        "max_steps": 1000,       // Maximum reduction steps (optional)
        "timeout": 10.0          // Timeout in seconds (optional)
    }
    
    Response:
    {
        "success": true,
        "result": {
            "initial_expression": "...",
            "final_expression": "...",
            "steps_taken": 42,
            "communications": [...],
            "terminated": true,
            "deadlocked": false,
            "execution_time": 0.123,
            "analysis": {...}
        }
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'expression' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: expression'
            }), 400
        
        # Process the system
        result = process_concurrent_system(data)
        
        return jsonify({
            'success': True,
            'result': result
        }), 200
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Validation error: {str(e)}'
        }), 400
    
    except Exception as e:
        app.logger.error(f"Execution error: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Execution failed: {str(e)}'
        }), 500


@app.route('/api/v1/analyze', methods=['POST'])
def analyze_system():
    """
    Analyze a concurrent system without executing it.
    
    Request body:
    {
        "expression": {...}  // Expression specification
    }
    
    Response:
    {
        "success": true,
        "analysis": {
            "free_variables": [...],
            "free_channels": [...],
            "send_operations": 2,
            "receive_operations": 2,
            "balanced_communication": true,
            "expression_size": 15
        }
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        
        if 'expression' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: expression'
            }), 400
        
        # Parse expression
        expr = parse_expression_dict(data['expression'])
        
        # Analyze without execution
        engine = ConcurrentEngine()
        analysis = engine._analyze_expression(expr)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'expression': str(expr)
        }), 200
        
    except Exception as e:
        app.logger.error(f"Analysis error: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500


@app.route('/api/v1/patterns', methods=['GET'])
def list_patterns():
    """
    List available concurrent patterns.
    
    Response:
    {
        "success": true,
        "patterns": [
            {
                "name": "simple_communication",
                "description": "Basic send/receive pattern",
                "parameters": ["channel_name", "value_name"]
            },
            ...
        ]
    }
    """
    patterns = [
        {
            'name': 'simple_communication',
            'description': 'Basic send/receive pattern: send(c, v) | recv(c)',
            'parameters': [
                {'name': 'channel_name', 'type': 'string', 'required': True},
                {'name': 'value_name', 'type': 'string', 'required': True}
            ]
        },
        {
            'name': 'producer_consumer',
            'description': 'Producer-consumer pattern with work queue',
            'parameters': [
                {'name': 'channel_name', 'type': 'string', 'required': True}
            ]
        },
        {
            'name': 'request_response',
            'description': 'Client-server request-response pattern',
            'parameters': [
                {'name': 'request_channel', 'type': 'string', 'required': True},
                {'name': 'response_channel', 'type': 'string', 'required': True}
            ]
        }
    ]
    
    return jsonify({
        'success': True,
        'patterns': patterns
    }), 200


@app.route('/api/v1/patterns/<pattern_name>', methods=['POST'])
def create_pattern(pattern_name: str):
    """
    Create a system from a predefined pattern.
    
    Request body:
    {
        "parameters": {
            "channel_name": "work",
            "value_name": "task"
        },
        "execute": true,        // Execute immediately (optional)
        "max_steps": 1000       // If execute=true (optional)
    }
    
    Response:
    {
        "success": true,
        "expression": "...",
        "result": {...}  // If execute=true
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        params = data.get('parameters', {})
        
        # Create expression based on pattern
        if pattern_name == 'simple_communication':
            channel = params.get('channel_name', 'c')
            value = params.get('value_name', 'v')
            expr = create_simple_communication(channel, value)
        
        elif pattern_name == 'producer_consumer':
            channel = params.get('channel_name', 'work')
            expr = create_producer_consumer(channel)
        
        elif pattern_name == 'request_response':
            req_ch = params.get('request_channel', 'request')
            resp_ch = params.get('response_channel', 'response')
            expr = create_request_response(req_ch, resp_ch)
        
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown pattern: {pattern_name}'
            }), 404
        
        response_data = {
            'success': True,
            'pattern': pattern_name,
            'expression': str(expr)
        }
        
        # Execute if requested
        if data.get('execute', False):
            result = process_concurrent_system({
                'expression': expr,
                'max_steps': data.get('max_steps', 1000),
                'timeout': data.get('timeout', 10.0)
            })
            response_data['result'] = result
        
        return jsonify(response_data), 200
        
    except Exception as e:
        app.logger.error(f"Pattern creation error: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Pattern creation failed: {str(e)}'
        }), 500


@app.route('/api/v1/validate', methods=['POST'])
def validate_expression():
    """
    Validate an expression without executing it.
    
    Request body:
    {
        "expression": {...}
    }
    
    Response:
    {
        "success": true,
        "valid": true,
        "expression": "...",
        "warnings": []
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        
        if 'expression' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: expression'
            }), 400
        
        # Try to parse
        expr = parse_expression_dict(data['expression'])
        
        # Check for warnings
        warnings = []
        
        free_vars = expr.free_vars()
        if free_vars:
            warnings.append(f"Expression has free variables: {list(free_vars)}")
        
        free_channels = expr.free_channels()
        if free_channels:
            warnings.append(f"Expression has free channels: {list(free_channels)}")
        
        # Check communication balance
        engine = ConcurrentEngine()
        sends, receives = engine._count_communications(expr)
        if sends != receives:
            warnings.append(f"Unbalanced communication: {sends} sends, {receives} receives")
        
        return jsonify({
            'success': True,
            'valid': True,
            'expression': str(expr),
            'warnings': warnings
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'valid': False,
            'error': str(e)
        }), 400


@app.route('/api/v1/batch', methods=['POST'])
def batch_execute():
    """
    Execute multiple systems in batch.
    
    Request body:
    {
        "systems": [
            {
                "id": "system1",
                "expression": {...},
                "max_steps": 1000
            },
            ...
        ]
    }
    
    Response:
    {
        "success": true,
        "results": [
            {
                "id": "system1",
                "success": true,
                "result": {...}
            },
            ...
        ]
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        systems = data.get('systems', [])
        
        if not systems:
            return jsonify({
                'success': False,
                'error': 'No systems provided'
            }), 400
        
        results = []
        
        for system_spec in systems:
            system_id = system_spec.get('id', 'unknown')
            
            try:
                result = process_concurrent_system(system_spec)
                results.append({
                    'id': system_id,
                    'success': True,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'id': system_id,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(systems),
            'successful': sum(1 for r in results if r['success']),
            'results': results
        }), 200
        
    except Exception as e:
        app.logger.error(f"Batch execution error: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Batch execution failed: {str(e)}'
        }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
