
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os
import redis
from celery import Celery
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'complex-algorithms-secret')

# Redis configuration for caching and task queue
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(redis_url)

# Celery configuration for background tasks
celery = Celery(app.name, broker=redis_url)
celery.conf.update(app.config)

class AdvancedMicroservice:
    def __init__(self):
        self.cache_timeout = 3600  # 1 hour
        
    def get_cached_result(self, key):
        """Get cached computation result"""
        try:
            cached = redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        return None
    
    def cache_result(self, key, result):
        """Cache computation result"""
        try:
            redis_client.setex(key, self.cache_timeout, json.dumps(result, default=str))
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    def complex_matrix_operations(self, matrices, operation='multiply'):
        """Perform complex matrix operations"""
        try:
            matrix_arrays = [np.array(matrix) for matrix in matrices]
            
            if operation == 'multiply':
                result = matrix_arrays[0]
                for matrix in matrix_arrays[1:]:
                    result = np.dot(result, matrix)
                    
            elif operation == 'eigenvalues':
                result = []
                for matrix in matrix_arrays:
                    eigenvals, eigenvects = np.linalg.eig(matrix)
                    result.append({
                        'eigenvalues': eigenvals.real.tolist(),
                        'eigenvectors': eigenvects.real.tolist()
                    })
                    
            elif operation == 'svd':
                result = []
                for matrix in matrix_arrays:
                    U, s, Vt = np.linalg.svd(matrix)
                    result.append({
                        'U': U.tolist(),
                        'singular_values': s.tolist(),
                        'Vt': Vt.tolist()
                    })
                    
            elif operation == 'inverse':
                result = []
                for matrix in matrix_arrays:
                    try:
                        inv_matrix = np.linalg.inv(matrix)
                        result.append({
                            'inverse': inv_matrix.tolist(),
                            'determinant': float(np.linalg.det(matrix))
                        })
                    except np.linalg.LinAlgError:
                        result.append({'error': 'Matrix is singular and cannot be inverted'})
                        
            else:
                return {'error': f'Unsupported operation: {operation}'}
            
            return {
                'operation': operation,
                'result': result,
                'computation_time': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Matrix operation failed: {str(e)}'}
    
    def advanced_signal_processing(self, signal_data, operation='fft'):
        """Advanced signal processing operations"""
        try:
            signal = np.array(signal_data)
            
            if operation == 'fft':
                # Fast Fourier Transform
                fft_result = np.fft.fft(signal)
                frequencies = np.fft.fftfreq(len(signal))
                
                result = {
                    'fft_real': fft_result.real.tolist(),
                    'fft_imag': fft_result.imag.tolist(),
                    'magnitude': np.abs(fft_result).tolist(),
                    'phase': np.angle(fft_result).tolist(),
                    'frequencies': frequencies.tolist()
                }
                
            elif operation == 'filter':
                # Apply various filters
                from scipy import signal as scipy_signal
                
                # Low-pass filter
                b, a = scipy_signal.butter(4, 0.3, 'low')
                filtered_low = scipy_signal.filtfilt(b, a, signal)
                
                # High-pass filter
                b, a = scipy_signal.butter(4, 0.1, 'high')
                filtered_high = scipy_signal.filtfilt(b, a, signal)
                
                result = {
                    'original': signal.tolist(),
                    'low_pass': filtered_low.tolist(),
                    'high_pass': filtered_high.tolist()
                }
                
            elif operation == 'spectral_analysis':
                from scipy import signal as scipy_signal
                
                # Spectral analysis
                freqs, psd = scipy_signal.periodogram(signal)
                
                # Find dominant frequencies
                dominant_freq_idx = np.argsort(psd)[-5:]  # Top 5 frequencies
                dominant_freqs = freqs[dominant_freq_idx]
                dominant_powers = psd[dominant_freq_idx]
                
                result = {
                    'frequencies': freqs.tolist(),
                    'power_spectral_density': psd.tolist(),
                    'dominant_frequencies': dominant_freqs.tolist(),
                    'dominant_powers': dominant_powers.tolist()
                }
                
            else:
                return {'error': f'Unsupported signal processing operation: {operation}'}
            
            return {
                'operation': operation,
                'result': result,
                'signal_length': len(signal),
                'processing_time': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Signal processing failed: {str(e)}'}
    
    def optimization_algorithms(self, objective_function, constraints, bounds):
        """Advanced optimization algorithms"""
        try:
            from scipy.optimize import minimize, differential_evolution, basinhopping
            
            # Define objective function based on string input
            if objective_function == 'quadratic':
                def obj_func(x):
                    return np.sum(x**2) + np.sum(x)
            elif objective_function == 'rosenbrock':
                def obj_func(x):
                    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
            elif objective_function == 'sphere':
                def obj_func(x):
                    return np.sum(x**2)
            else:
                def obj_func(x):
                    return np.sum(x**2)  # Default to sphere function
            
            # Initial guess
            x0 = np.random.uniform(-1, 1, len(bounds))
            
            results = {}
            
            # Method 1: SLSQP
            try:
                slsqp_result = minimize(obj_func, x0, method='SLSQP', bounds=bounds)
                results['slsqp'] = {
                    'success': bool(slsqp_result.success),
                    'optimal_value': float(slsqp_result.fun),
                    'optimal_solution': slsqp_result.x.tolist(),
                    'iterations': int(slsqp_result.nit) if hasattr(slsqp_result, 'nit') else 0
                }
            except Exception as e:
                results['slsqp'] = {'error': str(e)}
            
            # Method 2: Differential Evolution
            try:
                de_result = differential_evolution(obj_func, bounds, maxiter=100)
                results['differential_evolution'] = {
                    'success': bool(de_result.success),
                    'optimal_value': float(de_result.fun),
                    'optimal_solution': de_result.x.tolist(),
                    'iterations': int(de_result.nit)
                }
            except Exception as e:
                results['differential_evolution'] = {'error': str(e)}
            
            # Method 3: Basin Hopping
            try:
                bh_result = basinhopping(obj_func, x0, niter=50)
                results['basin_hopping'] = {
                    'success': True,
                    'optimal_value': float(bh_result.fun),
                    'optimal_solution': bh_result.x.tolist(),
                    'iterations': int(bh_result.nit)
                }
            except Exception as e:
                results['basin_hopping'] = {'error': str(e)}
            
            # Find best result
            best_method = None
            best_value = float('inf')
            
            for method, result in results.items():
                if 'optimal_value' in result and result['optimal_value'] < best_value:
                    best_value = result['optimal_value']
                    best_method = method
            
            return {
                'objective_function': objective_function,
                'all_methods': results,
                'best_method': best_method,
                'best_value': best_value if best_value != float('inf') else None,
                'optimization_time': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Optimization failed: {str(e)}'}

# Initialize microservice
microservice = AdvancedMicroservice()

@app.route('/')
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Bubble Advanced Algorithms Microservice',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/matrix', methods=['POST'])
def matrix_operations():
    """Handle complex matrix operations"""
    try:
        data = request.get_json()
        
        # Create cache key
        cache_key = f"matrix:{hash(str(data))}"
        cached_result = microservice.get_cached_result(cache_key)
        
        if cached_result:
            cached_result['from_cache'] = True
            return jsonify(cached_result)
        
        matrices = data.get('matrices', [])
        operation = data.get('operation', 'multiply')
        
        if not matrices:
            return jsonify({'error': 'No matrices provided'}), 400
        
        result = microservice.complex_matrix_operations(matrices, operation)
        
        # Cache result
        microservice.cache_result(cache_key, result)
        result['from_cache'] = False
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Matrix operations error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/signal', methods=['POST'])
def signal_processing():
    """Handle advanced signal processing"""
    try:
        data = request.get_json()
        
        # Create cache key
        cache_key = f"signal:{hash(str(data))}"
        cached_result = microservice.get_cached_result(cache_key)
        
        if cached_result:
            cached_result['from_cache'] = True
            return jsonify(cached_result)
        
        signal_data = data.get('signal', [])
        operation = data.get('operation', 'fft')
        
        if not signal_data:
            return jsonify({'error': 'No signal data provided'}), 400
        
        result = microservice.advanced_signal_processing(signal_data, operation)
        
        # Cache result
        microservice.cache_result(cache_key, result)
        result['from_cache'] = False
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Signal processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize', methods=['POST'])
def optimization():
    """Handle optimization problems"""
    try:
        data = request.get_json()
        
        # Create cache key
        cache_key = f"optimize:{hash(str(data))}"
        cached_result = microservice.get_cached_result(cache_key)
        
        if cached_result:
            cached_result['from_cache'] = True
            return jsonify(cached_result)
        
        objective_function = data.get('objective_function', 'quadratic')
        constraints = data.get('constraints', [])
        bounds = data.get('bounds', [(-10, 10)] * 2)  # Default 2D problem
        
        result = microservice.optimization_algorithms(objective_function, constraints, bounds)
        
        # Cache result
        microservice.cache_result(cache_key, result)
        result['from_cache'] = False
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        return jsonify({'error': str(e)}), 500

@celery.task
def complex_background_computation(data):
    """Background task for complex computations"""
    try:
        # Simulate complex computation
        result = {
            'computation_id': data.get('id'),
            'result': np.random.randn(1000).tolist(),  # Simulate complex result
            'status': 'completed',
            'completed_at': datetime.utcnow().isoformat()
        }
        
        # Store result in Redis
        redis_client.setex(f"computation:{data.get('id')}", 3600, json.dumps(result, default=str))
        
        return result
    except Exception as e:
        error_result = {
            'computation_id': data.get('id'),
            'error': str(e),
            'status': 'failed',
            'failed_at': datetime.utcnow().isoformat()
        }
        
        redis_client.setex(f"computation:{data.get('id')}", 3600, json.dumps(error_result, default=str))
        return error_result

@app.route('/api/compute/async', methods=['POST'])
def start_async_computation():
    """Start asynchronous computation"""
    try:
        data = request.get_json()
        computation_id = f"comp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        data['id'] = computation_id
        
        # Start background task
        task = complex_background_computation.delay(data)
        
        return jsonify({
            'computation_id': computation_id,
            'task_id': task.id,
            'status': 'started',
            'message': 'Computation started in background'
        })
        
    except Exception as e:
        logger.error(f"Async computation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compute/status/<computation_id>', methods=['GET'])
def get_computation_status(computation_id):
    """Get status of asynchronous computation"""
    try:
        result = redis_client.get(f"computation:{computation_id}")
        
        if result:
            return jsonify(json.loads(result))
        else:
            return jsonify({
                'computation_id': computation_id,
                'status': 'not_found',
                'message': 'Computation not found or expired'
            }), 404
            
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def detailed_health():
    """Detailed health check with system status"""
    try:
        # Check Redis connection
        redis_status = 'healthy'
        try:
            redis_client.ping()
        except:
            redis_status = 'unhealthy'
        
        # Check memory usage
        import psutil
        memory_usage = psutil.virtual_memory().percent
        
        return jsonify({
            'status': 'healthy',
            'components': {
                'redis': redis_status,
                'memory_usage_percent': memory_usage
            },
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_ENV') == 'development')
        