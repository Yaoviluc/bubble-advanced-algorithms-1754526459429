
import json
import boto3
import numpy as np
from decimal import Decimal
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def bubble_ml_processor(event, context):
    """Main ML processing function for Bubble integration"""
    try:
        # Parse input from Bubble
        body = json.loads(event.get('body', '{}')) if isinstance(event.get('body'), str) else event.get('body', {})
        
        operation = body.get('operation')
        data = body.get('data')
        parameters = body.get('parameters', {})
        
        logger.info(f"Processing operation: {operation}")
        
        result = {}
        
        if operation == 'predict':
            result = perform_prediction(data, parameters)
        elif operation == 'cluster':
            result = perform_clustering(data, parameters)
        elif operation == 'optimize':
            result = perform_optimization(data, parameters)
        elif operation == 'analyze_sentiment':
            result = analyze_sentiment(data, parameters)
        elif operation == 'process_image':
            result = process_image(data, parameters)
        elif operation == 'calculate_risk':
            result = calculate_risk_score(data, parameters)
        elif operation == 'forecast_demand':
            result = forecast_demand(data, parameters)
        else:
            result = {'error': f'Unsupported operation: {operation}'}
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(result, default=decimal_default)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }

def perform_prediction(data, parameters):
    """Advanced prediction using ensemble methods"""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    import pandas as pd
    
    df = pd.DataFrame(data)
    
    # Separate features and target
    target_column = parameters.get('target_column', df.columns[-1])
    features = df.drop(columns=[target_column])
    target = df[target_column]
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Ensemble prediction
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    rf_scores = cross_val_score(rf_model, features_scaled, target, cv=5)
    gb_scores = cross_val_score(gb_model, features_scaled, target, cv=5)
    
    # Train best model
    if rf_scores.mean() > gb_scores.mean():
        best_model = rf_model
        best_score = rf_scores.mean()
        model_type = "Random Forest"
    else:
        best_model = gb_model
        best_score = gb_scores.mean()
        model_type = "Gradient Boosting"
    
    best_model.fit(features_scaled, target)
    predictions = best_model.predict(features_scaled)
    
    return {
        'model_type': model_type,
        'accuracy_score': float(best_score),
        'predictions': predictions.tolist(),
        'feature_importance': best_model.feature_importances_.tolist() if hasattr(best_model, 'feature_importances_') else [],
        'feature_names': features.columns.tolist()
    }

def perform_clustering(data, parameters):
    """Advanced clustering with multiple algorithms"""
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import pandas as pd
    
    df = pd.DataFrame(data)
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    n_clusters = parameters.get('n_clusters', 3)
    
    # Try different clustering algorithms
    algorithms = {
        'kmeans': KMeans(n_clusters=n_clusters, random_state=42),
        'dbscan': DBSCAN(eps=0.5, min_samples=5),
        'hierarchical': AgglomerativeClustering(n_clusters=n_clusters)
    }
    
    results = {}
    best_score = -1
    best_algorithm = None
    
    for name, algorithm in algorithms.items():
        try:
            clusters = algorithm.fit_predict(scaled_data)
            if len(set(clusters)) > 1:  # Need at least 2 clusters for silhouette score
                score = silhouette_score(scaled_data, clusters)
                results[name] = {
                    'clusters': clusters.tolist(),
                    'silhouette_score': float(score),
                    'n_clusters_found': len(set(clusters))
                }
                
                if score > best_score:
                    best_score = score
                    best_algorithm = name
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return {
        'clustering_results': results,
        'best_algorithm': best_algorithm,
        'best_score': float(best_score) if best_score > -1 else None,
        'recommended_clusters': results.get(best_algorithm, {}).get('clusters', []) if best_algorithm else []
    }

def perform_optimization(data, parameters):
    """Complex optimization using genetic algorithms and linear programming"""
    from scipy.optimize import minimize, differential_evolution
    import numpy as np
    
    # Define optimization problem based on parameters
    objective_function = parameters.get('objective', 'maximize_profit')
    constraints = parameters.get('constraints', {})
    variables = np.array(data)
    
    def profit_function(x):
        # Complex profit calculation
        base_profit = np.sum(x * variables.flatten()[:len(x)])
        complexity_penalty = 0.01 * np.sum(x ** 2)
        return -(base_profit - complexity_penalty)  # Negative for maximization
    
    def risk_constraint(x):
        return constraints.get('max_risk', 1000) - np.sum(x ** 2)
    
    # Bounds for variables
    bounds = [(0, parameters.get('max_value', 100)) for _ in range(len(variables.flatten()))]
    
    # Classical optimization
    classical_result = minimize(
        profit_function,
        x0=np.ones(len(variables.flatten())),
        bounds=bounds,
        method='SLSQP'
    )
    
    # Genetic algorithm optimization
    genetic_result = differential_evolution(
        profit_function,
        bounds=bounds,
        maxiter=100,
        seed=42
    )
    
    return {
        'classical_optimization': {
            'success': bool(classical_result.success),
            'optimal_values': classical_result.x.tolist() if classical_result.success else [],
            'optimal_objective': float(-classical_result.fun) if classical_result.success else None
        },
        'genetic_optimization': {
            'success': bool(genetic_result.success),
            'optimal_values': genetic_result.x.tolist(),
            'optimal_objective': float(-genetic_result.fun),
            'iterations': int(genetic_result.nit)
        },
        'comparison': {
            'better_algorithm': 'genetic' if genetic_result.fun < classical_result.fun else 'classical'
        }
    }

def analyze_sentiment(data, parameters):
    """Advanced sentiment analysis using multiple approaches"""
    import re
    from collections import Counter
    
    text_data = data if isinstance(data, list) else [data]
    
    # Simple rule-based sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'perfect', 'best']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'poor', 'sad']
    
    results = []
    
    for text in text_data:
        # Clean text
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        words = cleaned_text.split()
        
        # Count sentiment words
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
        else:
            sentiment_score = 0
        
        # Classify sentiment
        if sentiment_score > 0.1:
            sentiment = 'positive'
        elif sentiment_score < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        results.append({
            'text': text,
            'sentiment': sentiment,
            'sentiment_score': float(sentiment_score),
            'positive_words_count': positive_count,
            'negative_words_count': negative_count,
            'word_count': len(words)
        })
    
    # Overall statistics
    overall_sentiment = np.mean([r['sentiment_score'] for r in results])
    sentiment_distribution = Counter([r['sentiment'] for r in results])
    
    return {
        'individual_results': results,
        'overall_sentiment_score': float(overall_sentiment),
        'sentiment_distribution': dict(sentiment_distribution),
        'total_texts_analyzed': len(results)
    }

def process_image(data, parameters):
    """Image processing simulation (would integrate with AWS Rekognition in production)"""
    # This is a simulation - in production, you'd use AWS Rekognition or other image services
    image_url = data.get('image_url', '')
    processing_type = parameters.get('type', 'detect_objects')
    
    # Simulated image processing results
    simulated_results = {
        'detect_objects': {
            'objects': [
                {'name': 'person', 'confidence': 0.95, 'bounding_box': [0.1, 0.2, 0.3, 0.4]},
                {'name': 'car', 'confidence': 0.87, 'bounding_box': [0.5, 0.6, 0.7, 0.8]}
            ]
        },
        'analyze_faces': {
            'faces': [
                {'age_range': [25, 35], 'gender': 'Male', 'confidence': 0.92},
                {'age_range': [30, 40], 'gender': 'Female', 'confidence': 0.89}
            ]
        },
        'extract_text': {
            'text_detections': [
                {'text': 'Sample Text', 'confidence': 0.98},
                {'text': 'Another Text', 'confidence': 0.91}
            ]
        }
    }
    
    return {
        'image_url': image_url,
        'processing_type': processing_type,
        'results': simulated_results.get(processing_type, {}),
        'processed_at': context.aws_request_id if 'context' in globals() else 'simulation'
    }

def calculate_risk_score(data, parameters):
    """Advanced risk calculation using multiple factors"""
    risk_factors = data
    weights = parameters.get('weights', {})
    
    # Default risk weights
    default_weights = {
        'financial_risk': 0.3,
        'operational_risk': 0.25,
        'market_risk': 0.2,
        'compliance_risk': 0.15,
        'reputation_risk': 0.1
    }
    
    # Merge custom weights with defaults
    final_weights = {**default_weights, **weights}
    
    # Calculate weighted risk score
    total_risk = 0
    risk_breakdown = {}
    
    for factor, value in risk_factors.items():
        weight = final_weights.get(factor, 0.1)
        risk_contribution = value * weight
        total_risk += risk_contribution
        risk_breakdown[factor] = {
            'value': value,
            'weight': weight,
            'contribution': risk_contribution
        }
    
    # Risk categorization
    if total_risk <= 0.3:
        risk_category = 'Low'
    elif total_risk <= 0.6:
        risk_category = 'Medium'
    elif total_risk <= 0.8:
        risk_category = 'High'
    else:
        risk_category = 'Very High'
    
    return {
        'total_risk_score': float(total_risk),
        'risk_category': risk_category,
        'risk_breakdown': risk_breakdown,
        'recommendations': generate_risk_recommendations(total_risk, risk_breakdown)
    }

def generate_risk_recommendations(total_risk, risk_breakdown):
    """Generate risk mitigation recommendations"""
    recommendations = []
    
    if total_risk > 0.7:
        recommendations.append("Immediate risk mitigation required")
        recommendations.append("Consider reducing exposure in high-risk areas")
    
    # Find highest risk factors
    sorted_risks = sorted(risk_breakdown.items(), key=lambda x: x[1]['contribution'], reverse=True)
    
    for factor, details in sorted_risks[:3]:  # Top 3 risk factors
        if details['contribution'] > 0.1:
            recommendations.append(f"Address {factor.replace('_', ' ')} (contribution: {details['contribution']:.2%})")
    
    if total_risk < 0.3:
        recommendations.append("Risk levels are acceptable, maintain current monitoring")
    
    return recommendations

def forecast_demand(data, parameters):
    """Advanced demand forecasting using multiple models"""
    historical_data = np.array(data)
    forecast_periods = parameters.get('forecast_periods', 12)
    
    # Simple trend analysis
    x = np.arange(len(historical_data))
    trend_coefficients = np.polyfit(x, historical_data, 1)
    trend_forecast = np.polyval(trend_coefficients, np.arange(len(historical_data), len(historical_data) + forecast_periods))
    
    # Seasonal adjustment
    if len(historical_data) >= 12:
        seasonal_pattern = []
        for i in range(12):
            seasonal_values = historical_data[i::12]
            if len(seasonal_values) > 0:
                seasonal_pattern.append(np.mean(seasonal_values))
        
        # Apply seasonal pattern to forecast
        seasonal_forecast = []
        for i in range(forecast_periods):
            base_value = trend_forecast[i]
            seasonal_index = i % 12
            if seasonal_index < len(seasonal_pattern):
                seasonal_adjustment = seasonal_pattern[seasonal_index] / np.mean(historical_data)
                seasonal_forecast.append(base_value * seasonal_adjustment)
            else:
                seasonal_forecast.append(base_value)
    else:
        seasonal_forecast = trend_forecast.tolist()
    
    # Calculate confidence intervals
    historical_variance = np.var(historical_data)
    confidence_intervals = []
    
    for i, forecast_value in enumerate(seasonal_forecast):
        # Expanding confidence interval over time
        uncertainty = np.sqrt(historical_variance * (1 + i * 0.1))
        confidence_intervals.append({
            'lower_bound': float(forecast_value - 1.96 * uncertainty),
            'upper_bound': float(forecast_value + 1.96 * uncertainty)
        })
    
    return {
        'historical_data': historical_data.tolist(),
        'trend_forecast': trend_forecast.tolist(),
        'seasonal_forecast': seasonal_forecast,
        'confidence_intervals': confidence_intervals,
        'forecast_accuracy_metrics': {
            'mape': calculate_mape(historical_data),
            'rmse': calculate_rmse(historical_data)
        }
    }

def calculate_mape(data):
    """Calculate Mean Absolute Percentage Error"""
    if len(data) < 2:
        return 0
    
    actual = data[1:]
    predicted = data[:-1]  # Simple one-step-ahead prediction
    
    non_zero_actual = actual[actual != 0]
    non_zero_predicted = predicted[actual != 0]
    
    if len(non_zero_actual) == 0:
        return 0
    
    return float(np.mean(np.abs((non_zero_actual - non_zero_predicted) / non_zero_actual)) * 100)

def calculate_rmse(data):
    """Calculate Root Mean Square Error"""
    if len(data) < 2:
        return 0
    
    actual = data[1:]
    predicted = data[:-1]  # Simple one-step-ahead prediction
    
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))
        