
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib
import json
import boto3

class AdvancedMLProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.clusterer = DBSCAN(eps=0.3, min_samples=10)
        
    def preprocess_data(self, data):
        """Advanced data preprocessing with feature engineering"""
        df = pd.DataFrame(data)
        
        # Feature engineering
        for col in df.select_dtypes(include=[np.number]):
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_log'] = np.log1p(df[col])
            
        # Outlier detection and treatment
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        
        return df[(df >= (Q1 - 1.5 * IQR)) & (df <= (Q3 + 1.5 * IQR))].all(axis=1)
    
    def advanced_classification(self, X, y):
        """Complex classification with ensemble methods"""
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        
        # Feature importance analysis
        feature_importance = self.classifier.feature_importances_
        
        return {
            'model_accuracy': self.classifier.score(X_scaled, y),
            'feature_importance': feature_importance.tolist(),
            'predictions': self.classifier.predict(X_scaled).tolist()
        }
    
    def time_series_forecasting(self, data, periods=30):
        """Advanced time series forecasting with multiple models"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.arima.model import ARIMA
        
        ts = pd.Series(data)
        
        # Seasonal decomposition
        decomposition = seasonal_decompose(ts, model='additive', period=12)
        
        # ARIMA modeling
        model = ARIMA(ts, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Forecasting
        forecast = fitted_model.forecast(steps=periods)
        
        return {
            'trend': decomposition.trend.dropna().tolist(),
            'seasonal': decomposition.seasonal.dropna().tolist(), 
            'forecast': forecast.tolist(),
            'confidence_intervals': fitted_model.get_forecast(periods).conf_int().values.tolist()
        }
    
    def anomaly_detection(self, data):
        """Advanced anomaly detection using multiple algorithms"""
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        
        X = np.array(data).reshape(-1, 1)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_anomalies = iso_forest.fit_predict(X)
        
        # One-Class SVM
        svm = OneClassSVM(gamma='scale', nu=0.1)
        svm_anomalies = svm.fit_predict(X)
        
        # DBSCAN clustering for anomalies
        clusters = self.clusterer.fit_predict(X)
        
        return {
            'isolation_forest_anomalies': (iso_anomalies == -1).tolist(),
            'svm_anomalies': (svm_anomalies == -1).tolist(),
            'cluster_anomalies': (clusters == -1).tolist(),
            'anomaly_scores': iso_forest.score_samples(X).tolist()
        }

def lambda_handler(event, context):
    processor = AdvancedMLProcessor()
    
    operation = event.get('operation')
    data = event.get('data')
    
    if operation == 'classification':
        X = event.get('features')
        y = event.get('labels')
        result = processor.advanced_classification(X, y)
    elif operation == 'forecasting':
        result = processor.time_series_forecasting(data, event.get('periods', 30))
    elif operation == 'anomaly_detection':
        result = processor.anomaly_detection(data)
    else:
        result = {'error': 'Unknown operation'}
    
    return {
        'statusCode': 200,
        'body': json.dumps(result),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }
        