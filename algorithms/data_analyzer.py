
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import networkx as nx
from collections import defaultdict
import json

class ComplexDataAnalyzer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def advanced_statistical_analysis(self, data):
        """Comprehensive statistical analysis"""
        df = pd.DataFrame(data)
        
        results = {}
        
        for column in df.select_dtypes(include=[np.number]):
            col_data = df[column].dropna()
            
            # Advanced statistical measures
            results[column] = {
                'descriptive_stats': {
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'mode': float(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                    'std': float(col_data.std()),
                    'variance': float(col_data.var()),
                    'skewness': float(stats.skew(col_data)),
                    'kurtosis': float(stats.kurtosis(col_data)),
                    'range': float(col_data.max() - col_data.min())
                },
                'distribution_tests': {
                    'shapiro_wilk': {
                        'statistic': float(stats.shapiro(col_data)[0]),
                        'p_value': float(stats.shapiro(col_data)[1])
                    },
                    'jarque_bera': {
                        'statistic': float(stats.jarque_bera(col_data)[0]),
                        'p_value': float(stats.jarque_bera(col_data)[1])
                    }
                },
                'outliers': self.detect_outliers(col_data),
                'percentiles': {
                    'p25': float(np.percentile(col_data, 25)),
                    'p50': float(np.percentile(col_data, 50)),
                    'p75': float(np.percentile(col_data, 75)),
                    'p95': float(np.percentile(col_data, 95)),
                    'p99': float(np.percentile(col_data, 99))
                }
            }
        
        return results
    
    def detect_outliers(self, data):
        """Multiple outlier detection methods"""
        data = np.array(data)
        
        # IQR method
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        iqr_outliers = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
        
        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        z_outliers = z_scores > 3
        
        # Modified Z-score method
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        modified_z_outliers = np.abs(modified_z_scores) > 3.5
        
        return {
            'iqr_outliers': iqr_outliers.tolist(),
            'z_score_outliers': z_outliers.tolist(),
            'modified_z_outliers': modified_z_outliers.tolist(),
            'outlier_indices': np.where(iqr_outliers | z_outliers | modified_z_outliers)[0].tolist()
        }
    
    def correlation_network_analysis(self, data):
        """Advanced correlation and network analysis"""
        df = pd.DataFrame(data)
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create network graph from correlations
        G = nx.Graph()
        
        for i, col1 in enumerate(numeric_df.columns):
            for j, col2 in enumerate(numeric_df.columns):
                if i < j:
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Strong correlation threshold
                        G.add_edge(col1, col2, weight=abs(corr_val))
        
        # Network metrics
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        clustering = nx.clustering(G)
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'network_metrics': {
                'centrality': centrality,
                'betweenness_centrality': betweenness,
                'clustering_coefficient': clustering,
                'network_density': nx.density(G),
                'number_of_components': nx.number_connected_components(G)
            },
            'strongly_correlated_pairs': [
                (col1, col2, float(corr_matrix.loc[col1, col2]))
                for col1 in numeric_df.columns
                for col2 in numeric_df.columns
                if col1 < col2 and abs(corr_matrix.loc[col1, col2]) > 0.7
            ]
        }
    
    def dimensionality_reduction_analysis(self, data):
        """Advanced dimensionality reduction techniques"""
        df = pd.DataFrame(data)
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Standardize data
        scaled_data = self.scaler.fit_transform(numeric_df)
        
        # PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Determine optimal number of components
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumsum >= 0.95) + 1
        
        return {
            'pca_analysis': {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': cumsum.tolist(),
                'components_for_95_variance': int(n_components_95),
                'principal_components': pca_result[:, :min(5, pca_result.shape[1])].tolist(),
                'feature_loadings': pca.components_[:min(5, pca.components_.shape[0])].tolist()
            }
        }
    
    def time_series_decomposition(self, data, timestamps):
        """Advanced time series analysis"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.stattools import adfuller
        
        ts = pd.Series(data, index=pd.to_datetime(timestamps))
        ts = ts.sort_index()
        
        # Stationarity test
        adf_test = adfuller(ts.dropna())
        
        # Seasonal decomposition
        if len(ts) >= 24:  # Need sufficient data points
            decomposition = seasonal_decompose(ts, model='additive', period=min(12, len(ts)//2))
            
            return {
                'stationarity_test': {
                    'adf_statistic': float(adf_test[0]),
                    'p_value': float(adf_test[1]),
                    'is_stationary': adf_test[1] < 0.05
                },
                'decomposition': {
                    'trend': decomposition.trend.dropna().tolist(),
                    'seasonal': decomposition.seasonal.dropna().tolist(),
                    'residual': decomposition.resid.dropna().tolist()
                },
                'seasonality_strength': float(1 - np.var(decomposition.resid.dropna()) / np.var(ts.dropna()))
            }
        else:
            return {
                'error': 'Insufficient data for time series analysis',
                'required_minimum': 24,
                'provided': len(ts)
            }

def lambda_handler(event, context):
    analyzer = ComplexDataAnalyzer()
    
    operation = event.get('operation')
    data = event.get('data')
    
    try:
        if operation == 'statistical_analysis':
            result = analyzer.advanced_statistical_analysis(data)
        elif operation == 'correlation_network':
            result = analyzer.correlation_network_analysis(data)
        elif operation == 'dimensionality_reduction':
            result = analyzer.dimensionality_reduction_analysis(data)
        elif operation == 'time_series':
            timestamps = event.get('timestamps')
            result = analyzer.time_series_decomposition(data, timestamps)
        else:
            result = {'error': f'Unknown operation: {operation}'}
        
        return {
            'statusCode': 200,
            'body': json.dumps(result, default=str),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
        