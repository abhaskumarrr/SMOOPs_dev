"""
Performance Monitoring System

This module provides tools for tracking and monitoring model performance over time.
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_LOGS_DIR = os.environ.get("PERFORMANCE_LOGS_DIR", "logs/performance")


class PerformanceTracker:
    """
    Track and monitor model performance metrics over time.
    """
    
    def __init__(self, symbol: str, logs_dir: Optional[str] = None):
        """
        Initialize the performance tracker.
        
        Args:
            symbol: Trading symbol or model identifier
            logs_dir: Directory for performance logs
        """
        self.symbol = symbol
        self.logs_dir = logs_dir or DEFAULT_LOGS_DIR
        
        # Normalize symbol name for file paths
        self.symbol_name = symbol.replace("/", "_")
        
        # Create logs directory
        self.symbol_logs_dir = os.path.join(self.logs_dir, self.symbol_name)
        os.makedirs(self.symbol_logs_dir, exist_ok=True)
        
        # Log file paths
        self.metrics_log_path = os.path.join(self.symbol_logs_dir, "metrics.csv")
        self.predictions_log_path = os.path.join(self.symbol_logs_dir, "predictions.csv")
        
        # Thread safety
        self.metrics_lock = Lock()
        self.predictions_lock = Lock()
        
        # Initialize logs if they don't exist
        self._initialize_logs()
        
        logger.info(f"Performance tracker initialized for {symbol}")
    
    def _initialize_logs(self):
        """Initialize log files if they don't exist"""
        # Create metrics log
        if not os.path.exists(self.metrics_log_path):
            metrics_df = pd.DataFrame(columns=[
                'timestamp', 'model_version', 'mse', 'rmse', 'mae', 'mape', 
                'r2', 'directional_accuracy', 'sharpe_ratio', 'custom_metrics'
            ])
            metrics_df.to_csv(self.metrics_log_path, index=False)
        
        # Create predictions log
        if not os.path.exists(self.predictions_log_path):
            predictions_df = pd.DataFrame(columns=[
                'timestamp', 'model_version', 'actual', 'predicted', 
                'horizon', 'features_json'
            ])
            predictions_df.to_csv(self.predictions_log_path, index=False)
    
    def log_metrics(
        self, 
        metrics: Dict[str, float], 
        model_version: str, 
        timestamp: Optional[str] = None,
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            model_version: Version of the model used
            timestamp: Timestamp for the log entry (default: current time)
            custom_metrics: Additional custom metrics to log
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Extract standard metrics
        metrics_row = {
            'timestamp': timestamp,
            'model_version': model_version,
            'mse': metrics.get('mse', np.nan),
            'rmse': metrics.get('rmse', np.nan),
            'mae': metrics.get('mae', np.nan),
            'mape': metrics.get('mape', np.nan),
            'r2': metrics.get('r2', np.nan),
            'directional_accuracy': metrics.get('directional_accuracy', np.nan),
            'sharpe_ratio': metrics.get('sharpe_ratio', np.nan),
            'custom_metrics': json.dumps(custom_metrics or {})
        }
        
        # Create DataFrame for the new row
        new_row_df = pd.DataFrame([metrics_row])
        
        # Append to metrics log with thread safety
        with self.metrics_lock:
            try:
                # Load existing metrics
                metrics_df = pd.read_csv(self.metrics_log_path)
                
                # Append new row
                metrics_df = pd.concat([metrics_df, new_row_df], ignore_index=True)
                
                # Save back to file
                metrics_df.to_csv(self.metrics_log_path, index=False)
                
                logger.info(f"Logged metrics for {self.symbol} (version: {model_version})")
            except Exception as e:
                logger.error(f"Error logging metrics: {str(e)}")
    
    def log_prediction(
        self,
        actual: float,
        predicted: float,
        model_version: str,
        horizon: int = 1,
        features: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None
    ) -> None:
        """
        Log a prediction and actual value pair.
        
        Args:
            actual: Actual observed value
            predicted: Model's predicted value
            model_version: Version of the model used
            horizon: Prediction horizon (e.g., 1-day ahead)
            features: Feature values used for prediction
            timestamp: Timestamp for the log entry (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Create prediction entry
        prediction_row = {
            'timestamp': timestamp,
            'model_version': model_version,
            'actual': actual,
            'predicted': predicted,
            'horizon': horizon,
            'features_json': json.dumps(features or {})
        }
        
        # Create DataFrame for the new row
        new_row_df = pd.DataFrame([prediction_row])
        
        # Append to predictions log with thread safety
        with self.predictions_lock:
            try:
                # Load existing predictions
                predictions_df = pd.read_csv(self.predictions_log_path)
                
                # Append new row
                predictions_df = pd.concat([predictions_df, new_row_df], ignore_index=True)
                
                # Save back to file
                predictions_df.to_csv(self.predictions_log_path, index=False)
                
                logger.debug(f"Logged prediction for {self.symbol} (version: {model_version})")
            except Exception as e:
                logger.error(f"Error logging prediction: {str(e)}")
    
    def get_metrics_history(
        self,
        model_version: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical performance metrics.
        
        Args:
            model_version: Filter by model version
            start_time: Start time for filtering (ISO format)
            end_time: End time for filtering (ISO format)
            limit: Maximum number of entries to return
            
        Returns:
            DataFrame with historical metrics
        """
        try:
            # Load metrics
            metrics_df = pd.read_csv(self.metrics_log_path)
            
            # Apply filters
            if model_version is not None:
                metrics_df = metrics_df[metrics_df['model_version'] == model_version]
            
            if start_time is not None:
                metrics_df = metrics_df[metrics_df['timestamp'] >= start_time]
            
            if end_time is not None:
                metrics_df = metrics_df[metrics_df['timestamp'] <= end_time]
            
            # Sort by timestamp
            metrics_df = metrics_df.sort_values('timestamp', ascending=False)
            
            # Apply limit
            if limit is not None:
                metrics_df = metrics_df.head(limit)
            
            return metrics_df
        
        except Exception as e:
            logger.error(f"Error getting metrics history: {str(e)}")
            return pd.DataFrame()
    
    def get_predictions_history(
        self,
        model_version: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical predictions.
        
        Args:
            model_version: Filter by model version
            start_time: Start time for filtering (ISO format)
            end_time: End time for filtering (ISO format)
            limit: Maximum number of entries to return
            
        Returns:
            DataFrame with historical predictions
        """
        try:
            # Load predictions
            predictions_df = pd.read_csv(self.predictions_log_path)
            
            # Apply filters
            if model_version is not None:
                predictions_df = predictions_df[predictions_df['model_version'] == model_version]
            
            if start_time is not None:
                predictions_df = predictions_df[predictions_df['timestamp'] >= start_time]
            
            if end_time is not None:
                predictions_df = predictions_df[predictions_df['timestamp'] <= end_time]
            
            # Sort by timestamp
            predictions_df = predictions_df.sort_values('timestamp', ascending=False)
            
            # Apply limit
            if limit is not None:
                predictions_df = predictions_df.head(limit)
            
            return predictions_df
        
        except Exception as e:
            logger.error(f"Error getting predictions history: {str(e)}")
            return pd.DataFrame()
    
    def calculate_drift_metrics(
        self,
        reference_version: str,
        current_version: str,
        window_size: int = 100
    ) -> Dict[str, Any]:
        """
        Calculate drift metrics between model versions.
        
        Args:
            reference_version: Reference model version
            current_version: Current model version
            window_size: Number of most recent predictions to compare
            
        Returns:
            Dictionary with drift metrics
        """
        try:
            # Load predictions
            predictions_df = pd.read_csv(self.predictions_log_path)
            
            # Get predictions for both versions
            ref_preds = predictions_df[predictions_df['model_version'] == reference_version]
            curr_preds = predictions_df[predictions_df['model_version'] == current_version]
            
            # Get the most recent predictions
            ref_preds = ref_preds.sort_values('timestamp', ascending=False).head(window_size)
            curr_preds = curr_preds.sort_values('timestamp', ascending=False).head(window_size)
            
            if len(ref_preds) == 0 or len(curr_preds) == 0:
                return {
                    'error': 'Insufficient data for one or both versions',
                    'ref_count': len(ref_preds),
                    'curr_count': len(curr_preds)
                }
            
            # Calculate basic drift metrics
            ref_error = ref_preds['predicted'] - ref_preds['actual']
            curr_error = curr_preds['predicted'] - curr_preds['actual']
            
            # Mean difference in error
            mean_error_diff = abs(curr_error.mean() - ref_error.mean())
            
            # Distribution statistics
            ref_std = ref_error.std()
            curr_std = curr_error.std()
            std_ratio = curr_std / ref_std if ref_std > 0 else float('inf')
            
            # Calculate statistical significance (p-value) of the difference
            try:
                from scipy import stats
                _, p_value = stats.ttest_ind(curr_error, ref_error, equal_var=False)
            except ImportError:
                p_value = np.nan
            
            # Return drift metrics
            return {
                'reference_version': reference_version,
                'current_version': current_version,
                'reference_sample_size': len(ref_preds),
                'current_sample_size': len(curr_preds),
                'mean_error_diff': mean_error_diff,
                'std_ratio': std_ratio,
                'p_value': p_value,
                'ref_mean_error': ref_error.mean(),
                'curr_mean_error': curr_error.mean(),
                'ref_std_error': ref_std,
                'curr_std_error': curr_std,
                'significant_drift': p_value < 0.05 if not np.isnan(p_value) else None
            }
        
        except Exception as e:
            logger.error(f"Error calculating drift metrics: {str(e)}")
            return {'error': str(e)}
    
    def generate_performance_report(
        self,
        model_version: Optional[str] = None,
        output_dir: Optional[str] = None,
        last_n_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a performance report.
        
        Args:
            model_version: Filter by model version
            output_dir: Directory to save report artifacts
            last_n_days: Filter to include only the last N days
            
        Returns:
            Dictionary with report summary
        """
        try:
            # Load metrics and predictions
            metrics_df = pd.read_csv(self.metrics_log_path)
            predictions_df = pd.read_csv(self.predictions_log_path)
            
            # Filter by model version
            if model_version is not None:
                metrics_df = metrics_df[metrics_df['model_version'] == model_version]
                predictions_df = predictions_df[predictions_df['model_version'] == model_version]
            
            # Filter by time if needed
            if last_n_days is not None:
                from datetime import datetime, timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=last_n_days)
                
                start_date_str = start_date.isoformat()
                
                metrics_df = metrics_df[metrics_df['timestamp'] >= start_date_str]
                predictions_df = predictions_df[predictions_df['timestamp'] >= start_date_str]
            
            # Create output directory if needed
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                
                # Convert timestamps to datetime for plotting
                metrics_df['datetime'] = pd.to_datetime(metrics_df['timestamp'])
                
                # Plot performance metrics over time if we have data
                if len(metrics_df) > 0:
                    plt.figure(figsize=(12, 8))
                    
                    plt.subplot(2, 2, 1)
                    plt.plot(metrics_df['datetime'], metrics_df['rmse'])
                    plt.title('RMSE Over Time')
                    plt.xticks(rotation=45)
                    
                    plt.subplot(2, 2, 2)
                    plt.plot(metrics_df['datetime'], metrics_df['mae'])
                    plt.title('MAE Over Time')
                    plt.xticks(rotation=45)
                    
                    plt.subplot(2, 2, 3)
                    plt.plot(metrics_df['datetime'], metrics_df['r2'])
                    plt.title('R² Over Time')
                    plt.xticks(rotation=45)
                    
                    plt.subplot(2, 2, 4)
                    if 'directional_accuracy' in metrics_df.columns:
                        plt.plot(metrics_df['datetime'], metrics_df['directional_accuracy'])
                        plt.title('Directional Accuracy Over Time')
                        plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    
                    # Save the figure
                    report_path = os.path.join(output_dir, f"{self.symbol_name}_performance.png")
                    plt.savefig(report_path)
                    plt.close()
                
                # Plot predictions vs actuals if we have data
                if len(predictions_df) > 0:
                    # Convert timestamps to datetime
                    predictions_df['datetime'] = pd.to_datetime(predictions_df['timestamp'])
                    
                    # Sort by datetime
                    predictions_df = predictions_df.sort_values('datetime')
                    
                    # Plot
                    plt.figure(figsize=(12, 6))
                    plt.plot(predictions_df['datetime'], predictions_df['actual'], label='Actual')
                    plt.plot(predictions_df['datetime'], predictions_df['predicted'], label='Predicted')
                    plt.title(f'Predictions vs Actuals for {self.symbol}')
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Save the figure
                    predictions_path = os.path.join(output_dir, f"{self.symbol_name}_predictions.png")
                    plt.savefig(predictions_path)
                    plt.close()
            
            # Calculate summary statistics
            summary = {
                'symbol': self.symbol,
                'model_version': model_version or 'all',
                'metrics_count': len(metrics_df),
                'predictions_count': len(predictions_df),
                'time_range': {
                    'start': metrics_df['timestamp'].min() if len(metrics_df) > 0 else None,
                    'end': metrics_df['timestamp'].max() if len(metrics_df) > 0 else None
                }
            }
            
            # Calculate average metrics
            if len(metrics_df) > 0:
                for metric in ['mse', 'rmse', 'mae', 'mape', 'r2', 'directional_accuracy']:
                    if metric in metrics_df.columns:
                        summary[f'avg_{metric}'] = metrics_df[metric].mean()
            
            # Calculate prediction error statistics
            if len(predictions_df) > 0:
                predictions_df['error'] = predictions_df['predicted'] - predictions_df['actual']
                summary['prediction_error'] = {
                    'mean': predictions_df['error'].mean(),
                    'std': predictions_df['error'].std(),
                    'min': predictions_df['error'].min(),
                    'max': predictions_df['error'].max()
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {'error': str(e)}


# Global registry of performance trackers
_trackers = {}
_trackers_lock = Lock()

def get_tracker(symbol: str) -> PerformanceTracker:
    """
    Get or create a performance tracker for a symbol.
    
    Args:
        symbol: Trading symbol or model identifier
        
    Returns:
        Performance tracker instance
    """
    with _trackers_lock:
        if symbol not in _trackers:
            _trackers[symbol] = PerformanceTracker(symbol)
        return _trackers[symbol] 