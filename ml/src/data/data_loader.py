"""
Data Loader for Crypto Market Data

This module provides functionality to load historical cryptocurrency data 
from Delta Exchange API or CSV files, and prepare it for ML model training.
"""

import os
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path

from ..api.delta_client import get_delta_client, DeltaExchangeClient
from ..utils.config import DATA_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataLoader:
    """Data loader for crypto market data from API or CSV files"""
    
    def __init__(self, client: Optional[DeltaExchangeClient] = None):
        """
        Initialize the data loader
        
        Args:
            client: Delta Exchange API client (uses default if None)
        """
        self.client = client or get_delta_client()
        
        # Ensure data directories exist
        self.raw_data_dir = Path(DATA_CONFIG["raw_data_dir"])
        self.processed_data_dir = Path(DATA_CONFIG["processed_data_dir"])
        
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def get_ohlcv_from_api(
        self, 
        symbol: str, 
        interval: str = "1h",
        days_back: int = 30,
        end_time: Optional[Union[int, datetime]] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV data from Delta Exchange API
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            interval: Timeframe ('1m', '5m', '1h', '4h', '1d', etc.)
            days_back: Number of days to look back
            end_time: End time for data fetching (defaults to now)
            
        Returns:
            DataFrame: OHLCV data with timestamp as index
        """
        logger.info(f"Fetching {symbol} {interval} data for the past {days_back} days from API")
        
        candles = self.client.get_historical_ohlcv(
            symbol=symbol,
            interval=interval,
            days_back=days_back,
            end_time=end_time
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        
        # Use timestamp as index
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    def get_ohlcv_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load OHLCV data from a CSV file
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame: OHLCV data with timestamp as index
        """
        logger.info(f"Loading OHLCV data from CSV: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime index if it exists
        if 'timestamp' in df.columns:
            if df['timestamp'].iloc[0].isdigit():  # Check if timestamp is numeric
                df['datetime'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            else:
                # Try to parse as ISO date
                df['datetime'] = pd.to_datetime(df['timestamp'])
                
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, symbol: str, interval: str) -> str:
        """
        Save OHLCV data to a CSV file
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol (e.g., 'BTCUSD')
            interval: Timeframe ('1h', '4h', '1d', etc.)
            
        Returns:
            str: Path to the saved CSV file
        """
        # Create filename with current date
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{symbol}_{interval}_{date_str}.csv"
        filepath = self.raw_data_dir / filename
        
        # Reset index to include datetime as a column
        df_to_save = df.reset_index()
        
        # Ensure we have a timestamp column (milliseconds since epoch)
        if 'timestamp' not in df_to_save.columns and 'datetime' in df_to_save.columns:
            df_to_save['timestamp'] = df_to_save['datetime'].astype(int) // 10**6
        
        logger.info(f"Saving OHLCV data to {filepath}")
        df_to_save.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def get_data(
        self, 
        symbol: str, 
        interval: str = "1h",
        days_back: int = 30,
        use_cache: bool = True,
        save_csv: bool = True
    ) -> pd.DataFrame:
        """
        Get OHLCV data with caching support
        
        This method first tries to load data from a cached CSV file if it exists and is recent.
        If no cache is available or it's too old, it fetches data from the API.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            interval: Timeframe ('1h', '4h', '1d', etc.)
            days_back: Number of days to look back
            use_cache: Whether to use cached data if available
            save_csv: Whether to save fetched data to CSV
            
        Returns:
            DataFrame: OHLCV data with timestamp as index
        """
        # If caching is enabled, try to find a recent CSV file
        if use_cache:
            # List all files in the raw data directory
            files = os.listdir(self.raw_data_dir)
            
            # Filter files for this symbol and interval
            matching_files = [
                f for f in files 
                if f.startswith(f"{symbol}_{interval}") and f.endswith(".csv")
            ]
            
            if matching_files:
                # Sort by modification time (most recent first)
                matching_files.sort(
                    key=lambda f: os.path.getmtime(os.path.join(self.raw_data_dir, f)),
                    reverse=True
                )
                
                most_recent = matching_files[0]
                file_path = os.path.join(self.raw_data_dir, most_recent)
                file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))).days
                
                # If the file is recent enough (less than 1 day old for intervals <= 1h, otherwise 7 days)
                max_age = 1 if interval.endswith(('m', 'h')) and int(interval[:-1]) <= 1 else 7
                
                if file_age_days < max_age:
                    logger.info(f"Using cached data from {most_recent} (age: {file_age_days} days)")
                    return self.get_ohlcv_from_csv(file_path)
                else:
                    logger.info(f"Cached data is too old ({file_age_days} days), fetching from API")
        
        # If no cache or cache is disabled/outdated, fetch from API
        df = self.get_ohlcv_from_api(symbol, interval, days_back)
        
        # Save to CSV if requested
        if save_csv and not df.empty:
            self.save_to_csv(df, symbol, interval)
        
        return df
    
    def preprocess_data(
        self, 
        df: pd.DataFrame, 
        add_features: bool = True,
        normalize: bool = True,
        target_column: str = 'close',
        sequence_length: int = 48,
        train_split: float = 0.8
    ) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
        """
        Preprocess OHLCV data for model training
        
        Args:
            df: DataFrame with OHLCV data
            add_features: Whether to add technical indicators
            normalize: Whether to normalize the data
            target_column: Column to use as prediction target
            sequence_length: Number of time steps for each sample
            train_split: Proportion of data to use for training
            
        Returns:
            Dict containing preprocessed data and metadata:
                - X_train: Training features
                - y_train: Training targets
                - X_val: Validation features
                - y_val: Validation targets
                - feature_columns: List of feature columns
                - target_column: Target column name
                - scaler_params: Parameters for reversing normalization
        """
        # Make a copy to avoid modifying the original DataFrame
        data = df.copy()
        
        # Verify required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {missing_columns}")
        
        # Add technical indicators as features if requested
        if add_features:
            data = self.add_technical_indicators(data)
        
        # Drop rows with NaN values (e.g., from technical indicators)
        data.dropna(inplace=True)
        
        # Select feature columns - exclude the timestamp if it's in columns
        feature_columns = [col for col in data.columns if col != 'timestamp']
        
        # Prepare normalization parameters
        scaler_params = {}
        
        # Normalize if requested
        if normalize:
            data, scaler_params = self.normalize_data(data[feature_columns])
        else:
            data = data[feature_columns].values
        
        # Create sequences for time-series prediction
        X, y = self.create_sequences(
            data=data,
            target_idx=feature_columns.index(target_column),
            sequence_length=sequence_length
        )
        
        # Split into training and validation sets
        split_idx = int(len(X) * train_split)
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return {
            'X_train': X_train, 
            'y_train': y_train,
            'X_val': X_val, 
            'y_val': y_val,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'scaler_params': scaler_params
        }
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        data = df.copy()
        
        # Moving Averages
        data['ma7'] = data['close'].rolling(window=7).mean()
        data['ma14'] = data['close'].rolling(window=14).mean()
        data['ma30'] = data['close'].rolling(window=30).mean()
        
        # Exponential Moving Average
        data['ema12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema26'] = data['close'].ewm(span=26, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        data['macd'] = data['ema12'] - data['ema26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        window = 20
        std_dev = 2
        rolling_mean = data['close'].rolling(window=window).mean()
        rolling_std = data['close'].rolling(window=window).std()
        data['bb_upper'] = rolling_mean + (rolling_std * std_dev)
        data['bb_lower'] = rolling_mean - (rolling_std * std_dev)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / rolling_mean
        
        # RSI (Relative Strength Index)
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Calculate first RSIs
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Price change percentages
        data['pct_change_1d'] = data['close'].pct_change(periods=1)
        data['pct_change_3d'] = data['close'].pct_change(periods=3)
        data['pct_change_7d'] = data['close'].pct_change(periods=7)
        
        # Volume indicators
        data['volume_ma7'] = data['volume'].rolling(window=7).mean()
        data['volume_change'] = data['volume'].pct_change()
        
        return data
    
    def normalize_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Normalize data using min-max scaling
        
        Args:
            df: DataFrame with feature data
            
        Returns:
            Tuple containing:
                - Normalized data as numpy array
                - Dictionary with scaling parameters for each column
        """
        # Store min and max values for each column to denormalize later
        scaler_params = {
            'min': {},
            'max': {}
        }
        
        normalized_data = np.zeros_like(df.values, dtype=np.float32)
        
        for i, col_name in enumerate(df.columns):
            col_data = df[col_name].values
            col_min = np.min(col_data)
            col_max = np.max(col_data)
            
            # Store params for denormalization
            scaler_params['min'][col_name] = float(col_min)
            scaler_params['max'][col_name] = float(col_max)
            
            # Apply min-max scaling
            if col_max > col_min:
                normalized_data[:, i] = (col_data - col_min) / (col_max - col_min)
            else:
                # If min equals max, set to 0.5 to avoid division by zero
                normalized_data[:, i] = 0.5
        
        return normalized_data, scaler_params
    
    def denormalize(self, data: np.ndarray, column_idx: int, scaler_params: Dict, column_name: str) -> np.ndarray:
        """
        Denormalize data from a specific column
        
        Args:
            data: Normalized data
            column_idx: Index of the column to denormalize
            scaler_params: Scaling parameters
            column_name: Name of the column
            
        Returns:
            Denormalized data
        """
        col_min = scaler_params['min'][column_name]
        col_max = scaler_params['max'][column_name]
        
        return data * (col_max - col_min) + col_min
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        target_idx: int,
        sequence_length: int,
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time-series prediction
        
        Args:
            data: Input data as numpy array
            target_idx: Index of the target column
            sequence_length: Number of time steps for each sequence
            forecast_horizon: Number of steps ahead to predict
            
        Returns:
            Tuple containing sequences of features (X) and targets (y)
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            # Sequence of historical data
            X.append(data[i:(i + sequence_length)])
            
            # Target value (next time step or further in the future)
            y.append(data[i + sequence_length + forecast_horizon - 1, target_idx])
        
        return np.array(X), np.array(y)
    
    def load_multiple_symbols(
        self, 
        symbols: List[str], 
        interval: str = "1h",
        days_back: int = 30,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple trading symbols
        
        Args:
            symbols: List of trading symbols
            interval: Timeframe for the data
            days_back: Number of days to look back
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        result = {}
        
        for symbol in symbols:
            try:
                df = self.get_data(symbol, interval, days_back, use_cache)
                result[symbol] = df
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {str(e)}")
        
        return result


def load_data(
    symbol: str = "BTCUSD",
    interval: str = "1h",
    days_back: int = 30,
    use_cache: bool = True,
    preprocess: bool = True,
    **preprocess_kwargs
) -> Union[pd.DataFrame, Dict]:
    """
    Convenience function to load and optionally preprocess data
    
    Args:
        symbol: Trading symbol
        interval: Timeframe
        days_back: Number of days to look back
        use_cache: Whether to use cached data
        preprocess: Whether to preprocess the data
        **preprocess_kwargs: Additional arguments for preprocessing
        
    Returns:
        Raw DataFrame or preprocessed data dictionary
    """
    loader = CryptoDataLoader()
    df = loader.get_data(symbol, interval, days_back, use_cache)
    
    if preprocess:
        return loader.preprocess_data(df, **preprocess_kwargs)
    
    return df


if __name__ == "__main__":
    # Example usage
    loader = CryptoDataLoader()
    
    # Load BTC/USD data and save to CSV
    btc_data = loader.get_data(symbol="BTCUSD", interval="1h", days_back=30)
    print(f"Loaded {len(btc_data)} rows of BTC/USD data")
    
    # Preprocess data for training
    processed_data = loader.preprocess_data(
        btc_data,
        add_features=True,
        normalize=True,
        target_column='close',
        sequence_length=24,  # 24 hours of history
        train_split=0.8
    )
    
    print(f"Training data shape: {processed_data['X_train'].shape}")
    print(f"Validation data shape: {processed_data['X_val'].shape}")
    print(f"Feature columns: {processed_data['feature_columns']}") 