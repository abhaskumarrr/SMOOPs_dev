"""
Base Model for Time Series Prediction

This module defines the base model class that all model architectures will extend.
It implements common functionality like device handling, saving/loading, etc.
"""

import os
import torch
import torch.nn as nn
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel(nn.Module, ABC):
    """Base class for all time series forecasting models"""
    
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        seq_len: int,
        forecast_horizon: int,
        device: Optional[str] = None
    ):
        """
        Initialize the base model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            seq_len: Length of input sequences
            forecast_horizon: Number of steps to forecast
            device: Device to use for computation ('cpu', 'cuda', 'mps', or None for auto-detection)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        
        # Auto-detect device if none is provided
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'  # Apple Silicon GPU
                logger.info("Using MPS (Apple Silicon) device")
            elif torch.cuda.is_available():
                device = 'cuda'
                logger.info("Using CUDA device")
            else:
                device = 'cpu'
                logger.info("Using CPU device")
        
        self.device = torch.device(device)
        logger.info(f"Model initialized on device: {self.device}")
    
    def to_device(self):
        """Move model to the specified device"""
        return self.to(self.device)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Predicted values tensor of shape (batch_size, forecast_horizon, output_dim)
        """
        pass
    
    def save(self, path: str):
        """
        Save model to disk
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'seq_len': self.seq_len,
                'forecast_horizon': self.forecast_horizon,
                'model_type': self.__class__.__name__
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """
        Load model from disk
        
        Args:
            path: Path to load the model from
            device: Device to load the model to
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        config = checkpoint['model_config']
        
        # Instantiate the correct model class based on saved type
        model_type = config.pop('model_type', cls.__name__)
        
        # Import all model classes 
        from .lstm_model import LSTMModel
        from .gru_model import GRUModel
        from .transformer_model import TransformerModel
        from .cnn_lstm_model import CNNLSTMModel
        
        # Get the class by name
        model_class = {
            'LSTMModel': LSTMModel,
            'GRUModel': GRUModel,
            'TransformerModel': TransformerModel, 
            'CNNLSTMModel': CNNLSTMModel,
            'BaseModel': cls
        }.get(model_type, cls)
        
        # Create model instance
        model = model_class(**config, device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to_device()
        model.eval()
        return model


class DirectionalLoss(nn.Module):
    """
    Custom loss function that penalizes incorrect direction predictions more heavily.
    
    This is especially important for trading models where predicting the direction
    correctly (up/down) is often more important than the exact magnitude.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 2.0):
        """
        Initialize the directional loss function.
        
        Args:
            alpha: Weight for the MSE component (0-1)
            beta: Multiplier for incorrect direction predictions
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate the directional loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Loss value
        """
        # Calculate MSE component
        mse_loss = self.mse(y_pred, y_true)
        
        # Calculate directional component
        # 1 if direction is the same, 0 if different
        direction_pred = (y_pred[:, 1:] - y_pred[:, :-1]) > 0
        direction_true = (y_true[:, 1:] - y_true[:, :-1]) > 0
        
        # Calculate directional accuracy
        direction_match = (direction_pred == direction_true).float()
        direction_loss = 1.0 - direction_match.mean()
        
        # Combined loss with higher penalty for incorrect directions
        combined_loss = self.alpha * mse_loss + (1 - self.alpha) * self.beta * direction_loss
        
        return combined_loss


class ModelFactory:
    """
    Factory class for creating different model architectures.
    Allows easy switching between model types for experimentation.
    """
    
    @staticmethod
    def create_model(
        model_type: str,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        forecast_horizon: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: Optional[str] = None,
        **kwargs
    ) -> BaseModel:
        """
        Create a model instance of the specified type.
        
        Args:
            model_type: Type of model to create ('lstm', 'gru', 'transformer', 'cnn_lstm')
            input_dim: Number of input features
            output_dim: Number of output features
            seq_len: Length of input sequences
            forecast_horizon: Number of steps to forecast
            hidden_dim: Size of hidden layers
            num_layers: Number of layers (for RNNs and Transformer)
            dropout: Dropout rate
            device: Device to use for computation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Model instance
        """
        # Import model classes
        from .lstm_model import LSTMModel
        from .gru_model import GRUModel
        from .transformer_model import TransformerModel
        from .cnn_lstm_model import CNNLSTMModel
        
        model_type = model_type.lower()
        
        if model_type == 'lstm':
            return LSTMModel(
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                forecast_horizon=forecast_horizon,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                device=device,
                **kwargs
            )
        elif model_type == 'gru':
            return GRUModel(
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                forecast_horizon=forecast_horizon,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                device=device,
                **kwargs
            )
        elif model_type == 'transformer':
            return TransformerModel(
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                forecast_horizon=forecast_horizon,
                d_model=hidden_dim,
                nhead=kwargs.get('nhead', 4),
                num_layers=num_layers,
                dropout=dropout,
                device=device,
                **kwargs
            )
        elif model_type == 'cnn_lstm':
            return CNNLSTMModel(
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                forecast_horizon=forecast_horizon,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                kernel_size=kwargs.get('kernel_size', 3),
                dropout=dropout,
                device=device,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}") 