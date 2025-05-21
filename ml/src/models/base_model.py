"""
Base Model for all model implementations
Provides common functionality and interface
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all models.
    Enforces a consistent interface for all models.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the base model.
        
        Args:
            model_path: Path to a saved model file (if loading a pre-trained model)
        """
        self.model = None
    
    @abstractmethod
    def _build_model(self):
        """
        Build the model architecture.
        To be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int,
        epochs: int,
        patience: int,
        log_dir: Optional[str],
        checkpoint_dir: Optional[str],
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            patience: Patience for early stopping
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for model checkpoints
            
        Returns:
            Dictionary with training history
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, model_path: str) -> None:
        """
        Save the model.
        
        Args:
            model_path: Path to save the model
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, model_path: str, **kwargs):
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the saved model
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded model
        """
        pass


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