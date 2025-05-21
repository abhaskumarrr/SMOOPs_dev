"""
GRU Model for Time Series Prediction

This module implements a GRU-based model for cryptocurrency price prediction.
GRUs are computationally more efficient than LSTMs while still capturing temporal dependencies.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging
from .base_model import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GRUModel(BaseModel):
    """GRU model for time series forecasting"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        forecast_horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialize the GRU model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            seq_len: Length of input sequences
            forecast_horizon: Number of steps to forecast
            hidden_dim: Size of hidden layers
            num_layers: Number of GRU layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional GRU
            device: Device to use for computation
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon,
            device=device
        )
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Define GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Adjust output dimension for bidirectional
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Define output layers with residual connections
        self.fc1 = nn.Linear(gru_output_dim, gru_output_dim // 2)
        self.bn1 = nn.BatchNorm1d(gru_output_dim // 2)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(gru_output_dim // 2, gru_output_dim // 2)
        self.bn2 = nn.BatchNorm1d(gru_output_dim // 2)
        self.dropout2 = nn.Dropout(dropout / 2)
        
        # Final output layer
        self.fc_out = nn.Linear(gru_output_dim // 2, forecast_horizon * output_dim)
        
        self.to_device()
        logger.info(f"Initialized GRU model with {num_layers} layers, {hidden_dim} hidden units")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Predicted values tensor of shape (batch_size, forecast_horizon, output_dim)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         batch_size, self.hidden_dim).to(self.device)
        
        # GRU forward pass
        gru_out, _ = self.gru(x, h0)
        
        # Use the last timestamp for prediction
        out = gru_out[:, -1, :]
        
        # First fully connected layer with batch normalization and residual connection
        fc1_out = self.fc1(out)
        fc1_out = self.bn1(fc1_out)
        fc1_out = torch.relu(fc1_out)
        fc1_out = self.dropout1(fc1_out)
        
        # Second fully connected layer with residual connection
        res_out = self.fc2(fc1_out)
        res_out = self.bn2(res_out)
        res_out = torch.relu(res_out)
        res_out = self.dropout2(res_out)
        
        # Residual connection
        res_out = res_out + fc1_out  # Residual connection requires same dimensions
        
        # Final output layer
        out = self.fc_out(res_out)
        
        # Reshape to (batch_size, forecast_horizon, output_dim)
        out = out.view(batch_size, self.forecast_horizon, self.output_dim)
        
        return out 