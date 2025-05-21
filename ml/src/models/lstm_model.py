"""
LSTM Model for Time Series Prediction

This module implements an LSTM-based model for cryptocurrency price prediction.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging
from .base_model import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMModel(BaseModel):
    """LSTM model for time series forecasting"""
    
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
        Initialize the LSTM model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            seq_len: Length of input sequences
            forecast_horizon: Number of steps to forecast
            hidden_dim: Size of hidden layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
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
        
        # Define LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Adjust output dimension for bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Define attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1)
        )
        
        # Define output layers with a reduction in dimensionality
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )
        
        # Final output layer
        self.fc_out = nn.Linear(lstm_output_dim // 4, forecast_horizon * output_dim)
        
        self.to_device()
        logger.info(f"Initialized LSTM model with {num_layers} layers, {hidden_dim} hidden units")
        
    def attention_net(self, lstm_output):
        """
        Attention mechanism to focus on important timesteps
        
        Args:
            lstm_output: Output from LSTM layer
            
        Returns:
            Context vector
        """
        # Calculate attention weights
        attn_weights = self.attention(lstm_output)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Apply attention weights
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_output)
        return context.squeeze(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Predicted values tensor of shape (batch_size, forecast_horizon, output_dim)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         batch_size, self.hidden_dim).to(self.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        # Use the last timestep's output if attention gives issues
        if lstm_out.size(1) > 1:  # Only use attention if we have multiple timesteps
            try:
                attn_out = self.attention_net(lstm_out)
            except Exception as e:
                logger.warning(f"Attention failed, using last timestep: {str(e)}")
                attn_out = lstm_out[:, -1, :]
        else:
            attn_out = lstm_out.squeeze(1)
        
        # Fully connected layers
        fc_out = self.fc_layers(attn_out)
        out = self.fc_out(fc_out)
        
        # Reshape to (batch_size, forecast_horizon, output_dim)
        out = out.view(batch_size, self.forecast_horizon, self.output_dim)
        
        return out 