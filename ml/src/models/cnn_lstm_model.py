"""
CNN-LSTM Hybrid Model for Time Series Prediction

This module implements a hybrid model that combines CNN layers for feature extraction 
with LSTM layers for temporal analysis. This architecture is particularly effective 
for technical indicators and pattern recognition in cryptocurrency prices.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import logging
from .base_model import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNLSTMModel(BaseModel):
    """Hybrid CNN-LSTM model for time series forecasting"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        forecast_horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        kernel_size: int = 3,
        cnn_channels: List[int] = None,
        dropout: float = 0.2,
        bidirectional: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialize the CNN-LSTM model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            seq_len: Length of input sequences
            forecast_horizon: Number of steps to forecast
            hidden_dim: Size of hidden layers
            num_layers: Number of LSTM layers
            kernel_size: Size of the convolutional kernel
            cnn_channels: List of CNN channels for each layer
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
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Default CNN channels if not provided
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        self.cnn_channels = cnn_channels
        
        # Define CNN layers for feature extraction
        cnn_layers = []
        in_channels = 1  # Start with a single channel (will reshape input)
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,  # Same padding
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate the output size of CNN
        # The sequence length remains roughly the same due to padding and maxpool
        cnn_output_dim = cnn_channels[-1]
        
        # Define LSTM layers for temporal analysis
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Adjust output dimension for bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Define output layers with skip connections
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout / 2)
        
        # Final output layer
        self.fc_out = nn.Linear(hidden_dim // 2, forecast_horizon * output_dim)
        
        self.to_device()
        logger.info(f"Initialized CNN-LSTM model with {len(cnn_channels)} CNN layers, {num_layers} LSTM layers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Predicted values tensor of shape (batch_size, forecast_horizon, output_dim)
        """
        batch_size = x.size(0)
        
        # Reshape input for CNN (batch_size, channels, sequence_length * features)
        # We'll treat the features as additional sequences for the CNN
        x = x.view(batch_size, 1, -1)
        
        # Apply CNN for feature extraction
        cnn_out = self.cnn(x)
        
        # Reshape for LSTM (batch_size, seq_len, features)
        # We'll keep the original sequence length and treat CNN outputs as features
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # Ensure sequence length matches the original
        # If CNN changes sequence length, we need to adjust
        if cnn_out.size(1) != self.seq_len:
            # Either truncate or pad
            if cnn_out.size(1) > self.seq_len:
                cnn_out = cnn_out[:, :self.seq_len, :]
            else:
                padding = torch.zeros(batch_size, self.seq_len - cnn_out.size(1), cnn_out.size(2)).to(self.device)
                cnn_out = torch.cat([cnn_out, padding], dim=1)
        
        # Initialize hidden state and cell state for LSTM
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         batch_size, self.hidden_dim).to(self.device)
        
        # Apply LSTM for temporal analysis
        lstm_out, _ = self.lstm(cnn_out, (h0, c0))
        
        # Use the last timestamp for prediction
        out = lstm_out[:, -1, :]
        
        # Fully connected layers with batch normalization
        fc1_out = self.fc1(out)
        fc1_out = self.bn1(fc1_out)
        fc1_out = torch.relu(fc1_out)
        fc1_out = self.dropout1(fc1_out)
        
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.bn2(fc2_out)
        fc2_out = torch.relu(fc2_out)
        fc2_out = self.dropout2(fc2_out)
        
        # Final output layer
        out = self.fc_out(fc2_out)
        
        # Reshape to (batch_size, forecast_horizon, output_dim)
        out = out.view(batch_size, self.forecast_horizon, self.output_dim)
        
        return out 