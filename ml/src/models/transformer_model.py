"""
Transformer Model for Time Series Prediction

This module implements a Transformer-based model for cryptocurrency price prediction.
Transformers excel at capturing long-range dependencies through their self-attention mechanism.
"""

import torch
import torch.nn as nn
import math
from typing import Optional
import logging
from .base_model import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model to give time-based context to sequence positions
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding
        
        Args:
            d_model: The embedding dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Encoded tensor of same shape
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(BaseModel):
    """Transformer model for time series forecasting"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        forecast_horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        activation: str = 'gelu',
        device: Optional[str] = None,
    ):
        """
        Initialize the Transformer model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            seq_len: Length of input sequences
            forecast_horizon: Number of steps to forecast
            d_model: Size of the transformer embedding
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Size of feedforward network in transformer
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu')
            device: Device to use for computation
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon,
            device=device
        )
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Input encoding layers
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len=seq_len, dropout=dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model * seq_len, d_model * 2)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout / 2)
        
        # Final output layer
        self.fc_out = nn.Linear(d_model, forecast_horizon * output_dim)
        
        # Create attention mask
        self.register_buffer('mask', self._generate_square_subsequent_mask(seq_len))
        
        self.to_device()
        logger.info(f"Initialized Transformer model with {num_layers} layers, {nhead} attention heads")
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate mask to prevent attending to future positions.
        This enforces causality for time series.
        
        Args:
            sz: Size of square matrix
            
        Returns:
            Mask tensor
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Predicted values tensor of shape (batch_size, forecast_horizon, output_dim)
        """
        batch_size = x.size(0)
        
        # Input embedding
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        x = self.positional_encoding(x)  # Add positional encoding
        
        # Transformer encoder
        # Using causal mask for time series
        try:
            x = self.transformer_encoder(x, self.mask)  # (batch_size, seq_len, d_model)
        except RuntimeError:
            # Fallback in case of mask issues
            logger.warning("Transformer mask error, using self-attention without mask")
            x = self.transformer_encoder(x)
        
        # Flatten the sequence dimension for the fully connected layers
        x = x.reshape(batch_size, -1)  # (batch_size, seq_len * d_model)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        
        # Final output
        x = self.fc_out(x)
        
        # Reshape to (batch_size, forecast_horizon, output_dim)
        x = x.view(batch_size, self.forecast_horizon, self.output_dim)
        
        return x 