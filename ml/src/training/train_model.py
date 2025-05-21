"""
Model Training Module

This module provides functions for training Smart Money Concepts models.
"""

import os
import logging
import json
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple

from ..data.data_loader import load_data
from ..models.model_registry import ModelRegistry
from ..utils.metrics import calculate_metrics

# Configure logging
logger = logging.getLogger(__name__)

def train_model(
    symbol: str,
    model_type: str,
    data_path: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    sequence_length: int = 60,
    forecast_horizon: int = 1,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Train a model for the specified symbol.
    
    Args:
        symbol: Trading symbol (e.g., "BTC-USDT")
        model_type: Type of model to train (e.g., "smc_transformer")
        data_path: Path to load data from
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        test_ratio: Proportion of data to use for testing
        sequence_length: Number of time steps in input sequence
        forecast_horizon: Number of time steps to predict
        batch_size: Batch size for training
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        **kwargs: Additional model-specific parameters
        
    Returns:
        Dictionary containing model info and metrics
    """
    logger.info(f"Training {model_type} model for {symbol}")
    
    # Load data
    train_dataloader, val_dataloader, test_dataloader = load_data(
        symbol=symbol,
        data_path=data_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size
    )
    
    # Initialize model based on type
    registry = ModelRegistry()
    model = registry.create_model(
        model_type=model_type,
        input_dim=train_dataloader.dataset.features_dim,
        output_dim=forecast_horizon,
        **kwargs
    )
    
    # Create trainer
    from .trainer import Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=learning_rate,
        device=torch.device("mps" if torch.backends.mps.is_available() else 
                            "cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Train the model
    trainer.train(
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_dataloader)
    
    # Save the model
    model_info = registry.save_model(
        model=model,
        symbol=symbol,
        model_type=model_type,
        metrics=test_metrics,
        params={
            "sequence_length": sequence_length,
            "forecast_horizon": forecast_horizon,
            "train_params": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": trainer.current_epoch
            },
            **kwargs
        }
    )
    
    return model_info 