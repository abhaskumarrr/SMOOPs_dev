"""
Model Training Script with TensorBoard Integration

This script provides a complete pipeline for training cryptocurrency price prediction models
with TensorBoard integration for monitoring training progress and Apple Silicon optimization.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import project modules
from src.data.data_loader import CryptoDataLoader, load_data
from src.data.preprocessor import EnhancedPreprocessor, preprocess_with_enhanced_features
from src.models.base_model import BaseModel, ModelFactory, DirectionalLoss
from src.training.trainer import Trainer
from src.training.data_preparation import prepare_time_series_data, prepare_multi_target_data
from src.training.evaluation import evaluate_model, evaluate_forecasts, plot_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train cryptocurrency price prediction models")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, help="Path to the data file (CSV)")
    parser.add_argument("--target-column", type=str, default="close", help="Name of the target column to predict")
    parser.add_argument("--seq-len", type=int, default=60, help="Length of input sequence")
    parser.add_argument("--forecast-horizon", type=int, default=5, help="Number of steps to forecast")
    
    # Model arguments
    parser.add_argument("--model-type", type=str, default="lstm", 
                        choices=["lstm", "gru", "transformer", "cnn_lstm"],
                        help="Type of model to train")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Size of hidden layers")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Number of epochs with no improvement before stopping early")
    parser.add_argument("--val-ratio", type=float, default=0.2, 
                        help="Ratio of data to use for validation")
    parser.add_argument("--test-ratio", type=float, default=0.1, 
                        help="Ratio of data to use for testing")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="models", 
                        help="Directory to save models and logs")
    parser.add_argument("--experiment-name", type=str, default=None, 
                        help="Name for the experiment (default: model_type_timestamp)")
    parser.add_argument("--save-freq", type=int, default=5, 
                        help="Save checkpoints every n epochs")
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default=None, 
                        choices=["cpu", "cuda", "mps", None],
                        help="Device to use for training (cpu, cuda, mps, or None for auto-detection)")
    parser.add_argument("--num-workers", type=int, default=4, 
                        help="Number of workers for data loading")
    parser.add_argument("--mixed-precision", action="store_true", 
                        help="Use mixed precision training if available")
    
    return parser.parse_args()


def load_and_preprocess_data(args):
    """Load and preprocess data"""
    
    if args.data_path:
        # Load data from file
        logger.info(f"Loading data from {args.data_path}")
        data = pd.read_csv(args.data_path)
    else:
        # Use data loader to fetch data
        logger.info("Fetching data using CryptoDataLoader")
        loader = CryptoDataLoader()
        data = loader.get_historical_ohlcv("BTC/USDT", "1h", limit=5000)
    
    # Apply preprocessing
    logger.info("Applying preprocessing")
    preprocessor = EnhancedPreprocessor()
    data = preprocessor.add_all_features(data)
    
    # Remove rows with NaN values (usually at the beginning due to indicators)
    data = data.dropna()
    
    # Normalize data
    scaled_data, scalers = preprocessor.normalize_data(
        data, 
        method='standard',
        exclude_columns=['timestamp', 'date']
    )
    
    logger.info(f"Data shape after preprocessing: {scaled_data.shape}")
    return scaled_data, scalers


def prepare_datasets(data, args):
    """Prepare datasets for training"""
    
    # Determine feature columns (exclude timestamp, date)
    feature_columns = [col for col in data.columns 
                       if col != args.target_column 
                       and 'timestamp' not in col 
                       and 'date' not in col]
    
    logger.info(f"Target column: {args.target_column}")
    logger.info(f"Number of features: {len(feature_columns)}")
    
    # Prepare data
    data_dict = prepare_time_series_data(
        data=data,
        target_column=args.target_column,
        feature_columns=feature_columns,
        seq_len=args.seq_len,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers
    )
    
    logger.info(f"Train dataset size: {data_dict['train_size']}")
    logger.info(f"Validation dataset size: {data_dict['val_size']}")
    logger.info(f"Test dataset size: {data_dict['test_size']}")
    
    return data_dict


def create_model(args, input_dim, output_dim):
    """Create a model instance based on command line arguments"""
    
    model = ModelFactory.create_model(
        model_type=args.model_type,
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=args.seq_len,
        forecast_horizon=args.forecast_horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=args.device
    )
    
    logger.info(f"Created {args.model_type.upper()} model")
    logger.info(f"Model parameters: input_dim={input_dim}, output_dim={output_dim}, "
               f"hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")
    
    return model


def train_model(model, data_dict, args):
    """Train the model"""
    
    # Set up optimizer
    optimizer_kwargs = {"lr": args.learning_rate, "weight_decay": 1e-5}
    
    # Set up learning rate scheduler
    lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
    lr_scheduler_kwargs = {"mode": "min", "factor": 0.5, "patience": 5, "verbose": True}
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=data_dict["train_loader"],
        val_dataloader=data_dict["val_loader"],
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler_cls=lr_scheduler_cls,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        log_dir=os.path.join(args.output_dir, "logs", "tensorboard", args.experiment_name),
        checkpoints_dir=os.path.join(args.output_dir, "checkpoints", args.experiment_name),
        experiment_name=args.experiment_name,
        mixed_precision=args.mixed_precision
    )
    
    # Train model
    logger.info(f"Training model for {args.epochs} epochs")
    train_losses, val_losses = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_frequency=args.save_freq
    )
    
    return trainer, train_losses, val_losses


def evaluate(model, data_dict, args):
    """Evaluate the trained model"""
    
    logger.info("Evaluating model on test set")
    test_results = evaluate_model(model, data_dict["test_loader"], return_predictions=True)
    
    # Save evaluation metrics
    metrics_path = os.path.join(args.output_dir, "results", args.experiment_name, "metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    with open(metrics_path, "w") as f:
        json.dump(test_results["metrics"], f, indent=4)
    
    logger.info(f"Evaluation metrics saved to {metrics_path}")
    
    # Generate forecast plots
    logger.info("Generating forecast plots")
    forecast_dir = os.path.join(args.output_dir, "results", args.experiment_name, "forecasts")
    os.makedirs(forecast_dir, exist_ok=True)
    
    # Plot overall predictions
    plot_path = os.path.join(forecast_dir, "test_predictions.png")
    plot_predictions(
        test_results["y_true"],
        test_results["y_pred"],
        save_path=plot_path,
        title=f"{args.model_type.upper()} Model Predictions vs Actual Values"
    )
    
    # Generate individual forecasts
    forecast_results = evaluate_forecasts(
        model,
        data_dict["test_loader"],
        n_forecasts=5,
        output_dir=forecast_dir
    )
    
    return test_results, forecast_results


def main():
    """Main function to train and evaluate models"""
    
    # Parse command line arguments
    args = parse_args()
    
    # Set experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.model_type}_{timestamp}"
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "logs", "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    
    # Save arguments
    args_path = os.path.join(args.output_dir, "results", args.experiment_name, "args.json")
    os.makedirs(os.path.dirname(args_path), exist_ok=True)
    
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    
    logger.info(f"Arguments saved to {args_path}")
    
    # Load and preprocess data
    data, scalers = load_and_preprocess_data(args)
    
    # Prepare datasets
    data_dict = prepare_datasets(data, args)
    
    # Create model
    model = create_model(args, data_dict["input_dim"], data_dict["output_dim"])
    
    # Train model
    trainer, train_losses, val_losses = train_model(model, data_dict, args)
    
    # Evaluate model
    test_results, forecast_results = evaluate(model, data_dict, args)
    
    logger.info("Training and evaluation completed successfully")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")
    logger.info(f"Test RMSE: {test_results['metrics']['rmse']:.6f}")
    logger.info(f"Test Directional Accuracy: {test_results['metrics']['directional_accuracy']:.4f}")
    
    # Log final message with location of results
    results_dir = os.path.join(args.output_dir, "results", args.experiment_name)
    logger.info(f"Results saved to {results_dir}")
    logger.info(f"TensorBoard logs saved to {os.path.join(args.output_dir, 'logs', 'tensorboard', args.experiment_name)}")
    logger.info(f"Run 'tensorboard --logdir={os.path.join(args.output_dir, 'logs', 'tensorboard')}' to view training metrics")


if __name__ == "__main__":
    main() 