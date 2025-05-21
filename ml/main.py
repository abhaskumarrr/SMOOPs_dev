"""
SMOOPs ML System Main Entry Point

This module serves as the entry point for the SMOOPs ML system,
providing command-line interface for training and serving models.
"""

import os
import argparse
import logging
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_command(args):
    """
    Train a new model based on command line arguments.
    
    Args:
        args: Command line arguments
    """
    from src.data.data_loader import load_data
    from src.training.trainer import Trainer
    from src.training.train_model import train_model
    
    logger.info("Starting model training")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Symbol: {args.symbol}")
    
    # Parse additional training args from JSON if provided
    training_args = {}
    if args.training_args:
        try:
            training_args = json.loads(args.training_args)
            logger.info(f"Additional training args: {training_args}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON for training args: {args.training_args}")
    
    # Train the model
    model_info = train_model(
        symbol=args.symbol,
        model_type=args.model_type,
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        **training_args
    )
    
    logger.info(f"Model training completed. Model saved as {model_info.get('version')}")
    logger.info(f"Metrics: {model_info.get('metrics')}")
    
    return model_info


def tune_command(args):
    """
    Tune hyperparameters for a model based on command line arguments.
    
    Args:
        args: Command line arguments
    """
    from src.data.data_loader import load_data
    from src.training.hyperparameter_tuning import HyperparameterTuner
    
    logger.info("Starting hyperparameter tuning")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Tuning method: {args.method}")
    
    # Load data
    train_dataloader, val_dataloader, test_dataloader = load_data(
        symbol=args.symbol,
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size
    )
    
    # Parse parameter grid from JSON if provided
    param_grid = None
    if args.param_grid:
        try:
            param_grid = json.loads(args.param_grid)
            logger.info(f"Parameter grid: {param_grid}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON for parameter grid: {args.param_grid}")
    
    # Create tuner
    tuner = HyperparameterTuner(
        param_grid=param_grid,
        model_type=args.model_type,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    
    # Choose tuning method
    if args.method == "bayesian":
        tuning_results = tuner.bayesian_search(
            n_trials=args.n_trials,
            max_epochs=args.max_epochs,
            early_stopping_patience=args.early_stopping_patience
        )
    elif args.method == "random":
        tuning_results = tuner.random_search(
            n_trials=args.n_trials,
            max_epochs=args.max_epochs,
            early_stopping_patience=args.early_stopping_patience
        )
    else:
        tuning_results = tuner.grid_search(
            max_epochs=args.max_epochs,
            early_stopping_patience=args.early_stopping_patience
        )
    
    logger.info("Hyperparameter tuning completed")
    logger.info(f"Best parameters: {tuning_results.get('best_params')}")
    logger.info(f"Best metric: {tuning_results.get('best_metric')}")
    
    return tuning_results


def serve_command(args):
    """
    Start the model serving API.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting model serving API")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    
    # Set environment variables for the API
    os.environ["ML_API_PORT"] = str(args.port)
    os.environ["MODEL_REGISTRY_PATH"] = args.registry_path
    
    # Import here to avoid circular imports
    import uvicorn
    from src.api.app import create_app
    
    # Create the FastAPI application
    app = create_app()
    
    # Start the server
    uvicorn.run(app, host=args.host, port=args.port)


def evaluate_command(args):
    """
    Evaluate a trained model.
    
    Args:
        args: Command line arguments
    """
    from src.models.model_registry import ModelRegistry
    from src.data.data_loader import load_data
    from src.training.evaluation import evaluate_model
    
    logger.info("Starting model evaluation")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Model version: {args.version or 'latest'}")
    
    # Load the model
    registry = ModelRegistry()
    model = registry.load_model(
        symbol=args.symbol,
        version=args.version
    )
    
    # Load test data
    _, _, test_dataloader = load_data(
        symbol=args.symbol,
        data_path=args.data_path,
        train_ratio=0,
        val_ratio=0,
        test_ratio=1.0,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size
    )
    
    # Evaluate the model
    evaluation_results = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        return_predictions=True
    )
    
    # Print metrics
    logger.info("Evaluation metrics:")
    for name, value in evaluation_results['metrics'].items():
        logger.info(f"  {name}: {value:.4f}")
    
    # Save predictions if output path is provided
    if args.output_path:
        predictions_df = pd.DataFrame({
            'actual': evaluation_results['y_true'].flatten(),
            'predicted': evaluation_results['y_pred'].flatten()
        })
        
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {args.output_path}")
    
    return evaluation_results


def predict_command(args):
    """
    Make predictions using a trained model.
    
    Args:
        args: Command line arguments
    """
    from src.models.model_registry import ModelRegistry
    from src.data.preprocessor import EnhancedPreprocessor
    import torch
    
    logger.info("Starting prediction")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Model version: {args.version or 'latest'}")
    
    # Load the model and preprocessor
    registry = ModelRegistry()
    model, metadata = registry.load_model(
        symbol=args.symbol,
        version=args.version,
        return_metadata=True
    )
    
    # Load input data
    input_data = None
    if args.input_path.endswith('.csv'):
        input_data = pd.read_csv(args.input_path)
    elif args.input_path.endswith('.json'):
        with open(args.input_path, 'r') as f:
            input_data = json.load(f)
            
            # Convert to DataFrame if it's a dict
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
    else:
        logger.error(f"Unsupported input file format: {args.input_path}")
        return
    
    # Preprocess input data
    preprocessor = metadata.get('preprocessor')
    if preprocessor:
        preprocessor = EnhancedPreprocessor.load(preprocessor)
        input_features = preprocessor.transform(input_data)
    else:
        # Fallback to simple normalization
        input_features = input_data.values
    
    # Prepare input tensor
    if len(input_features.shape) == 2:
        # Create sequence if needed
        seq_len = getattr(model, 'seq_len', args.sequence_length)
        
        # Simple approach: use the same features for each timestep (for demonstration)
        input_tensor = torch.tensor(
            np.tile(input_features[-1], (seq_len, 1)).reshape(1, seq_len, -1),
            dtype=torch.float32
        )
    else:
        # Already a sequence
        input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor).cpu().numpy()[0]
    
    # Postprocess predictions
    if preprocessor and hasattr(preprocessor, 'inverse_transform'):
        predictions = preprocessor.inverse_transform(predictions)
    
    # Create output
    timestamp = datetime.now().isoformat()
    output = {
        'symbol': args.symbol,
        'model_version': metadata.get('version', args.version or 'latest'),
        'timestamp': timestamp,
        'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
        'input_data': input_data.to_dict('records') if isinstance(input_data, pd.DataFrame) else input_data
    }
    
    # Save predictions if output path is provided
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Predictions saved to {args.output_path}")
    else:
        # Print predictions
        logger.info(f"Predictions: {predictions}")
    
    return output


def monitor_command(args):
    """
    Generate performance monitoring report.
    
    Args:
        args: Command line arguments
    """
    from src.monitoring.performance_tracker import get_tracker
    
    logger.info("Generating performance report")
    logger.info(f"Symbol: {args.symbol}")
    
    # Get the performance tracker
    tracker = get_tracker(args.symbol)
    
    # Generate the report
    report = tracker.generate_performance_report(
        model_version=args.version,
        output_dir=args.output_dir,
        last_n_days=args.last_n_days
    )
    
    # Print the report summary
    logger.info(f"Performance report generated for {args.symbol}")
    if 'avg_rmse' in report:
        logger.info(f"Average RMSE: {report['avg_rmse']:.4f}")
    if 'avg_directional_accuracy' in report:
        logger.info(f"Average Directional Accuracy: {report['avg_directional_accuracy']:.4f}")
    
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report summary saved to {args.output_path}")
    
    return report


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="SMOOPs ML System")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., BTC/USDT)")
    train_parser.add_argument("--model-type", default="lstm", choices=["lstm", "gru", "transformer", "cnn_lstm"], help="Model type")
    train_parser.add_argument("--data-path", default="data/processed", help="Path to data directory")
    train_parser.add_argument("--train-ratio", type=float, default=0.7, help="Training data ratio")
    train_parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation data ratio")
    train_parser.add_argument("--test-ratio", type=float, default=0.15, help="Test data ratio")
    train_parser.add_argument("--sequence-length", type=int, default=60, help="Input sequence length")
    train_parser.add_argument("--forecast-horizon", type=int, default=5, help="Forecast horizon")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    train_parser.add_argument("--training-args", help="Additional training arguments as JSON")
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Tune model hyperparameters")
    tune_parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., BTC/USDT)")
    tune_parser.add_argument("--model-type", default="lstm", choices=["lstm", "gru", "transformer", "cnn_lstm"], help="Model type")
    tune_parser.add_argument("--data-path", default="data/processed", help="Path to data directory")
    tune_parser.add_argument("--method", default="grid", choices=["grid", "random", "bayesian"], help="Tuning method")
    tune_parser.add_argument("--param-grid", help="Parameter grid as JSON")
    tune_parser.add_argument("--n-trials", type=int, default=20, help="Number of trials for random/bayesian search")
    tune_parser.add_argument("--train-ratio", type=float, default=0.7, help="Training data ratio")
    tune_parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation data ratio")
    tune_parser.add_argument("--test-ratio", type=float, default=0.15, help="Test data ratio")
    tune_parser.add_argument("--sequence-length", type=int, default=60, help="Input sequence length")
    tune_parser.add_argument("--forecast-horizon", type=int, default=5, help="Forecast horizon")
    tune_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    tune_parser.add_argument("--max-epochs", type=int, default=50, help="Maximum number of epochs")
    tune_parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start model serving API")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    serve_parser.add_argument("--registry-path", default="models/registry", help="Path to model registry")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    evaluate_parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., BTC/USDT)")
    evaluate_parser.add_argument("--version", help="Model version (default: latest)")
    evaluate_parser.add_argument("--data-path", default="data/processed", help="Path to data directory")
    evaluate_parser.add_argument("--sequence-length", type=int, default=60, help="Input sequence length")
    evaluate_parser.add_argument("--forecast-horizon", type=int, default=5, help="Forecast horizon")
    evaluate_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    evaluate_parser.add_argument("--output-path", help="Path to save predictions")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions using a trained model")
    predict_parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., BTC/USDT)")
    predict_parser.add_argument("--version", help="Model version (default: latest)")
    predict_parser.add_argument("--input-path", required=True, help="Path to input data (CSV or JSON)")
    predict_parser.add_argument("--output-path", help="Path to save predictions")
    predict_parser.add_argument("--sequence-length", type=int, default=60, help="Input sequence length")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Generate performance monitoring report")
    monitor_parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., BTC/USDT)")
    monitor_parser.add_argument("--version", help="Model version (default: all)")
    monitor_parser.add_argument("--output-dir", default="reports", help="Directory to save report charts")
    monitor_parser.add_argument("--output-path", help="Path to save report summary JSON")
    monitor_parser.add_argument("--last-n-days", type=int, help="Include only the last N days")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create the project directory structure
    for directory in ["models/registry", "logs/tensorboard", "data/processed", "reports"]:
        os.makedirs(directory, exist_ok=True)
    
    # Run appropriate command
    try:
        if args.command == "train":
            train_command(args)
        elif args.command == "tune":
            tune_command(args)
        elif args.command == "serve":
            serve_command(args)
        elif args.command == "evaluate":
            evaluate_command(args)
        elif args.command == "predict":
            predict_command(args)
        elif args.command == "monitor":
            monitor_command(args)
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Error running command: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main() 