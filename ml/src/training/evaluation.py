"""
Model Evaluation Module

This module provides utilities for evaluating trained models and calculating
various performance metrics for time series forecasting.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import os
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Calculate regression metrics for model evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        prefix: Prefix for metric names (e.g., 'test_', 'val_')
        
    Returns:
        Dictionary of metrics
    """
    # Ensure arrays are flattened
    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero by adding small epsilon where true value is zero
    epsilon = 1e-10
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    # Calculate directional accuracy
    direction_true = np.diff(y_true, axis=0) > 0
    direction_pred = np.diff(y_pred, axis=0) > 0
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    # Return metrics with optional prefix
    return {
        f"{prefix}mse": mse,
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}r2": r2,
        f"{prefix}mape": mape,
        f"{prefix}directional_accuracy": directional_accuracy,
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to use for evaluation (if None, use model's device)
        return_predictions: Whether to return predictions and true values
        
    Returns:
        Dictionary containing evaluation metrics and optionally predictions
    """
    # Set model device
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # Move data to the appropriate device
            X_batch = X_batch.to(device)
            
            # Forward pass
            predictions = model(X_batch)
            
            # Move predictions and targets to CPU for numpy conversion
            predictions = predictions.cpu().numpy()
            targets = y_batch.numpy()
            
            # Store predictions and targets
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    # Concatenate batches
    y_pred = np.vstack(all_predictions)
    y_true = np.vstack(all_targets)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Log metrics
    logger.info(f"Evaluation metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.6f}")
    
    # Return results
    results = {"metrics": metrics}
    if return_predictions:
        results["y_pred"] = y_pred
        results["y_true"] = y_true
    
    return results


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    start_idx: int = 0,
    n_samples: int = 100,
    save_path: Optional[str] = None,
    title: str = "Model Predictions vs Actual Values",
) -> plt.Figure:
    """
    Plot model predictions against true values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        start_idx: Starting index for the plot
        n_samples: Number of samples to plot
        save_path: Path to save the plot (if None, plot is not saved)
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Select data to plot
    end_idx = min(start_idx + n_samples, len(y_true))
    x_values = np.arange(start_idx, end_idx)
    y_true_plot = y_true[start_idx:end_idx]
    y_pred_plot = y_pred[start_idx:end_idx]
    
    # Flatten if needed
    if y_true_plot.ndim > 1 and y_true_plot.shape[1] == 1:
        y_true_plot = y_true_plot.flatten()
    if y_pred_plot.ndim > 1 and y_pred_plot.shape[1] == 1:
        y_pred_plot = y_pred_plot.flatten()
    
    # Plot data
    ax.plot(x_values, y_true_plot, 'b-', label='Actual')
    ax.plot(x_values, y_pred_plot, 'r--', label='Predicted')
    
    # Calculate metrics for the plotted range
    metrics = calculate_metrics(y_true_plot, y_pred_plot)
    metric_text = (
        f"RMSE: {metrics['rmse']:.4f}\n"
        f"MAE: {metrics['mae']:.4f}\n"
        f"MAPE: {metrics['mape']:.2f}%\n"
        f"Directional Accuracy: {metrics['directional_accuracy']:.2f}"
    )
    
    # Add metrics as text
    ax.text(0.02, 0.95, metric_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labels and legend
    ax.set_title(title)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Values')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Save the plot if a path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    return fig


def plot_forecast(
    input_sequence: np.ndarray,
    true_future: np.ndarray,
    predicted_future: np.ndarray,
    feature_idx: int = 0,
    save_path: Optional[str] = None,
    title: str = "Forecast vs Actual Values",
) -> plt.Figure:
    """
    Plot a single forecast from a model.
    
    Args:
        input_sequence: Input sequence used for prediction
        true_future: True future values
        predicted_future: Predicted future values
        feature_idx: Index of the feature to plot
        save_path: Path to save the plot (if None, plot is not saved)
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract the feature to plot
    if input_sequence.ndim > 1 and input_sequence.shape[1] > 1:
        input_sequence_plot = input_sequence[:, feature_idx]
    else:
        input_sequence_plot = input_sequence.flatten()
    
    if true_future.ndim > 1 and true_future.shape[1] > 1:
        true_future_plot = true_future[:, feature_idx]
        predicted_future_plot = predicted_future[:, feature_idx]
    else:
        true_future_plot = true_future.flatten()
        predicted_future_plot = predicted_future.flatten()
    
    # Create time indices
    n_input = len(input_sequence_plot)
    n_future = len(true_future_plot)
    
    input_indices = np.arange(0, n_input)
    future_indices = np.arange(n_input, n_input + n_future)
    
    # Plot data
    ax.plot(input_indices, input_sequence_plot, 'b-', label='Historical Data')
    ax.plot(future_indices, true_future_plot, 'g-', label='True Future')
    ax.plot(future_indices, predicted_future_plot, 'r--', label='Predicted Future')
    
    # Add a vertical line separating input from prediction
    ax.axvline(x=n_input-1, color='k', linestyle='--', alpha=0.5)
    
    # Calculate metrics for the forecast
    metrics = calculate_metrics(true_future_plot, predicted_future_plot)
    metric_text = (
        f"RMSE: {metrics['rmse']:.4f}\n"
        f"MAE: {metrics['mae']:.4f}\n"
        f"MAPE: {metrics['mape']:.2f}%\n"
        f"Directional Accuracy: {metrics['directional_accuracy']:.2f}"
    )
    
    # Add metrics as text
    ax.text(0.02, 0.95, metric_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labels and legend
    ax.set_title(title)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Values')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Save the plot if a path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    return fig


def evaluate_forecasts(
    model: torch.nn.Module,
    dataloader: DataLoader,
    n_forecasts: int = 5,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate and evaluate multiple forecasts.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the evaluation data
        n_forecasts: Number of forecasts to generate and evaluate
        output_dir: Directory to save forecast plots (if None, plots are not saved)
        
    Returns:
        Dictionary containing evaluation metrics and forecast data
    """
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    # List to store forecast data
    forecasts = []
    
    # Get a few samples from the dataloader
    data_iter = iter(dataloader)
    samples_evaluated = 0
    
    # Iterate through the first n_forecasts samples
    while samples_evaluated < n_forecasts:
        try:
            X_batch, y_batch = next(data_iter)
        except StopIteration:
            # Restart iterator if we reach the end of the dataloader
            data_iter = iter(dataloader)
            X_batch, y_batch = next(data_iter)
        
        # For each sample in the batch
        for i in range(min(X_batch.shape[0], n_forecasts - samples_evaluated)):
            # Get a single sample
            X = X_batch[i].unsqueeze(0).to(device)  # Add batch dimension
            y_true = y_batch[i].numpy()
            
            # Generate forecast
            with torch.no_grad():
                y_pred = model(X).cpu().numpy()[0]
            
            # Store forecast data
            forecast_data = {
                "input_sequence": X_batch[i].numpy(),
                "true_future": y_true,
                "predicted_future": y_pred
            }
            forecasts.append(forecast_data)
            
            # Create forecast plot
            if output_dir is not None:
                save_path = os.path.join(output_dir, f"forecast_{samples_evaluated+1}.png")
                plot_forecast(
                    forecast_data["input_sequence"],
                    forecast_data["true_future"],
                    forecast_data["predicted_future"],
                    save_path=save_path,
                    title=f"Forecast {samples_evaluated+1}"
                )
            
            samples_evaluated += 1
            if samples_evaluated >= n_forecasts:
                break
    
    # Calculate aggregate metrics across all forecasts
    all_true = np.vstack([f["true_future"] for f in forecasts])
    all_pred = np.vstack([f["predicted_future"] for f in forecasts])
    metrics = calculate_metrics(all_true, all_pred)
    
    # Log metrics
    logger.info(f"Forecast evaluation metrics (n={n_forecasts}):")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.6f}")
    
    return {
        "metrics": metrics,
        "forecasts": forecasts
    }


def evaluate_walk_forward(
    model_factory_fn,
    data: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    n_splits: int = 5,
    test_size: int = 30,
    seq_len: int = 60,
    forecast_horizon: int = 5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model using walk-forward validation.
    
    Args:
        model_factory_fn: Function that creates and returns a new model instance
        data: Input DataFrame containing time series data
        target_column: Name of the target column
        feature_columns: List of feature column names
        n_splits: Number of splits for walk-forward validation
        test_size: Size of test set in each split
        seq_len: Length of input sequence
        forecast_horizon: Number of steps to forecast
        device: Device to use for evaluation
        
    Returns:
        Dictionary containing evaluation metrics for each split
    """
    from sklearn.model_selection import TimeSeriesSplit
    from .data_preparation import TimeSeriesDataset
    
    logger.info(f"Performing walk-forward validation with {n_splits} splits")
    
    # Create splits
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    # Extract features and target
    X = data[feature_columns].values
    y = data[target_column].values
    
    # Store metrics for each split
    all_metrics = []
    
    # Evaluate each split
    for i, (train_idx, test_idx) in enumerate(tscv.split(data)):
        logger.info(f"Evaluating split {i+1}/{n_splits}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train, seq_len, forecast_horizon)
        test_dataset = TimeSeriesDataset(X_test, y_test, seq_len, forecast_horizon)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create model
        model = model_factory_fn(
            input_dim=len(feature_columns),
            output_dim=1 if len(y.shape) == 1 else y.shape[1],
            seq_len=seq_len,
            forecast_horizon=forecast_horizon,
            device=device
        )
        
        # Train model for a few epochs (simplified training)
        from .trainer import Trainer
        trainer = Trainer(model, train_loader)
        trainer.train(num_epochs=5, early_stopping_patience=2)
        
        # Evaluate model
        eval_results = evaluate_model(model, test_loader, return_predictions=True)
        
        # Add split index to metrics
        split_metrics = eval_results["metrics"]
        split_metrics["split"] = i
        all_metrics.append(split_metrics)
        
        # Log metrics
        logger.info(f"Split {i+1} metrics:")
        for name, value in split_metrics.items():
            if name != "split":
                logger.info(f"  {name}: {value:.6f}")
    
    # Calculate average metrics across all splits
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        if metric != "split":
            avg_metrics[f"avg_{metric}"] = np.mean([m[metric] for m in all_metrics])
    
    # Log average metrics
    logger.info(f"Average metrics across {n_splits} splits:")
    for name, value in avg_metrics.items():
        logger.info(f"  {name}: {value:.6f}")
    
    return {
        "metrics_by_split": all_metrics,
        "avg_metrics": avg_metrics
    } 