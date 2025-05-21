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
    trading_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Calculate regression metrics for model evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        prefix: Prefix for metric names (e.g., 'test_', 'val_')
        trading_info: Additional trading data for calculating trading-specific metrics
        
    Returns:
        Dictionary of metrics
    """
    # Ensure arrays are flattened
    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    
    # Calculate standard metrics
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
    
    # Initialize metrics dictionary with standard metrics
    metrics = {
        f"{prefix}mse": mse,
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}r2": r2,
        f"{prefix}mape": mape,
        f"{prefix}directional_accuracy": directional_accuracy,
    }
    
    # Calculate trading-specific metrics if trading info is provided
    if trading_info is not None:
        # Calculate Sharpe ratio
        if 'returns' in trading_info:
            returns = trading_info['returns']
            # Calculate Sharpe ratio (risk-adjusted return)
            sharpe_ratio = 0.0
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            metrics[f"{prefix}sharpe_ratio"] = sharpe_ratio
        
        # Calculate maximum drawdown
        if 'equity_curve' in trading_info:
            equity = trading_info['equity_curve']
            # Calculate maximum drawdown
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_drawdown = np.max(drawdown)
            metrics[f"{prefix}max_drawdown"] = max_drawdown
        
        # Calculate profit factor
        if 'profits' in trading_info and 'losses' in trading_info:
            profits = trading_info['profits']
            losses = trading_info['losses']
            # Calculate profit factor (gross profit / gross loss)
            profit_factor = 0.0
            if np.sum(np.abs(losses)) > 0:
                profit_factor = np.sum(profits) / np.sum(np.abs(losses))
            metrics[f"{prefix}profit_factor"] = profit_factor
        
        # Calculate win rate
        if 'trades' in trading_info:
            trades = trading_info['trades']
            # Calculate win rate (percentage of profitable trades)
            win_rate = np.mean([t > 0 for t in trades]) if trades else 0.0
            metrics[f"{prefix}win_rate"] = win_rate
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    return_predictions: bool = False,
    trading_simulation: bool = False,
    fee_rate: float = 0.001,  # 0.1% trading fee
) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to use for evaluation (if None, use model's device)
        return_predictions: Whether to return predictions and true values
        trading_simulation: Whether to simulate trading based on predictions
        fee_rate: Trading fee rate for simulations (e.g., 0.001 for 0.1%)
        
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
    
    # Initialize trading info
    trading_info = None
    
    # Simulate trading if requested
    if trading_simulation:
        trading_info = simulate_trading(y_true, y_pred, fee_rate=fee_rate)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, trading_info=trading_info)
    
    # Log metrics
    logger.info(f"Evaluation metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.6f}")
    
    # Return results
    results = {"metrics": metrics}
    if return_predictions:
        results["y_pred"] = y_pred
        results["y_true"] = y_true
    
    if trading_simulation and trading_info is not None:
        results["trading_simulation"] = trading_info
    
    return results


def simulate_trading(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
    position_size: float = 1.0,
) -> Dict[str, Any]:
    """
    Simulate trading based on model predictions.
    
    Args:
        y_true: True price values
        y_pred: Predicted price values
        initial_capital: Initial capital for the simulation
        fee_rate: Trading fee rate (e.g., 0.001 for 0.1%)
        position_size: Position size as a fraction of capital (0.0-1.0)
        
    Returns:
        Dictionary with trading simulation results
    """
    # Ensure arrays are flattened
    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    
    # Initialize simulation variables
    capital = initial_capital
    position = 0  # 0: no position, 1: long, -1: short
    equity_curve = [capital]
    trades = []
    profits = []
    losses = []
    
    # Generate trading signals (price direction predictions)
    signals = np.zeros(len(y_pred) - 1)
    for i in range(len(signals)):
        pred_direction = y_pred[i+1] > y_pred[i]
        signals[i] = 1 if pred_direction else -1
    
    # Simulate trading based on signals
    for i in range(len(signals)):
        price = y_true[i]
        next_price = y_true[i+1]
        signal = signals[i]
        
        # Close existing position if signal changes direction
        if position != 0 and position != signal:
            # Calculate P&L from position
            price_diff = (next_price - price) if position == 1 else (price - next_price)
            position_value = capital * position_size
            trade_pnl = position_value * (price_diff / price)
            
            # Apply trading fees
            fees = position_value * fee_rate * 2  # Entry and exit
            net_pnl = trade_pnl - fees
            
            # Update capital
            capital += net_pnl
            
            # Record trade
            trades.append(net_pnl)
            if net_pnl > 0:
                profits.append(net_pnl)
            else:
                losses.append(net_pnl)
            
            # Close position
            position = 0
        
        # Open new position if no current position
        if position == 0:
            position = signal
        
        # Update equity curve
        equity_curve.append(capital)
    
    # Calculate returns
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # Return trading simulation results
    return {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': (capital / initial_capital) - 1,
        'equity_curve': equity_curve,
        'trades': trades,
        'num_trades': len(trades),
        'profits': profits,
        'losses': losses,
        'returns': returns,
    }


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
    
    # Simulate trading for the plotted range
    trading_info = simulate_trading(y_true_plot, y_pred_plot)
    
    # Create metrics text with traditional and trading metrics
    metric_text = (
        f"RMSE: {metrics['rmse']:.4f}\n"
        f"MAE: {metrics['mae']:.4f}\n"
        f"MAPE: {metrics['mape']:.2f}%\n"
        f"Directional Accuracy: {metrics['directional_accuracy']:.2f}\n"
        f"Total Return: {trading_info['total_return']:.2%}\n"
        f"# Trades: {trading_info['num_trades']}"
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


def plot_trading_simulation(
    trading_results: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "Trading Simulation Results",
) -> plt.Figure:
    """
    Plot trading simulation results.
    
    Args:
        trading_results: Dictionary with trading simulation results
        save_path: Path to save the plot (if None, plot is not saved)
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Get equity curve and trades
    equity_curve = trading_results['equity_curve']
    trades = trading_results['trades']
    
    # Plot equity curve
    x_values = range(len(equity_curve))
    ax1.plot(x_values, equity_curve, 'b-', label='Equity Curve')
    
    # Add horizontal line at initial capital
    initial_capital = trading_results['initial_capital']
    ax1.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital')
    
    # Add labels and legend for equity curve
    ax1.set_title(title)
    ax1.set_ylabel('Equity ($)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot trade results as bar chart
    if trades:
        trade_indices = range(len(trades))
        colors = ['g' if t > 0 else 'r' for t in trades]
        ax2.bar(trade_indices, trades, color=colors)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Profit/Loss ($)')
        ax2.grid(True, alpha=0.3)
    
    # Add text with performance metrics
    metrics_text = (
        f"Initial Capital: ${trading_results['initial_capital']:.2f}\n"
        f"Final Capital: ${trading_results['final_capital']:.2f}\n"
        f"Total Return: {trading_results['total_return']:.2%}\n"
        f"Number of Trades: {trading_results['num_trades']}\n"
        f"Win Rate: {len(trading_results['profits']) / trading_results['num_trades']:.2%} if trading_results['num_trades'] > 0 else 0.0\n"
        f"Profit Factor: {sum(trading_results['profits']) / abs(sum(trading_results['losses'])):.2f} if trading_results['losses'] and sum(trading_results['losses']) != 0 else 'N/A'"
    )
    
    ax1.text(0.02, 0.05, metrics_text, transform=ax1.transAxes, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Trading simulation plot saved to {save_path}")
    
    return fig


def evaluate_forecasts(
    model: torch.nn.Module,
    dataloader: DataLoader,
    n_forecasts: int = 5,
    output_dir: Optional[str] = None,
    trading_simulation: bool = False,
) -> Dict[str, Any]:
    """
    Generate and evaluate multiple forecasts.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the evaluation data
        n_forecasts: Number of forecasts to generate and evaluate
        output_dir: Directory to save forecast plots (if None, plots are not saved)
        trading_simulation: Whether to simulate trading based on predictions
        
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
    
    with torch.no_grad():
        while samples_evaluated < n_forecasts:
            try:
                # Get next batch
                X_batch, y_batch = next(data_iter)
                
                # Just use the first item in the batch
                X = X_batch[0:1].to(device)
                y_true = y_batch[0:1].numpy()
                
                # Generate prediction
                y_pred = model(X).cpu().numpy()
                
                # Initialize trading info
                trading_info = None
                
                # Simulate trading if requested
                if trading_simulation:
                    trading_info = simulate_trading(y_true.flatten(), y_pred.flatten())
                
                # Calculate metrics
                metrics = calculate_metrics(y_true, y_pred, trading_info=trading_info)
                
                # Save forecast plots
                if output_dir is not None:
                    # Plot predictions
                    plot_path = os.path.join(output_dir, f"forecast_{samples_evaluated+1}.png")
                    plot_predictions(
                        y_true=y_true.flatten(),
                        y_pred=y_pred.flatten(),
                        save_path=plot_path,
                        title=f"Forecast {samples_evaluated+1}"
                    )
                    
                    # Plot trading simulation if performed
                    if trading_simulation and trading_info is not None:
                        trading_plot_path = os.path.join(output_dir, f"trading_sim_{samples_evaluated+1}.png")
                        plot_trading_simulation(
                            trading_results=trading_info,
                            save_path=trading_plot_path,
                            title=f"Trading Simulation - Forecast {samples_evaluated+1}"
                        )
                
                # Store forecast data
                forecast_data = {
                    "id": samples_evaluated + 1,
                    "metrics": metrics,
                    "y_true": y_true.tolist(),
                    "y_pred": y_pred.tolist()
                }
                
                if trading_simulation and trading_info is not None:
                    # Add trading simulation results to forecast data
                    forecast_data["trading_simulation"] = {
                        "total_return": trading_info["total_return"],
                        "num_trades": trading_info["num_trades"],
                        "win_rate": len(trading_info["profits"]) / trading_info["num_trades"] if trading_info["num_trades"] > 0 else 0.0,
                    }
                
                forecasts.append(forecast_data)
                samples_evaluated += 1
                
            except StopIteration:
                # If we run out of data, break the loop
                break
    
    # Calculate average metrics across all forecasts
    avg_metrics = {}
    metric_keys = list(forecasts[0]["metrics"].keys()) if forecasts else []
    
    for key in metric_keys:
        avg_metrics[f"avg_{key}"] = np.mean([f["metrics"][key] for f in forecasts])
    
    # Log average metrics
    logger.info(f"Average metrics across {len(forecasts)} forecasts:")
    for name, value in avg_metrics.items():
        logger.info(f"  {name}: {value:.6f}")
    
    return {
        "metrics": avg_metrics,
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
    trading_simulation: bool = False,
    output_dir: Optional[str] = None,
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
        trading_simulation: Whether to simulate trading based on predictions
        output_dir: Directory to save evaluation plots
        
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
    all_trading_results = []
    
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
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
        trading_sim = trading_simulation
        eval_results = evaluate_model(model, test_loader, return_predictions=True, trading_simulation=trading_sim)
        
        # Add split index to metrics
        split_metrics = eval_results["metrics"]
        split_metrics["split"] = i
        all_metrics.append(split_metrics)
        
        # Save trading results if simulation was performed
        if trading_sim and "trading_simulation" in eval_results:
            trading_results = eval_results["trading_simulation"]
            trading_results["split"] = i
            all_trading_results.append(trading_results)
            
            # Plot trading simulation results if output directory is provided
            if output_dir is not None:
                trading_plot_path = os.path.join(output_dir, f"trading_sim_split_{i+1}.png")
                plot_trading_simulation(
                    trading_results=trading_results,
                    save_path=trading_plot_path,
                    title=f"Trading Simulation - Split {i+1}/{n_splits}"
                )
        
        # Plot predictions if output directory is provided
        if output_dir is not None and "y_true" in eval_results and "y_pred" in eval_results:
            plot_path = os.path.join(output_dir, f"predictions_split_{i+1}.png")
            plot_predictions(
                y_true=eval_results["y_true"],
                y_pred=eval_results["y_pred"],
                save_path=plot_path,
                title=f"Predictions - Split {i+1}/{n_splits}",
                n_samples=min(100, len(eval_results["y_true"]))
            )
        
        # Log metrics
        logger.info(f"Split {i+1} metrics:")
        for name, value in split_metrics.items():
            if name != "split" and isinstance(value, (int, float)):
                logger.info(f"  {name}: {value:.6f}")
    
    # Calculate average metrics across all splits
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        if metric != "split" and isinstance(all_metrics[0][metric], (int, float)):
            avg_metrics[f"avg_{metric}"] = np.mean([m[metric] for m in all_metrics])
    
    # Log average metrics
    logger.info(f"Average metrics across {n_splits} splits:")
    for name, value in avg_metrics.items():
        logger.info(f"  {name}: {value:.6f}")
    
    results = {
        "metrics_by_split": all_metrics,
        "avg_metrics": avg_metrics
    }
    
    # Add trading results if simulations were performed
    if all_trading_results:
        results["trading_results_by_split"] = all_trading_results
        
        # Calculate average trading metrics
        avg_trading_metrics = {}
        for key in ["total_return", "num_trades"]:
            if all([key in tr for tr in all_trading_results]):
                avg_trading_metrics[f"avg_{key}"] = np.mean([tr[key] for tr in all_trading_results])
        
        results["avg_trading_metrics"] = avg_trading_metrics
    
    return results


def calculate_trading_metrics(
    predictions: np.ndarray,
    actual_prices: np.ndarray,
    initial_capital: float = 10000.0,
    position_size: float = 1.0,
    fee_rate: float = 0.001,
) -> Dict[str, float]:
    """
    Calculate trading-specific metrics based on model predictions.
    
    Args:
        predictions: Model's price predictions
        actual_prices: Actual prices observed
        initial_capital: Initial capital for simulation
        position_size: Position size as fraction of capital (0.0-1.0)
        fee_rate: Trading fee rate (e.g., 0.001 for 0.1%)
        
    Returns:
        Dictionary of trading metrics
    """
    # Run trading simulation
    sim_results = simulate_trading(
        y_true=actual_prices,
        y_pred=predictions,
        initial_capital=initial_capital,
        position_size=position_size,
        fee_rate=fee_rate
    )
    
    # Extract key metrics
    metrics = {
        "total_return": sim_results["total_return"],
        "num_trades": sim_results["num_trades"],
    }
    
    # Calculate additional metrics if we have trades
    if sim_results["num_trades"] > 0:
        metrics["win_rate"] = len(sim_results["profits"]) / sim_results["num_trades"]
        
        # Calculate profit factor if we have losses
        if sim_results["losses"] and sum(abs(np.array(sim_results["losses"]))) > 0:
            metrics["profit_factor"] = sum(sim_results["profits"]) / sum(abs(np.array(sim_results["losses"])))
        else:
            metrics["profit_factor"] = float('inf')  # No losses
        
        # Calculate maximum drawdown
        equity_curve = np.array(sim_results["equity_curve"])
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        metrics["max_drawdown"] = np.max(drawdown)
        
        # Calculate Sharpe ratio (annualized)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            metrics["sharpe_ratio"] = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            metrics["sharpe_ratio"] = 0.0
    else:
        # No trades
        metrics["win_rate"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["max_drawdown"] = 0.0
        metrics["sharpe_ratio"] = 0.0
    
    return metrics 