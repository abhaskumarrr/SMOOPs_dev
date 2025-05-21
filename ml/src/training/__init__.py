"""
Training Package

This package provides utilities for training and evaluating time series forecasting models
with TensorBoard integration and Apple Silicon optimization.
"""

from .trainer import Trainer
from .data_preparation import (
    TimeSeriesDataset, 
    prepare_time_series_data, 
    prepare_multi_target_data,
    create_walk_forward_cv_splits
)
from .evaluation import (
    calculate_metrics,
    evaluate_model,
    plot_predictions,
    plot_forecast,
    evaluate_forecasts,
    evaluate_walk_forward
)

__all__ = [
    'Trainer',
    'TimeSeriesDataset',
    'prepare_time_series_data',
    'prepare_multi_target_data',
    'create_walk_forward_cv_splits',
    'calculate_metrics',
    'evaluate_model',
    'plot_predictions',
    'plot_forecast',
    'evaluate_forecasts',
    'evaluate_walk_forward'
] 