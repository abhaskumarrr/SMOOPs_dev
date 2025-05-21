"""
Model Training Pipeline

This module implements a training pipeline for cryptocurrency price prediction models
with TensorBoard integration for monitoring and Apple Silicon optimization.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path
from datetime import datetime
import json

# Import modules from the project
from ..models import BaseModel, ModelFactory, DirectionalLoss
from ..data.preprocessor import EnhancedPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Model trainer with TensorBoard integration and MPS (Apple Silicon) support"""
    
    def __init__(
        self,
        model: BaseModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer_cls: Callable = optim.Adam,
        optimizer_kwargs: Dict[str, Any] = None,
        loss_fn: nn.Module = None,
        lr_scheduler_cls: Optional[Callable] = None,
        lr_scheduler_kwargs: Dict[str, Any] = None,
        log_dir: Optional[str] = None,
        checkpoints_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        mixed_precision: bool = True,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            optimizer_cls: Optimizer class to use (default: Adam)
            optimizer_kwargs: Arguments for optimizer initialization
            loss_fn: Loss function to use (default: DirectionalLoss)
            lr_scheduler_cls: Optional learning rate scheduler class
            lr_scheduler_kwargs: Arguments for scheduler initialization
            log_dir: Directory for TensorBoard logs
            checkpoints_dir: Directory for model checkpoints
            experiment_name: Name for the experiment
            mixed_precision: Whether to use mixed precision training
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Set up optimizer with default parameters if not provided
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 0.001, "weight_decay": 1e-5}
        self.optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        
        # Set up loss function
        self.loss_fn = loss_fn if loss_fn is not None else DirectionalLoss(alpha=0.7, beta=2.0)
        
        # Set up learning rate scheduler if provided
        self.lr_scheduler = None
        if lr_scheduler_cls is not None:
            if lr_scheduler_kwargs is None:
                lr_scheduler_kwargs = {}
            self.lr_scheduler = lr_scheduler_cls(self.optimizer, **lr_scheduler_kwargs)
        
        # Set up experiment name and directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"{model.__class__.__name__}_{timestamp}"
        
        # Set up TensorBoard writer
        self.log_dir = log_dir or os.path.join("logs", "tensorboard", self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Set up checkpoints directory
        self.checkpoints_dir = checkpoints_dir or os.path.join("models", "checkpoints", self.experiment_name)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # Set up mixed precision training
        self.mixed_precision = mixed_precision
        if torch.cuda.is_available() and mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            if mixed_precision:
                logger.warning("Mixed precision training is enabled but not supported on this device")
                self.mixed_precision = False
        
        # Initialize training metrics
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.training_time = 0
        
        # Log model architecture and hyperparameters
        self._log_model_info()
        
        logger.info(f"Trainer initialized with {model.__class__.__name__} on {model.device}")
        logger.info(f"TensorBoard logs will be saved to {self.log_dir}")
        logger.info(f"Checkpoints will be saved to {self.checkpoints_dir}")
    
    def _log_model_info(self):
        """Log model architecture and hyperparameters to TensorBoard"""
        # Create a text summary of model architecture
        model_summary = str(self.model)
        
        # Log model architecture as text
        self.writer.add_text("Model/Architecture", model_summary, 0)
        
        # Log hyperparameters
        hparams = {
            "model_type": self.model.__class__.__name__,
            "input_dim": self.model.input_dim,
            "output_dim": self.model.output_dim,
            "seq_len": self.model.seq_len,
            "forecast_horizon": self.model.forecast_horizon,
            "optimizer": self.optimizer.__class__.__name__,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "batch_size": self.train_dataloader.batch_size,
            "mixed_precision": self.mixed_precision,
        }
        
        # Add more detailed model hyperparameters if available
        if hasattr(self.model, "hidden_dim"):
            hparams["hidden_dim"] = self.model.hidden_dim
        if hasattr(self.model, "num_layers"):
            hparams["num_layers"] = self.model.num_layers
        if hasattr(self.model, "dropout"):
            hparams["dropout"] = self.model.dropout
        
        # Convert all values to strings for TensorBoard
        hparams = {k: str(v) for k, v in hparams.items()}
        
        # Log hyperparameters as text
        self.writer.add_text("Hyperparameters", json.dumps(hparams, indent=2), 0)
        
    def train_epoch(self) -> float:
        """
        Train the model for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        start_time = time.time()
        
        for batch_idx, (X_batch, y_batch) in enumerate(self.train_dataloader):
            # Move data to the appropriate device
            X_batch = X_batch.to(self.model.device)
            y_batch = y_batch.to(self.model.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(X_batch)
                    loss = self.loss_fn(outputs, y_batch)
                    
                # Backward and optimize with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular forward pass and optimization
                outputs = self.model(X_batch)
                loss = self.loss_fn(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            batch_count += 1
            
            # Log batch loss (every 10 batches)
            if batch_idx % 10 == 0:
                logger.info(f"Epoch: {self.current_epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.6f}")
                self.writer.add_scalar('Loss/train_batch', loss.item(), 
                                     self.current_epoch * len(self.train_dataloader) + batch_idx)
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        epoch_time = time.time() - start_time
        self.training_time += epoch_time
        
        # Log epoch metrics
        self.writer.add_scalar('Loss/train', avg_loss, self.current_epoch)
        self.writer.add_scalar('Time/epoch', epoch_time, self.current_epoch)
        self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        
        # Update learning rate if scheduler exists
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Store loss history
        self.train_losses.append(avg_loss)
        
        logger.info(f"Epoch {self.current_epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.6f}")
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate the model on the validation dataset.
        
        Returns:
            Average validation loss
        """
        if self.val_dataloader is None:
            logger.warning("No validation data provided, skipping validation")
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_dataloader:
                # Move data to the appropriate device
                X_batch = X_batch.to(self.model.device)
                y_batch = y_batch.to(self.model.device)
                
                # Forward pass
                outputs = self.model(X_batch)
                loss = self.loss_fn(outputs, y_batch)
                
                # Update metrics
                total_loss += loss.item()
                batch_count += 1
                
        # Calculate average loss
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        
        # Log validation metrics
        self.writer.add_scalar('Loss/validation', avg_loss, self.current_epoch)
        
        # Calculate and log directional accuracy
        if self.val_dataloader is not None:
            direction_accuracy = self._calculate_directional_accuracy()
            self.writer.add_scalar('Metrics/directional_accuracy', direction_accuracy, self.current_epoch)
            logger.info(f"Validation - Directional Accuracy: {direction_accuracy:.4f}")
        
        # Store validation loss history
        self.val_losses.append(avg_loss)
        
        logger.info(f"Validation Loss: {avg_loss:.6f}")
        return avg_loss
    
    def _calculate_directional_accuracy(self) -> float:
        """
        Calculate directional accuracy on validation set.
        
        Returns:
            Directional accuracy value (0-1)
        """
        self.model.eval()
        direction_correct = 0
        total_directions = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_dataloader:
                # Move data to the appropriate device
                X_batch = X_batch.to(self.model.device)
                y_batch = y_batch.to(self.model.device)
                
                # Forward pass
                outputs = self.model(X_batch)
                
                # Calculate direction (up/down) for predictions and targets
                pred_direction = (outputs[:, 1:] - outputs[:, :-1]) > 0
                true_direction = (y_batch[:, 1:] - y_batch[:, :-1]) > 0
                
                # Compare directions
                direction_correct += torch.sum(pred_direction == true_direction).item()
                total_directions += pred_direction.numel()
        
        # Calculate directional accuracy
        accuracy = direction_correct / total_directions if total_directions > 0 else 0
        return accuracy
    
    def save_checkpoint(self, is_best: bool = False) -> str:
        """
        Save a checkpoint of the current model state.
        
        Args:
            is_best: Whether this checkpoint is the best so far
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoints_dir, f"checkpoint_epoch_{self.current_epoch}.pt")
        
        # Create checkpoint
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'input_dim': self.model.input_dim,
                'output_dim': self.model.output_dim,
                'seq_len': self.model.seq_len,
                'forecast_horizon': self.model.forecast_horizon,
                'model_type': self.model.__class__.__name__
            }
        }
        
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # If this is the best model so far, also save as best.pt
        if is_best:
            best_path = os.path.join(self.checkpoints_dir, "best.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint and restore the training state.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found at {checkpoint_path}")
            return
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore learning rate scheduler if available
        if 'lr_scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        # Update best validation loss if available
        if checkpoint['val_loss'] is not None:
            self.best_val_loss = min(self.best_val_loss, checkpoint['val_loss'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path} (epoch {self.current_epoch})")
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10, save_frequency: int = 5) -> Tuple[List[float], List[float]]:
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            early_stopping_patience: Number of epochs with no improvement before stopping early
            save_frequency: Save checkpoints every n epochs
            
        Returns:
            Training and validation loss history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        start_time = time.time()
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Check if this is the best model so far
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                is_best = True
                logger.info(f"New best validation loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter} epochs (best: {self.best_val_loss:.6f})")
            
            # Save checkpoint
            if (epoch + 1) % save_frequency == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # Calculate total training time
        total_time = time.time() - start_time
        
        # Log final metrics
        self.writer.add_text("Training/Summary", 
                           f"Training completed in {total_time:.2f}s ({num_epochs} epochs)\n"
                           f"Best validation loss: {self.best_val_loss:.6f} (epoch {best_epoch+1})", 0)
        
        logger.info(f"Training completed in {total_time:.2f}s ({num_epochs} epochs)")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f} (epoch {best_epoch+1})")
        
        # Close the TensorBoard writer
        self.writer.close()
        
        return self.train_losses, self.val_losses 