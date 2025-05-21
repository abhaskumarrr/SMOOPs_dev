"""
CNN-LSTM Hybrid Model Architecture for cryptocurrency trading with Smart Money Concepts features
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, 
    Conv1D, MaxPooling1D, Bidirectional, Concatenate, 
    LayerNormalization, GlobalAveragePooling1D, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import numpy as np
import os
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class CNNLSTMModel(BaseModel):
    """
    CNN-LSTM hybrid model for time-series prediction of cryptocurrency markets
    with Smart Money Concepts features.
    
    This model combines CNN layers for feature extraction with LSTM layers
    for temporal modeling. The CNN effectively captures local patterns while
    the LSTM handles temporal dependencies.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_units: int = 1,
        cnn_filters: List[int] = [64, 128, 128],
        cnn_kernel_sizes: List[int] = [3, 3, 3],
        pool_sizes: List[int] = [2, 2, 2],
        lstm_units: List[int] = [128, 64],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        bidirectional: bool = True,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the CNN-LSTM model.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            output_units: Number of output units (1 for binary classification)
            cnn_filters: Number of filters in each CNN layer
            cnn_kernel_sizes: Kernel sizes for each CNN layer
            pool_sizes: Pooling sizes for each CNN layer
            lstm_units: Number of units in LSTM layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            bidirectional: Whether to use bidirectional LSTM
            model_path: Path to load a pre-trained model (if exists)
        """
        super().__init__(model_path)
        
        self.input_shape = input_shape
        self.output_units = output_units
        self.cnn_filters = cnn_filters
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.pool_sizes = pool_sizes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        
        self.model = self._build_model()
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                logger.info(f"Loaded weights from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load weights: {e}")
    
    def _build_model(self) -> Model:
        """
        Build and compile the CNN-LSTM model.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # CNN layers for feature extraction
        x = inputs
        
        # Apply multiple convolutional layers with max pooling
        for i in range(len(self.cnn_filters)):
            # Convolutional layer
            x = Conv1D(
                filters=self.cnn_filters[i],
                kernel_size=self.cnn_kernel_sizes[i],
                padding='same',
                activation='relu'
            )(x)
            
            # Batch normalization for faster and more stable training
            x = BatchNormalization()(x)
            
            # Max pooling to reduce dimension and extract important features
            if i < len(self.pool_sizes):
                x = MaxPooling1D(pool_size=self.pool_sizes[i])(x)
                
            # Add dropout for regularization
            x = Dropout(self.dropout_rate)(x)
        
        # LSTM layers for temporal modeling
        for i, units in enumerate(self.lstm_units):
            # Determine if we should return sequences
            return_sequences = i < len(self.lstm_units) - 1
            
            # Apply bidirectional LSTM if specified
            if self.bidirectional:
                x = Bidirectional(
                    LSTM(
                        units=units,
                        return_sequences=return_sequences,
                        recurrent_dropout=0.1
                    )
                )(x)
            else:
                x = LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    recurrent_dropout=0.1
                )(x)
            
            # Layer normalization for stable training
            x = LayerNormalization()(x)
            
            # Dropout for regularization
            x = Dropout(self.dropout_rate)(x)
        
        # Dense layers for final prediction
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Final prediction layer
        if self.output_units == 1:
            outputs = Dense(1, activation='sigmoid')(x)  # Binary classification
        else:
            outputs = Dense(self.output_units, activation='softmax')(x)  # Multi-class
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with appropriate loss function and optimizer
        if self.output_units == 1:
            loss = 'binary_crossentropy'
        else:
            loss = 'sparse_categorical_crossentropy'
            
        optimizer = Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        logger.info(f"Built CNN-LSTM model: {model.summary()}")
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 10,
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the CNN-LSTM model.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            patience: Patience for early stopping
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for model checkpoints
            
        Returns:
            Dictionary with training history
        """
        callbacks = []
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpointing
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"cnn_lstm_model_" + "{epoch:02d}_{val_accuracy:.4f}.h5"
            )
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # TensorBoard logging
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            tensorboard = TensorBoard(
                log_dir=os.path.join(log_dir, 'cnn_lstm_logs'),
                histogram_freq=1,
                write_graph=True
            )
            callbacks.append(tensorboard)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Evaluate model
        results = self.model.evaluate(X, y, verbose=0)
        
        # Convert to dictionary
        metrics = {
            'loss': results[0],
            'accuracy': results[1]
        }
        
        # For binary classification, compute precision, recall, and F1
        if self.output_units == 1:
            y_pred = (self.predict(X) > 0.5).astype(int).flatten()
            y_true = y.astype(int).flatten()
            
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            
            # Calculate standard classification metrics
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)
            metrics['f1_score'] = f1_score(y_true, y_pred)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Add trading-specific metrics
            if len(y_true) > 0 and len(y_pred) > 0:
                # Win rate: percentage of correct predictions when model predicts positive
                positive_preds = (y_pred == 1)
                if positive_preds.sum() > 0:
                    win_rate = (y_true[positive_preds] == 1).sum() / positive_preds.sum()
                    metrics['win_rate'] = float(win_rate)
                else:
                    metrics['win_rate'] = 0.0
                
                # Profit factor: ratio of gains to losses
                metrics['profit_factor'] = float(metrics['precision'] / (1.0 - metrics['precision']) if metrics['precision'] < 1.0 else 10.0)
        
        return metrics
    
    def save(self, model_path: str) -> None:
        """
        Save the model.
        
        Args:
            model_path: Path to save the model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_weights(model_path)
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, model_path: str, input_shape: Tuple[int, int], **kwargs) -> 'CNNLSTMModel':
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the saved model
            input_shape: Shape of input data
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded model
        """
        model = cls(input_shape=input_shape, model_path=model_path, **kwargs)
        return model
    
    def visualize_feature_maps(self, X: np.ndarray, layer_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Visualize feature maps from CNN layers.
        
        Args:
            X: Input data
            layer_names: Names of layers to visualize
            
        Returns:
            Dictionary mapping layer names to feature maps
        """
        if layer_names is None:
            # Get all convolutional layer names
            layer_names = [layer.name for layer in self.model.layers if isinstance(layer, Conv1D)]
        
        # Create models to output feature maps
        feature_maps = {}
        for layer_name in layer_names:
            layer = next((l for l in self.model.layers if l.name == layer_name), None)
            if layer is not None:
                intermediate_model = Model(
                    inputs=self.model.input,
                    outputs=layer.output
                )
                # Get feature maps for the first sample
                feature_maps[layer_name] = intermediate_model.predict(X[:1])
                
        return feature_maps
    
    def get_feature_importance(self, X: np.ndarray, n_samples: int = 10) -> Dict[str, float]:
        """
        Get feature importance using permutation importance.
        
        Args:
            X: Input data
            n_samples: Number of samples to use
            
        Returns:
            Dictionary mapping feature indices to importance scores
        """
        if len(X) < n_samples:
            n_samples = len(X)
            
        # Select random samples
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_samples = X[indices]
        
        # Get baseline predictions
        baseline_preds = self.model.predict(X_samples)
        
        # Initialize importance dict
        importance_dict = {}
        
        # For each feature, permute its values and measure impact
        for feature_idx in range(X_samples.shape[2]):
            # Create permuted data
            X_permuted = X_samples.copy()
            
            # Permute the feature across all time steps
            for i in range(X_samples.shape[0]):
                perm_indices = np.random.permutation(X_samples.shape[1])
                X_permuted[i, :, feature_idx] = X_samples[i, perm_indices, feature_idx]
            
            # Get predictions with permuted feature
            permuted_preds = self.model.predict(X_permuted)
            
            # Calculate importance as mean absolute difference
            importance = np.mean(np.abs(baseline_preds - permuted_preds))
            importance_dict[feature_idx] = float(importance)
        
        # Sort by importance (descending)
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data for testing
    sequence_length = 60
    n_features = 30
    n_samples = 1000
    
    X = np.random.random((n_samples, sequence_length, n_features))
    y = np.random.randint(0, 2, size=(n_samples, 1))
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Initialize model
    model = CNNLSTMModel(
        input_shape=(sequence_length, n_features),
        output_units=1,
        cnn_filters=[32, 64, 128],
        cnn_kernel_sizes=[3, 3, 3],
        pool_sizes=[2, 2, 2],
        lstm_units=[64, 32],
        bidirectional=True
    )
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=5,  # Small number for testing
        batch_size=32
    )
    
    # Evaluate model
    evaluation = model.evaluate(X_val, y_val)
    print(f"Evaluation: {evaluation}")
    
    # Get feature importance
    importance = model.get_feature_importance(X_val[:10])
    print(f"Top 5 important features: {list(importance.items())[:5]}") 