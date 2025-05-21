"""
GRU Model Architecture for cryptocurrency trading with Smart Money Concepts features
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GRU, Dense, Dropout, BatchNormalization, 
    Bidirectional, Concatenate, LayerNormalization, TimeDistributed
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

class GRUModel(BaseModel):
    """
    GRU model for time-series prediction of cryptocurrency markets
    with Smart Money Concepts features
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_units: int = 1,
        gru_units: Union[List[int], int] = [128, 64],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        bidirectional: bool = True,
        use_residual: bool = True,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the GRU model.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            output_units: Number of output units (1 for binary classification)
            gru_units: Number of units in GRU layers (list or single int)
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            bidirectional: Whether to use bidirectional GRU
            use_residual: Whether to use residual connections
            model_path: Path to load a pre-trained model (if exists)
        """
        super().__init__(model_path)
        
        self.input_shape = input_shape
        self.output_units = output_units
        self.gru_units = gru_units if isinstance(gru_units, list) else [gru_units]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.use_residual = use_residual
        
        self.model = self._build_model()
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                logger.info(f"Loaded weights from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load weights: {e}")
    
    def _build_model(self) -> Model:
        """
        Build and compile the GRU model.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # GRU layers with potential residual connections
        x = inputs
        
        for i, units in enumerate(self.gru_units):
            # Determine whether to return sequences
            return_sequences = (i < len(self.gru_units) - 1)
            
            # GRU layer (bidirectional or not)
            if self.bidirectional:
                gru_layer = Bidirectional(
                    GRU(units, return_sequences=return_sequences, 
                        recurrent_dropout=0.1, implementation=2)
                )(x)
            else:
                gru_layer = GRU(
                    units, return_sequences=return_sequences, 
                    recurrent_dropout=0.1, implementation=2
                )(x)
            
            # Add normalization for faster and more stable training
            gru_norm = LayerNormalization()(gru_layer)
            
            # Add dropout for regularization
            gru_dropout = Dropout(self.dropout_rate)(gru_norm)
            
            # Implement residual connection if requested and dimensions match
            if self.use_residual and i > 0 and return_sequences:
                # Project input to match dimensions if necessary
                if x.shape[-1] != gru_dropout.shape[-1]:
                    projection = TimeDistributed(
                        Dense(gru_dropout.shape[-1])
                    )(x)
                    x = tf.keras.layers.add([projection, gru_dropout])
                else:
                    # Direct residual connection
                    x = tf.keras.layers.add([x, gru_dropout])
            else:
                x = gru_dropout
        
        # Additional dense layers for better feature extraction
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
        
        logger.info(f"Built GRU model: {model.summary()}")
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
        Train the GRU model.
        
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
                f"gru_model_" + "{epoch:02d}_{val_accuracy:.4f}.h5"
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
                log_dir=os.path.join(log_dir, 'gru_logs'),
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
            
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)
            metrics['f1_score'] = f1_score(y_true, y_pred)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Add trading-specific metrics
            if len(y_true) > 0 and len(y_pred) > 0:
                # Calculate win rate (accuracy for positive predictions)
                positive_preds = (y_pred == 1)
                if positive_preds.sum() > 0:
                    win_rate = (y_true[positive_preds] == 1).sum() / positive_preds.sum()
                    metrics['win_rate'] = float(win_rate)
                else:
                    metrics['win_rate'] = 0.0
                
                # Calculate risk-reward ratio (if applicable to the problem)
                metrics['risk_reward'] = float(metrics['precision'] / (1.0 - metrics['precision']) if metrics['precision'] < 1.0 else 10.0)
        
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
    def load(cls, model_path: str, input_shape: Tuple[int, int], **kwargs) -> 'GRUModel':
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
    model = GRUModel(
        input_shape=(sequence_length, n_features),
        output_units=1,
        gru_units=[64, 32],
        bidirectional=True,
        use_residual=True
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