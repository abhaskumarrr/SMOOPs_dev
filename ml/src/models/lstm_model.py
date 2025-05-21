"""
LSTM Model Architecture for cryptocurrency trading with Smart Money Concepts features
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, 
    Bidirectional, Attention, Concatenate, LayerNormalization
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

class LSTMModel(BaseModel):
    """
    LSTM model for time-series prediction of cryptocurrency markets
    with Smart Money Concepts features
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_units: int = 1,
        lstm_units: Union[List[int], int] = [128, 64],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        bidirectional: bool = True,
        use_attention: bool = True,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the LSTM model.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            output_units: Number of output units (1 for binary classification)
            lstm_units: Number of units in LSTM layers (list or single int)
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
            model_path: Path to load a pre-trained model (if exists)
        """
        super().__init__(model_path)
        
        self.input_shape = input_shape
        self.output_units = output_units
        self.lstm_units = lstm_units if isinstance(lstm_units, list) else [lstm_units]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        self.model = self._build_model()
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                logger.info(f"Loaded weights from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load weights: {e}")
    
    def _build_model(self) -> Model:
        """
        Build and compile the LSTM model.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # LSTM layers with residual connections if multiple layers
        x = inputs
        lstm_outputs = []
        
        for i, units in enumerate(self.lstm_units):
            # LSTM layer (bidirectional or not)
            if self.bidirectional:
                if i < len(self.lstm_units) - 1:
                    # Return sequences for all but the last layer
                    lstm_layer = Bidirectional(
                        LSTM(units, return_sequences=True, recurrent_dropout=0.1)
                    )(x)
                else:
                    # Last layer may return sequences for attention, or not if no attention
                    return_seq = self.use_attention
                    lstm_layer = Bidirectional(
                        LSTM(units, return_sequences=return_seq, recurrent_dropout=0.1)
                    )(x)
            else:
                if i < len(self.lstm_units) - 1:
                    lstm_layer = LSTM(units, return_sequences=True, recurrent_dropout=0.1)(x)
                else:
                    return_seq = self.use_attention
                    lstm_layer = LSTM(units, return_sequences=return_seq, recurrent_dropout=0.1)(x)
            
            # Add batch normalization for faster training
            lstm_normalized = LayerNormalization()(lstm_layer)
            
            # Add dropout for regularization
            lstm_dropout = Dropout(self.dropout_rate)(lstm_normalized)
            
            # Save output for potential residual connections
            lstm_outputs.append(lstm_dropout)
            
            # Update x for next layer
            x = lstm_dropout
        
        # Apply attention if requested
        if self.use_attention:
            # Self-attention on the output of the last LSTM layer
            # This helps focus on the most important parts of the sequence
            attention_layer = tf.keras.layers.MultiHeadAttention(
                key_dim=64, num_heads=2, dropout=0.1
            )(x, x)
            attention_add = tf.keras.layers.Add()([attention_layer, x])
            x = LayerNormalization()(attention_add)
            
            # Global pooling to reduce sequence dimension
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
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
        
        logger.info(f"Built LSTM model: {model.summary()}")
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
        Train the LSTM model.
        
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
                f"lstm_model_" + "{epoch:02d}_{val_accuracy:.4f}.h5"
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
                log_dir=os.path.join(log_dir, 'lstm_logs'),
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
            
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)
            metrics['f1_score'] = f1_score(y_true, y_pred)
        
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
    def load(cls, model_path: str, input_shape: Tuple[int, int], **kwargs) -> 'LSTMModel':
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
    
    def get_feature_importance(self, X: np.ndarray, sequence_index: int = -1) -> Dict[str, float]:
        """
        Get feature importance using integrated gradients.
        
        Args:
            X: Input data (batch of sequences)
            sequence_index: Index of the sequence in the batch to explain
            
        Returns:
            Dictionary mapping feature indices to importance scores
        """
        if sequence_index >= len(X):
            raise ValueError(f"sequence_index {sequence_index} out of bounds for X with length {len(X)}")
        
        X_sample = X[sequence_index:sequence_index+1]
        
        # Create a baseline (zeros)
        baseline = np.zeros_like(X_sample)
        
        # Create a GradientTape to compute gradients
        with tf.GradientTape() as tape:
            tape.watch(X_sample)
            predictions = self.model(X_sample)
        
        # Get gradients of the output with respect to the input
        gradients = tape.gradient(predictions, X_sample)
        
        # Average gradients along the sequence axis (axis=1)
        feature_importance = np.mean(np.abs(gradients.numpy()), axis=1)[0]
        
        # Create a dictionary mapping feature indices to importance
        importance_dict = {i: float(importance) for i, importance in enumerate(feature_importance)}
        
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
    model = LSTMModel(
        input_shape=(sequence_length, n_features),
        output_units=1,
        lstm_units=[64, 32],
        bidirectional=True,
        use_attention=True
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
    importance = model.get_feature_importance(X_val[:5])
    print(f"Top 5 important features: {list(importance.items())[:5]}") 