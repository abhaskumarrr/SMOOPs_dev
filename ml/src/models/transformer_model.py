"""
Transformer Model Architecture for cryptocurrency trading with Smart Money Concepts features
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, LayerNormalization,
    GlobalAveragePooling1D, Flatten, Concatenate, MultiHeadAttention,
    Conv1D, MaxPooling1D
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

class TransformerBlock(tf.keras.layers.Layer):
    """
    Transformer block with multi-head attention and feed-forward network
    """
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            rate: Dropout rate
        """
        super(TransformerBlock, self).__init__()
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training=False):
        """
        Forward pass through transformer block.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        # Self-attention with residual connection and layer normalization
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network with residual connection and layer normalization
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional encoding layer for transformer models
    """
    
    def __init__(self, max_seq_len: int, embed_dim: int):
        """
        Initialize positional encoding layer.
        
        Args:
            max_seq_len: Maximum sequence length
            embed_dim: Embedding dimension
        """
        super(PositionalEncoding, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Create positional encoding matrix
        pos_enc = np.zeros((max_seq_len, embed_dim))
        positions = np.arange(0, max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        
        # Apply sine to even indices and cosine to odd indices
        pos_enc[:, 0::2] = np.sin(positions * div_term)
        pos_enc[:, 1::2] = np.cos(positions * div_term)
        
        # Add batch dimension
        self.pos_enc = tf.cast(pos_enc[np.newaxis, ...], dtype=tf.float32)
    
    def call(self, inputs):
        """
        Add positional encoding to inputs.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Input tensor with positional encoding added
        """
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_enc[:, :seq_len, :]

class TransformerModel(BaseModel):
    """
    Transformer model for time-series prediction of cryptocurrency markets
    with Smart Money Concepts features
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_units: int = 1,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_transformer_blocks: int = 2,
        mlp_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        use_positional_encoding: bool = True,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the Transformer model.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            output_units: Number of output units (1 for binary classification)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            num_transformer_blocks: Number of transformer blocks
            mlp_units: Units in final MLP layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            use_positional_encoding: Whether to use positional encoding
            model_path: Path to load a pre-trained model (if exists)
        """
        super().__init__(model_path)
        
        self.input_shape = input_shape
        self.output_units = output_units
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_positional_encoding = use_positional_encoding
        
        self.model = self._build_model()
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                logger.info(f"Loaded weights from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load weights: {e}")
    
    def _build_model(self) -> Model:
        """
        Build and compile the Transformer model.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Linear projection to embedding dimension
        x = Conv1D(self.embed_dim, kernel_size=1, padding="same")(inputs)
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = PositionalEncoding(self.input_shape[0], self.embed_dim)(x)
        
        # Transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(
                self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate
            )(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Additional feature extraction with SMOTC-specific features
        # This branch processes time-domain features using Conv1D
        time_features = Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
        time_features = MaxPooling1D(pool_size=2)(time_features)
        time_features = Conv1D(64, kernel_size=3, padding="same", activation="relu")(time_features)
        time_features = GlobalAveragePooling1D()(time_features)
        
        # Concatenate transformer features with time-domain features
        x = Concatenate()([x, time_features])
        
        # Final MLP
        for dim in self.mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        if self.output_units == 1:
            outputs = Dense(1, activation="sigmoid")(x)  # Binary classification
        else:
            outputs = Dense(self.output_units, activation="softmax")(x)  # Multi-class
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        if self.output_units == 1:
            loss = "binary_crossentropy"
        else:
            loss = "sparse_categorical_crossentropy"
            
        optimizer = Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"]
        )
        
        logger.info(f"Built Transformer model: {model.summary()}")
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
        Train the Transformer model.
        
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
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
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
                f"transformer_model_" + "{epoch:02d}_{val_accuracy:.4f}.h5"
            )
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # TensorBoard logging
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            tensorboard = TensorBoard(
                log_dir=os.path.join(log_dir, "transformer_logs"),
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
            "loss": results[0],
            "accuracy": results[1]
        }
        
        # For binary classification, compute precision, recall, and F1
        if self.output_units == 1:
            y_pred = (self.predict(X) > 0.5).astype(int).flatten()
            y_true = y.astype(int).flatten()
            
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            
            metrics["precision"] = precision_score(y_true, y_pred)
            metrics["recall"] = recall_score(y_true, y_pred)
            metrics["f1_score"] = f1_score(y_true, y_pred)
            
            # Trading-specific metrics - profit factor (if applicable)
            metrics["profit_factor"] = self._calculate_profit_factor(y_true, y_pred)
            
            # Calculate Sharpe ratio (if applicable)
            metrics["sharpe_ratio"] = self._calculate_sharpe_ratio(y_true, y_pred)
        
        return metrics
    
    def _calculate_profit_factor(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate profit factor (sum of profits / sum of losses).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Profit factor
        """
        # This is a simplified calculation assuming binary classification
        true_positives = np.logical_and(y_true == 1, y_pred == 1).sum()
        false_positives = np.logical_and(y_true == 0, y_pred == 1).sum()
        
        if false_positives == 0:
            return float('inf') if true_positives > 0 else 1.0
            
        return float(true_positives / false_positives)
    
    def _calculate_sharpe_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Sharpe ratio (return / risk).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Sharpe ratio
        """
        # Simplified calculation assuming binary classification
        # We use accuracy of positive predictions as a proxy for return
        # and standard deviation as a proxy for risk
        positive_preds = (y_pred == 1)
        
        if positive_preds.sum() == 0:
            return 0.0
            
        accuracy = (y_true[positive_preds] == 1).mean()
        std_dev = (y_true[positive_preds] == 1).std()
        
        if std_dev == 0:
            return float('inf') if accuracy > 0 else 0.0
            
        return float(accuracy / std_dev)
    
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
    def load(cls, model_path: str, input_shape: Tuple[int, int], **kwargs) -> 'TransformerModel':
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
    
    def get_attention_weights(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Get attention weights for visualization.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary mapping transformer block index to attention weights
        """
        # This is a simplified implementation
        # In a real application, you would need to create a custom model
        # that outputs attention weights from transformer blocks
        attention_weights = {}
        
        # Create a model that outputs intermediate activations
        intermediate_models = []
        for i in range(self.num_transformer_blocks):
            # Get the ith transformer block's attention weights
            # This requires modifications to the TransformerBlock class
            # to expose attention weights
            layer_name = f"transformer_block_{i}"
            layer = next((l for l in self.model.layers if l.name == layer_name), None)
            
            if layer is not None:
                intermediate_model = Model(
                    inputs=self.model.input,
                    outputs=layer.output  # This would need to be attention weights
                )
                intermediate_models.append(intermediate_model)
        
        # Get attention weights for each block
        for i, model in enumerate(intermediate_models):
            try:
                weights = model.predict(X)
                attention_weights[i] = weights
            except Exception as e:
                logger.error(f"Error getting attention weights for block {i}: {e}")
        
        return attention_weights


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
    model = TransformerModel(
        input_shape=(sequence_length, n_features),
        output_units=1,
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        num_transformer_blocks=2
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
    
    # Make predictions
    predictions = model.predict(X_val[:5])
    print(f"Predictions for first 5 samples: {predictions.flatten()}") 