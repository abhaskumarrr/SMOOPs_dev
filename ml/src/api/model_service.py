"""
Model Service API

This module provides API endpoints for serving model predictions.
"""

import os
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

# Import project modules
from ..models import ModelFactory
from ..data.preprocessor import EnhancedPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# Model registry path
MODEL_REGISTRY_PATH = os.environ.get("MODEL_REGISTRY_PATH", "models/registry")


class PredictionInput(BaseModel):
    """Input data for prediction"""
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTC/USDT')")
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
    sequence_length: Optional[int] = Field(60, description="Length of input sequence")
    

class PredictionOutput(BaseModel):
    """Output data for prediction"""
    symbol: str
    predictions: List[float]
    confidence: Optional[float]
    prediction_time: str
    model_version: str


class ModelService:
    """Service for loading and running models"""
    
    def __init__(self):
        self.models = {}
        self.preprocessors = {}
        self.model_info = {}
        
    def load_model(self, symbol: str, model_version: Optional[str] = None) -> bool:
        """
        Load a model for a specific symbol.
        
        Args:
            symbol: Trading symbol
            model_version: Specific model version to load (default: latest)
            
        Returns:
            True if model was loaded successfully
        """
        try:
            # Normalize symbol name for file paths
            symbol_name = symbol.replace("/", "_")
            
            # Determine model path
            if model_version is None:
                # Find latest model version
                model_dir = Path(MODEL_REGISTRY_PATH) / symbol_name
                if not model_dir.exists():
                    logger.error(f"No models found for {symbol}")
                    return False
                
                # Find the latest version
                versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
                if not versions:
                    logger.error(f"No model versions found for {symbol}")
                    return False
                
                model_version = sorted(versions)[-1]
            
            model_path = Path(MODEL_REGISTRY_PATH) / symbol_name / model_version / "best.pt"
            
            if not model_path.exists():
                logger.error(f"Model not found at {model_path}")
                return False
            
            # Load the model
            logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Get model config
            model_config = checkpoint.get('model_config', {})
            if not model_config:
                logger.error("Model config not found in checkpoint")
                return False
            
            # Create model instance
            model_type = model_config.get('model_type', 'lstm').lower()
            model = ModelFactory.create_model(
                model_type=model_type,
                input_dim=model_config.get('input_dim', 10),
                output_dim=model_config.get('output_dim', 1),
                seq_len=model_config.get('seq_len', 60),
                forecast_horizon=model_config.get('forecast_horizon', 5),
                hidden_dim=model_config.get('hidden_dim', 128),
                num_layers=model_config.get('num_layers', 2)
            )
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Load or create preprocessor
            preprocessor_path = Path(MODEL_REGISTRY_PATH) / symbol_name / model_version / "preprocessor.pkl"
            if preprocessor_path.exists():
                import pickle
                with open(preprocessor_path, 'rb') as f:
                    preprocessor = pickle.load(f)
            else:
                logger.warning(f"Preprocessor not found at {preprocessor_path}, creating new one")
                preprocessor = EnhancedPreprocessor()
            
            # Store model and preprocessor
            self.models[symbol] = model
            self.preprocessors[symbol] = preprocessor
            self.model_info[symbol] = {
                'version': model_version,
                'type': model_type,
                'config': model_config
            }
            
            logger.info(f"Model for {symbol} loaded successfully (version: {model_version})")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {str(e)}")
            return False
    
    def predict(self, symbol: str, features: Dict[str, float], sequence_length: int = 60) -> Dict[str, Any]:
        """
        Make predictions using the loaded model.
        
        Args:
            symbol: Trading symbol
            features: Feature values for prediction
            sequence_length: Length of input sequence
            
        Returns:
            Dictionary with prediction results
        """
        # Check if model is loaded
        if symbol not in self.models:
            loaded = self.load_model(symbol)
            if not loaded:
                raise HTTPException(status_code=404, detail=f"Model for {symbol} not found")
        
        model = self.models[symbol]
        preprocessor = self.preprocessors[symbol]
        
        try:
            # Convert features to DataFrame
            df = pd.DataFrame([features])
            
            # Preprocess features
            preprocessed_features = preprocessor.transform(df)
            
            # Ensure we have the right sequence length
            # For simplicity, we'll just replicate the features to create a sequence
            # In a real system, you'd use actual historical data
            if len(preprocessed_features.shape) == 2:
                # Replicate the features to create a sequence
                features_tensor = torch.tensor(
                    np.tile(preprocessed_features, (sequence_length, 1)), 
                    dtype=torch.float32
                ).unsqueeze(0)  # Add batch dimension
            else:
                features_tensor = torch.tensor(preprocessed_features, dtype=torch.float32).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                predictions = model(features_tensor).cpu().numpy()[0]
            
            # Postprocess predictions (inverse transform)
            if hasattr(preprocessor, 'inverse_transform'):
                predictions = preprocessor.inverse_transform(predictions)
            
            from datetime import datetime
            return {
                'symbol': symbol,
                'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                'prediction_time': datetime.now().isoformat(),
                'model_version': self.model_info[symbol]['version']
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Create model service instance
model_service = ModelService()


def get_model_service():
    """Dependency for getting model service instance"""
    return model_service


@router.post("/predict", response_model=PredictionOutput)
async def predict(
    input_data: PredictionInput,
    service: ModelService = Depends(get_model_service)
) -> Dict[str, Any]:
    """
    Make a prediction for the given input data.
    """
    result = service.predict(
        symbol=input_data.symbol,
        features=input_data.features,
        sequence_length=input_data.sequence_length
    )
    return result


@router.get("/models/{symbol}")
async def get_model_info(
    symbol: str,
    service: ModelService = Depends(get_model_service)
) -> Dict[str, Any]:
    """
    Get information about the model for a specific symbol.
    """
    if symbol not in service.model_info:
        # Try to load the model
        loaded = service.load_model(symbol)
        if not loaded:
            raise HTTPException(status_code=404, detail=f"Model for {symbol} not found")
    
    return service.model_info[symbol]


@router.post("/models/{symbol}/load")
async def load_model(
    symbol: str,
    model_version: Optional[str] = None,
    service: ModelService = Depends(get_model_service)
) -> Dict[str, Any]:
    """
    Load a model for a specific symbol.
    """
    success = service.load_model(symbol, model_version)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model for {symbol} not found")
    
    return {"status": "success", "model_info": service.model_info[symbol]} 