"""
Model Registry

This module provides utilities for managing model versioning, storage, and retrieval.
"""

import os
import json
import shutil
import pickle
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default registry path
DEFAULT_REGISTRY_PATH = os.environ.get("MODEL_REGISTRY_PATH", "models/registry")


class ModelRegistry:
    """
    Registry for managing ML models with versioning support.
    
    This class handles saving, loading, and tracking model versions.
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path to the model registry directory
        """
        self.registry_path = Path(registry_path or DEFAULT_REGISTRY_PATH)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model registry initialized at {self.registry_path}")
    
    def save_model(
        self,
        model: torch.nn.Module,
        symbol: str,
        metadata: Dict[str, Any],
        preprocessor: Optional[Any] = None,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        artifacts: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a model to the registry with versioning.
        
        Args:
            model: The PyTorch model to save
            symbol: Trading symbol or model identifier
            metadata: Model metadata (training params, architecture, etc.)
            preprocessor: Optional preprocessor used with the model
            version: Optional specific version to use (default: timestamp-based)
            metrics: Optional performance metrics
            artifacts: Optional additional artifacts to save
            
        Returns:
            The version string of the saved model
        """
        # Normalize symbol name for file paths
        symbol_name = symbol.replace("/", "_")
        
        # Create version string if not provided
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v_{timestamp}"
        
        # Create model directory
        model_dir = self.registry_path / symbol_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model checkpoint
        model_path = model_dir / "model.pt"
        
        # Get model configuration
        model_config = {}
        for attr in ['input_dim', 'output_dim', 'hidden_dim', 'num_layers', 
                    'dropout', 'seq_len', 'forecast_horizon']:
            if hasattr(model, attr):
                model_config[attr] = getattr(model, attr)
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'model_type': model.__class__.__name__,
        }
        
        # Save the model
        torch.save(checkpoint, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Also save as best.pt for compatibility with existing code
        torch.save(checkpoint, model_dir / "best.pt")
        
        # Save preprocessor if provided
        if preprocessor is not None:
            preprocessor_path = model_dir / "preprocessor.pkl"
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")
        
        # Save metrics if provided
        if metrics is not None:
            metrics_path = model_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_path}")
        
        # Save additional artifacts if provided
        if artifacts is not None:
            artifacts_dir = model_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            
            for name, artifact in artifacts.items():
                if isinstance(artifact, (dict, list)):
                    # Save JSON serializable artifacts
                    artifact_path = artifacts_dir / f"{name}.json"
                    with open(artifact_path, 'w') as f:
                        json.dump(artifact, f, indent=2)
                elif isinstance(artifact, (str, bytes, Path)):
                    # Save file artifacts
                    if isinstance(artifact, (str, Path)):
                        src_path = Path(artifact)
                        if src_path.exists():
                            dst_path = artifacts_dir / src_path.name
                            shutil.copy(src_path, dst_path)
                            logger.info(f"Artifact {name} copied to {dst_path}")
                    else:
                        # Save bytes
                        artifact_path = artifacts_dir / f"{name}.bin"
                        with open(artifact_path, 'wb') as f:
                            f.write(artifact)
                else:
                    try:
                        # Try to pickle the artifact
                        artifact_path = artifacts_dir / f"{name}.pkl"
                        with open(artifact_path, 'wb') as f:
                            pickle.dump(artifact, f)
                        logger.info(f"Artifact {name} saved to {artifact_path}")
                    except Exception as e:
                        logger.warning(f"Could not save artifact {name}: {str(e)}")
        
        return version
    
    def load_model(
        self,
        symbol: str,
        version: Optional[str] = None,
        return_metadata: bool = False,
        return_preprocessor: bool = False,
        device: Optional[str] = None
    ) -> Union[torch.nn.Module, Tuple]:
        """
        Load a model from the registry.
        
        Args:
            symbol: Trading symbol or model identifier
            version: Specific version to load (default: latest)
            return_metadata: Whether to return metadata with the model
            return_preprocessor: Whether to return the preprocessor with the model
            device: Device to load the model on (default: CPU)
            
        Returns:
            The loaded model or a tuple of (model, [metadata], [preprocessor])
        """
        from ..models import ModelFactory
        
        # Normalize symbol name for file paths
        symbol_name = symbol.replace("/", "_")
        
        # Determine model directory
        model_dir = self.registry_path / symbol_name
        if not model_dir.exists():
            raise FileNotFoundError(f"No models found for {symbol}")
        
        # Find the specific version or latest
        if version is None:
            versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
            if not versions:
                raise FileNotFoundError(f"No model versions found for {symbol}")
            version = sorted(versions)[-1]
        
        version_dir = model_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found for {symbol}")
        
        # Load model
        model_path = version_dir / "best.pt"
        if not model_path.exists():
            model_path = version_dir / "model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found in {version_dir}")
        
        # Map model to device
        device_map = torch.device('cpu' if device is None else device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device_map)
        
        # Get model config and type
        model_config = checkpoint.get('model_config', {})
        model_type = checkpoint.get('model_type', '').lower()
        if not model_type and 'model_type' in model_config:
            model_type = model_config['model_type'].lower()
        
        # Default to LSTM if type not found
        if not model_type:
            model_type = 'lstm'
            logger.warning(f"Model type not found in checkpoint, defaulting to {model_type}")
        
        # Create model instance
        model = ModelFactory.create_model(
            model_type=model_type,
            input_dim=model_config.get('input_dim', 10),
            output_dim=model_config.get('output_dim', 1),
            seq_len=model_config.get('seq_len', 60),
            forecast_horizon=model_config.get('forecast_horizon', 5),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_layers=model_config.get('num_layers', 2),
            device=device
        )
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        results = [model]
        
        # Load metadata if requested
        if return_metadata:
            metadata_path = version_dir / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            results.append(metadata)
        
        # Load preprocessor if requested
        if return_preprocessor:
            preprocessor = None
            preprocessor_path = version_dir / "preprocessor.pkl"
            if preprocessor_path.exists():
                with open(preprocessor_path, 'rb') as f:
                    preprocessor = pickle.load(f)
            results.append(preprocessor)
        
        if len(results) == 1:
            return results[0]
        return tuple(results)
    
    def get_versions(self, symbol: str) -> List[str]:
        """
        Get all available versions for a symbol.
        
        Args:
            symbol: Trading symbol or model identifier
            
        Returns:
            List of available versions
        """
        # Normalize symbol name for file paths
        symbol_name = symbol.replace("/", "_")
        
        # Get directory
        model_dir = self.registry_path / symbol_name
        if not model_dir.exists():
            return []
        
        # Get versions
        versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
        return sorted(versions)
    
    def get_symbols(self) -> List[str]:
        """
        Get all available symbols in the registry.
        
        Returns:
            List of available symbols
        """
        if not self.registry_path.exists():
            return []
        
        # Get all directories
        symbols = [d.name for d in self.registry_path.iterdir() if d.is_dir()]
        
        # Convert back to original format
        symbols = [s.replace("_", "/") for s in symbols]
        return sorted(symbols)
    
    def get_metadata(self, symbol: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a specific model version.
        
        Args:
            symbol: Trading symbol or model identifier
            version: Specific version to get metadata for (default: latest)
            
        Returns:
            Model metadata
        """
        # Normalize symbol name for file paths
        symbol_name = symbol.replace("/", "_")
        
        # Determine model directory
        model_dir = self.registry_path / symbol_name
        if not model_dir.exists():
            raise FileNotFoundError(f"No models found for {symbol}")
        
        # Find the specific version or latest
        if version is None:
            versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
            if not versions:
                raise FileNotFoundError(f"No model versions found for {symbol}")
            version = sorted(versions)[-1]
        
        version_dir = model_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found for {symbol}")
        
        # Load metadata
        metadata_path = version_dir / "metadata.json"
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def delete_version(self, symbol: str, version: str, force: bool = False) -> bool:
        """
        Delete a specific model version.
        
        Args:
            symbol: Trading symbol or model identifier
            version: Specific version to delete
            force: Force deletion even if it's the only version
            
        Returns:
            True if deletion was successful
        """
        # Normalize symbol name for file paths
        symbol_name = symbol.replace("/", "_")
        
        # Determine version directory
        version_dir = self.registry_path / symbol_name / version
        if not version_dir.exists():
            logger.warning(f"Model version {version} not found for {symbol}")
            return False
        
        # Check if it's the only version
        if not force:
            versions = self.get_versions(symbol)
            if len(versions) <= 1:
                logger.warning(f"Cannot delete the only version for {symbol}. Use force=True to override.")
                return False
        
        # Delete the directory
        shutil.rmtree(version_dir)
        logger.info(f"Deleted model version {version} for {symbol}")
        return True
    
    def compare_versions(
        self,
        symbol: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            symbol: Trading symbol or model identifier
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Comparison results
        """
        # Load metrics for both versions
        metadata1 = self.get_metadata(symbol, version1)
        metadata2 = self.get_metadata(symbol, version2)
        
        # Load metrics
        metrics1 = {}
        metrics2 = {}
        
        metrics_path1 = self.registry_path / symbol.replace("/", "_") / version1 / "metrics.json"
        metrics_path2 = self.registry_path / symbol.replace("/", "_") / version2 / "metrics.json"
        
        if metrics_path1.exists():
            with open(metrics_path1, 'r') as f:
                metrics1 = json.load(f)
        
        if metrics_path2.exists():
            with open(metrics_path2, 'r') as f:
                metrics2 = json.load(f)
        
        # Prepare comparison results
        comparison = {
            "symbol": symbol,
            "versions": {
                "v1": version1,
                "v2": version2
            },
            "metadata_diff": {
                k: {"v1": metadata1.get(k), "v2": metadata2.get(k)}
                for k in set(metadata1.keys()).union(set(metadata2.keys()))
                if k in metadata1 and k in metadata2 and metadata1.get(k) != metadata2.get(k)
            },
            "metrics_diff": {
                k: {"v1": metrics1.get(k), "v2": metrics2.get(k)}
                for k in set(metrics1.keys()).union(set(metrics2.keys()))
            }
        }
        
        return comparison


# Singleton instance of the registry
_registry_instance = None

def get_registry() -> ModelRegistry:
    """
    Get or create the singleton registry instance.
    
    Returns:
        The model registry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance 