#!/usr/bin/env python3
"""
ML Service API server for SMOOPs trading bot
"""

import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Add the project root to the sys.path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv(project_root / ".env")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ml-service")

# Create the FastAPI app
app = FastAPI(
    title="SMOOPs ML Service",
    description="Machine Learning service for Smart Money Order Blocks trading bot",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "SMOOPs ML Service"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api/predict/{symbol}")
async def predict(symbol: str):
    """
    Get predictions for a trading symbol
    
    This is a placeholder endpoint. In a real implementation, it would:
    1. Load the appropriate model for the symbol
    2. Get market data from the database or API
    3. Preprocess the data
    4. Run inference using the model
    5. Return predictions with confidence scores
    
    Args:
        symbol: Trading symbol (e.g., "BTC-USDT")
        
    Returns:
        dict: Prediction results with direction, confidence, and technical indicators
    """
    # This is dummy data for now
    return {
        "symbol": symbol,
        "prediction": {
            "direction": "bullish",
            "confidence": 0.85,
            "price_target": 45000.0,
            "stop_loss": 42500.0,
            "time_frame": "4h"
        },
        "smart_money_indicators": {
            "order_blocks": [
                {"price": 43200.0, "type": "bullish", "strength": 0.9},
                {"price": 41800.0, "type": "bullish", "strength": 0.75}
            ],
            "fair_value_gaps": [
                {"high": 44100.0, "low": 43800.0, "type": "bullish"}
            ],
            "liquidity_levels": [
                {"price": 45500.0, "type": "buy_side", "volume": "high"},
                {"price": 42000.0, "type": "sell_side", "volume": "medium"}
            ]
        },
        "timestamp": "2023-05-21T12:34:56Z"
    }

@app.get("/api/models")
async def list_models():
    """List available trained models and their performance metrics"""
    # This is dummy data for now
    return {
        "models": [
            {
                "id": "smc-btc-4h-v1",
                "symbol": "BTC-USDT",
                "time_frame": "4h",
                "type": "smart_money_concepts",
                "accuracy": 0.72,
                "profit_factor": 2.1,
                "sharpe_ratio": 1.8,
                "created_at": "2023-05-15T10:00:00Z",
                "last_updated": "2023-05-20T08:30:00Z"
            },
            {
                "id": "smc-eth-4h-v1",
                "symbol": "ETH-USDT",
                "time_frame": "4h",
                "type": "smart_money_concepts",
                "accuracy": 0.68,
                "profit_factor": 1.9,
                "sharpe_ratio": 1.6,
                "created_at": "2023-05-16T14:20:00Z",
                "last_updated": "2023-05-20T09:15:00Z"
            }
        ],
        "count": 2
    }

def main():
    """Run the FastAPI server"""
    port = int(os.getenv("ML_PORT", "3002"))
    logger.info(f"Starting ML service on port {port}")
    uvicorn.run("ml.backend.src.scripts.server:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    main() 