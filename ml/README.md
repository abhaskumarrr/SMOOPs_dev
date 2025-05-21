# SMOOPs ML Module

The SMOOPs ML (Machine Learning) module provides advanced trading model development, training, and serving capabilities for algorithmic trading. It includes components for data preprocessing, model training, hyperparameter tuning, model registry, and performance monitoring.

## Features

- **Time Series Forecasting Models**: LSTM, GRU, Transformer, and CNN-LSTM hybrid architectures
- **Hyperparameter Tuning**: Grid search, random search, and Bayesian optimization
- **Model Registry**: Version control and tracking for models
- **Performance Monitoring**: Track model performance over time with drift detection 
- **REST API**: Serve predictions via a FastAPI-based REST API
- **CLI Interface**: Train, tune, evaluate, predict, and monitor via command line

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SMOOPs.git
cd SMOOPs/ml

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training a Model

```bash
python main.py train --symbol BTC/USDT --model-type lstm --sequence-length 60 --forecast-horizon 5
```

### Tuning Hyperparameters

```bash
python main.py tune --symbol BTC/USDT --model-type lstm --method bayesian --n-trials 20
```

### Starting the API Server

```bash
python main.py serve --port 8000
```

### Making Predictions

```bash
python main.py predict --symbol BTC/USDT --input-path data/samples/input.json --output-path predictions.json
```

### Monitoring Performance

```bash
python main.py monitor --symbol BTC/USDT --last-n-days 30 --output-path report.json
```

## Project Structure

```
ml/
├── backend/              # Simplified backend for local development
├── data/                 # Data directory
│   ├── processed/        # Processed datasets
│   └── raw/              # Raw data
├── logs/                 # Log files
│   └── tensorboard/      # TensorBoard logs
├── models/               # Trained models
│   └── registry/         # Model registry
├── reports/              # Performance reports and visualizations
├── src/                  # Source code
│   ├── api/              # API endpoints
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model definitions
│   ├── monitoring/       # Performance monitoring
│   ├── training/         # Training utilities
│   └── utils/            # Utility functions
├── main.py               # CLI entry point
└── requirements.txt      # Project dependencies
```

## API Documentation

When the API server is running, visit `http://localhost:8000/docs` for Swagger UI documentation of all endpoints.

### Key Endpoints

- **/api/models/predict/{symbol}**: Make predictions for a symbol
- **/api/models/batch_predict**: Make batch predictions
- **/api/registry/symbols**: List available models by symbol
- **/api/registry/versions/{symbol}**: List available versions for a symbol
- **/api/monitoring/metrics/{symbol}**: Get performance metrics history
- **/api/monitoring/report/{symbol}**: Generate performance report

## Command Line Interface

### Train a Model

```bash
python main.py train --symbol BTC/USDT \
                     --model-type lstm \
                     --sequence-length 60 \
                     --forecast-horizon 5 \
                     --batch-size 32 \
                     --num-epochs 100 \
                     --learning-rate 0.001
```

### Tune Hyperparameters

```bash
python main.py tune --symbol BTC/USDT \
                    --model-type lstm \
                    --method bayesian \
                    --n-trials 20 \
                    --param-grid '{"num_layers": [1, 2, 3], "hidden_size": [32, 64, 128]}'
```

### Start API Server

```bash
python main.py serve --host 0.0.0.0 --port 8000
```

### Evaluate a Model

```bash
python main.py evaluate --symbol BTC/USDT \
                        --output-path evaluation.csv
```

### Make Predictions

```bash
python main.py predict --symbol BTC/USDT \
                       --input-path data/samples/input.json \
                       --output-path predictions.json
```

### Monitor Performance

```bash
python main.py monitor --symbol BTC/USDT \
                       --last-n-days 30 \
                       --output-path report.json
```

## Development

### Running Tests

```bash
pytest tests/
```

With coverage:

```bash
pytest --cov=src tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests for your changes
5. Run the tests
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 