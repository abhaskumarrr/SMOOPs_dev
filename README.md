# SMOOPs: Automated Crypto ML Trading Pipeline

## Overview
SMOOPs is a production-grade, fully automated machine learning pipeline for cryptocurrency trading. It features robust model training, strict preprocessor and feature alignment, reproducible inference, and a powerful backtesting engine. The system is managed and documented using Taskmaster for maximum reproducibility and team collaboration.

## Key Features
- **End-to-end ML pipeline**: Data ingestion, feature engineering, model training, evaluation, prediction, and backtesting.
- **Strict feature alignment**: Ensures that features used in training, prediction, and backtesting are always identical in name and order, preventing silent bugs.
- **Preprocessor persistence**: The exact fitted preprocessor (e.g., StandardScaler) is saved and loaded with each model checkpoint.
- **Robust backtesting**: Automated backtest engine with ML model integration, strict input dimension checks, and clear error reporting.
- **Taskmaster integration**: All tasks, subtasks, and workflow documentation are managed with Taskmaster for transparency and reproducibility.

## Workflow
1. **Train a Model**
   ```bash
   python3 -m ml.src.cli train --symbol BTCUSD --model-type lstm --data-path sample_data/BTCUSD_15m.csv --num-epochs 100 --batch-size 32 --sequence-length 60 --forecast-horizon 1
   ```
   - Saves model, preprocessor, and metadata in `models/registry/<SYMBOL>/<VERSION>/`.

2. **Make Predictions**
   ```bash
   python3 -m ml.src.cli predict --symbol BTCUSD --data-file sample_data/BTCUSD_15m.csv --output-file predictions.csv
   ```
   - Loads the latest model and preprocessor, applies exact feature engineering, and outputs predictions.

3. **Run Backtest**
   ```bash
   python3 -m ml.src.cli backtest --data-file sample_data/BTCUSD_15m.csv --strategy ml_model --symbol BTCUSD --model-type lstm --model-checkpoint models/registry/BTCUSD/<VERSION>/model.pt --preprocessor models/registry/BTCUSD/<VERSION>/preprocessor.pkl --output-dir runs/backtest/
   ```
   - Ensures feature and preprocessor alignment, outputs robust backtest results.

## Troubleshooting
- **Input Dimension Errors**: If you see errors like `size mismatch for lstm.weight_ih_l0`, check that your feature engineering and preprocessor match exactly between training and inference. See [PyTorch LSTM size mismatch discussion](https://discuss.pytorch.org/t/time-series-lstm-size-mismatch-beginner-question/4704).
- **Feature Mismatch**: The pipeline will log expected vs. actual feature columns and raise a clear error if they do not match.

## Taskmaster Usage
- All project tasks, subtasks, and workflow documentation are managed with Taskmaster.
- To regenerate markdown documentation and task files:
  ```bash
  task-master generate
  ```
- For more, see `.taskmasterconfig` and the `tasks/` directory.

## References
- [PyTorch LSTM: Size mismatch and feature alignment](https://discuss.pytorch.org/t/time-series-lstm-size-mismatch-beginner-question/4704)
- [Best practices for ML pipeline automation](https://www.markovml.com/blog/machine-learning-pipeline)
- [Taskmaster documentation](./.taskmasterconfig)

---
For questions or contributions, see the `docs/` directory or open an issue.

## Documentation

- **[Development Guide](docs/DEVELOPMENT.md)** - Comprehensive setup and development workflow
- **[Deployment Guide](docs/deployment-guide.md)** - Deployment options and procedures
- **[Environment Setup](docs/environment-setup.md)** - Environment configuration details
- **[Project Structure](docs/project-structure.md)** - Detailed breakdown of the codebase

## Project Structure
```
SMOOPs_dev/
├── .github/            # GitHub workflows for CI/CD
├── backend/            # Node.js/Express backend API
│   ├── prisma/         # Database schema and migrations
│   ├── src/            # Backend source code
│   │   ├── controllers/# API controllers
│   │   ├── middleware/ # Express middleware
│   │   ├── routes/     # API routes
│   │   ├── services/   # Business logic
│   │   └── utils/      # Utility functions (encryption, etc.)
├── frontend/           # Next.js frontend application
│   ├── pages/          # Next.js pages
│   └── public/         # Static assets
├── ml/                 # Python ML models and services
│   ├── src/            # ML source code
│   │   ├── api/        # ML service API
│   │   ├── backtesting/# Backtesting framework
│   │   ├── data/       # Data processing pipelines
│   │   ├── models/     # ML model definitions
│   │   ├── training/   # Training pipelines
│   │   ├── monitoring/ # Performance monitoring
│   │   └── utils/      # Utility functions
│   ├── data/           # Data storage
│   │   ├── raw/        # Raw market data
│   │   └── processed/  # Processed datasets
│   ├── models/         # Saved model checkpoints
│   └── logs/           # Training and evaluation logs
├── scripts/            # Utility scripts and tooling
├── tasks/              # Task definitions and project management
├── docker-compose.yml  # Docker services configuration
└── README.md           # Project documentation
```

## Installation

### Prerequisites
- macOS (Apple Silicon recommended) or Linux
- Node.js 20+ and npm
- Python 3.10+
- Docker and Docker Compose (for containerized setup)
- Delta Exchange API credentials (testnet/real net)

### Quick Setup
The fastest way to get started is using our automated setup script:

```bash
# Clone the repository
git clone https://github.com/abhaskumarrr/SMOOPs_dev.git
cd SMOOPs_dev

# Run the development setup script
npm run dev:setup
```

For detailed setup instructions, see the [Development Guide](docs/DEVELOPMENT.md).

### Docker Setup (Recommended)
The easiest way to run the project is using Docker Compose:
```bash
docker-compose up -d
```

This will start all services:
- PostgreSQL database
- Backend API (available at http://localhost:3001)
- Frontend dashboard (available at http://localhost:3000)
- ML service (available at http://localhost:3002)

### Development Tools

SMOOPs includes several helpful development tools:

```bash
# Run common development tasks
npm run dev:tasks

# View specific development task options
npm run dev:tasks help
```

## Usage

### Trading Dashboard
Access the trading dashboard at `http://localhost:3000` to:
- View real-time market data with SMC indicators
- Monitor trading signals and executed trades
- Analyze performance metrics
- Configure trading strategies
- Manage API keys securely

### API Endpoints
The backend provides several API endpoints:

#### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/register` - Create a new user account

#### API Key Management
- `GET /api/keys` - List all API keys for a user
- `POST /api/keys` - Add a new API key
- `DELETE /api/keys/:id` - Remove an API key

#### Trading
- `GET /api/delta/instruments` - Get available trading instruments
- `GET /api/delta/market/:symbol` - Get real-time market data
- `POST /api/delta/order` - Place a new order
- `GET /api/delta/orders` - Get order history

### ML Service
The ML service exposes endpoints for model training and prediction:
- `