# SMOOPs: Smart Money Order Blocks Trading Bot

## Overview
This project is an institutional-grade crypto trading bot that leverages advanced machine learning (ML) models to identify Smart Money Order Blocks and execute trades on Delta Exchange. The system uses Smart Money Concepts (SMC), Fair Value Gaps (FVGs), and liquidity analysis for generating high-probability trading signals. Optimized for the Apple MacBook Air M2, the system ensures efficient resource usage and high performance on consumer-grade hardware.

## Key Features
- **Smart Money Concepts Trading:** Implements Order Blocks, Fair Value Gaps, Break of Structure, and liquidity analysis
- **Machine Learning Predictions:** Utilizes PyTorch for building, training, and deploying ML models optimized for Apple Silicon
- **Delta Exchange API:** Supports both testnet and real net trading environments
- **Real-time Dashboard:** TradingView-style charts with live indicators, signals, and trade execution
- **Institutional-Grade Risk Management:** Advanced position sizing, drawdown protection, and performance monitoring
- **Secure API Key Management:** Advanced encryption system for safely storing and retrieving exchange API credentials

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

### Setup Steps

#### 1. Clone the repository
```bash
git clone https://github.com/yourusername/SMOOPs_dev.git
cd SMOOPs_dev
```

#### 2. Set up environment variables
Copy the example environment file and update with your credentials:
```bash
cp example.env .env
```
Edit the `.env` file with your:
- Delta Exchange API keys
- Database connection information
- Other configuration options

#### 3. Docker Setup (Recommended)
The easiest way to run the project is using Docker Compose:
```bash
docker-compose up -d
```

This will start all services:
- PostgreSQL database
- Backend API (available at http://localhost:3001)
- Frontend dashboard (available at http://localhost:3000)
- ML service (available at http://localhost:3002)

#### 4. Manual Setup (Alternative)
For development without Docker:

```bash
# Install dependencies for all services
npm install

# Set up the database
cd backend
npx prisma migrate dev --name init
npx prisma generate
cd ..

# Start all services
npm run dev
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
- `POST /ml/train` - Train a new model or update existing one
- `GET /ml/predict/:symbol` - Get prediction for a symbol
- `GET /ml/models` - List available models and their performance

## System Architecture

### Backend (Node.js/Express)
- RESTful API for trading operations and data access
- WebSocket server for real-time market data and signals
- Secure API key management with encryption
- Database access via Prisma ORM

### Frontend (Next.js)
- React-based dashboard with TradingView-style charts
- Real-time data visualization and order management
- Mobile-responsive design

### ML Services (Python/PyTorch)
- PyTorch models optimized for Apple Silicon
- Smart Money Concepts detection algorithms
- Backtesting framework for strategy validation
- Performance monitoring and reporting

## ML Models and Strategies

The system implements several trading strategies based on Smart Money Concepts:

- **Order Block Detection:** Identifies institutional buying and selling zones
- **Fair Value Gap Analysis:** Detects imbalances between supply and demand
- **Break of Structure & Change of Character:** Identifies trend changes
- **Liquidity Engineering Analysis:** Detects liquidity sweeps and stop hunts

These strategies are enhanced with machine learning models that optimize entry and exit points, position sizing, and risk management.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 