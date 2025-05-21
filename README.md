# SMOOPs: SmartMarketOOPs

## Overview
This project is an institutional-grade crypto trading bot that leverages advanced machine learning (ML) models to identify Smart Money Order Blocks and execute trades on Delta Exchange. The system uses Smart Money Concepts (SMC), Fair Value Gaps (FVGs), and liquidity analysis for generating high-probability trading signals. Optimized for the Apple MacBook Air M2, the system ensures efficient resource usage and high performance on consumer-grade hardware.

## Key Features
- **Smart Money Concepts Trading:** Implements Order Blocks, Fair Value Gaps, Break of Structure, and liquidity analysis
- **Machine Learning Predictions:** Utilizes PyTorch for building, training, and deploying ML models optimized for Apple Silicon
- **Delta Exchange API:** Supports both testnet and real net trading environments
- **Real-time Dashboard:** TradingView-style charts with live indicators, signals, and trade execution
- **Institutional-Grade Risk Management:** Advanced position sizing, drawdown protection, and performance monitoring
- **Secure API Key Management:** Advanced encryption system for safely storing and retrieving exchange API credentials
- **Real-Time Trading Analytics:** Computes and stores trading metrics (PnL, Sharpe ratio, drawdown, win rate, etc.) on trade/position events.
- **Automated Compliance Reporting:** Automated script generates compliance reports, aggregating audit logs, erasure/export requests, and encryption status.

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
- `POST /ml/train` - Train a new model or update existing one
- `GET /ml/predict/:symbol` - Get prediction for a symbol
- `GET /ml/models` - List available models and their performance

### Real-Time Analytics

- **Backend**: Emits `analytics_update` events via Socket.IO when trades are executed.
- **Frontend**: Subscribe to `analytics_update` events to display live trading metrics and alerts.
- **Example**: Emit a `trade_executed` event to the backend to trigger analytics computation.

### Compliance Automation

- **Generate Compliance Report**:
  - Run: `npm run compliance:report`
  - Output: `compliance_report.json` (includes access logs, erasure requests, encryption status)
  - Schedule with cron for regular reporting.

- **GDPR Endpoints**:
  - `POST /api/compliance/export` with `{ userId }` to export user data.
  - `POST /api/compliance/erase` with `{ userId }` to soft-delete user data.

- **Checklist**: See `backend/prisma/SECURITY_COMPLIANCE.md` for compliance automation status and best practices.

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