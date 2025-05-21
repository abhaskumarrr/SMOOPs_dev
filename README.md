# Crypto Trading Bot: Institutional-Grade ML System for Delta Exchange India

## Overview
This project is an institutional-grade crypto trading bot that leverages advanced machine learning (ML) models to predict market movements and execute trades on Delta Exchange India (testnet and real net). The system is optimized for the Apple MacBook Air M2, ensuring efficient resource usage and high performance on consumer-grade hardware. The bot combines ML-based predictions with robust trading strategies to maximize returns and manage risk.

## Key Features
- **Machine Learning Predictions:** Utilizes PyTorch for building, training, and deploying ML models for market prediction.
- **Strategy Integration:** Combines ML signals with customizable trading strategies for optimal trade execution.
- **Delta Exchange India API:** Supports both testnet and real net trading environments.
- **MacBook Air M2 Optimization:** All code and dependencies are optimized for Apple Silicon (M2), ensuring smooth operation on baseline hardware.
- **Institutional-Grade Architecture:** Designed for reliability, scalability, and robust risk management.
- **Taskmaster & Ollama Integration:** Uses Taskmaster for AI-driven task management and Ollama for local LLM operations.
- **Secure API Key Management:** Advanced encryption system for safely storing and retrieving Delta Exchange API credentials.
- **Modern Tech Stack:** Built with Next.js frontend, Node.js/Express backend, and Python ML services.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [System Optimization](#system-optimization)
- [Development Workflow](#development-workflow)
- [Security Features](#security-features)
- [Contributing](#contributing)
- [License](#license)

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
├── scripts/            # Utility scripts and tooling
├── tasks/              # Taskmaster task definitions
├── docker-compose.yml  # Docker services configuration
└── README.md           # Project documentation
```

## Installation

### Prerequisites
- macOS (Apple Silicon, M2 recommended)
- Node.js 18+ and npm/yarn
- Python 3.10+
- Docker and Docker Compose (optional, for containerized setup)
- [PyTorch](https://pytorch.org/) (Apple Silicon version)
- [Taskmaster](https://github.com/your-org/taskmaster)
- [Ollama](https://ollama.com/)
- Delta Exchange India API credentials (testnet/real net)
- PostgreSQL database (or use the provided Docker setup)

### Setup Steps

#### 1. Clone the repository
```bash
git clone https://github.com/abhaskumarrr/SMOOPs_dev.git
cd SMOOPs_dev
```

#### 2. Set up environment variables
Copy the example environment file and update with your credentials:
```bash
cp example.env .env
```
Edit the `.env` file with your:
- Delta Exchange API keys
- Database connection string
- Other configuration options

#### 3. Install dependencies
For the full monorepo setup:
```bash
npm install
```

For individual services:
```bash
# Backend dependencies
cd backend && npm install

# Frontend dependencies
cd frontend && npm install

# ML dependencies
cd ml && pip install -r requirements.txt
```

#### 4. Database setup
The project uses Prisma ORM for database management:
```bash
cd backend
npx prisma migrate dev
npx prisma generate
```

#### 5. Start the development servers
```bash
# Start all services
npm run dev

# Or start individual services
npm run dev:backend
npm run dev:frontend
npm run dev:ml
```

## Usage

### API Endpoints
The backend provides several API endpoints:

#### API Key Management
- `GET /api/keys` - List all API keys for a user (masked for security)
- `POST /api/keys` - Add a new API key
- `GET /api/keys/validate` - Validate an API key
- `DELETE /api/keys/:keyId` - Revoke an API key

#### Health Check
- `GET /health` - Check server status

### Frontend Dashboard
- Access the trading dashboard at `http://localhost:3000`
- View real-time market data, trading history, and ML model predictions
- Manage your Delta Exchange API keys securely

### ML Services
- ML prediction services run on `http://localhost:3002`
- Models are automatically trained and deployed for prediction

### Docker Deployment
For a containerized setup:
```bash
docker-compose up -d
```

## System Architecture

### Backend (Node.js/Express)
- RESTful API design with proper error handling
- Secure API key management with AES-256-GCM encryption
- Prisma ORM for type-safe database access
- Authentication and authorization middleware

### Frontend (Next.js)
- Modern React application with server-side rendering
- Real-time data visualization with charting libraries
- Responsive UI for desktop and mobile devices
- Secure communication with backend services

### ML Services (Python/PyTorch)
- PyTorch models optimized for Apple Silicon
- Feature engineering and data preprocessing pipelines
- Model versioning and A/B testing framework
- Real-time and batch prediction capabilities

### Database (PostgreSQL)
- Relational database for storing user data, API keys, and trade logs
- Proper indexing and optimization for performance
- Regular backups and disaster recovery strategy

## System Optimization
- All ML models and data pipelines are optimized for the MacBook Air M2 (Apple Silicon).
- Use efficient data loaders, mixed precision, and batch processing to minimize resource usage.
- Monitor system performance and adjust batch sizes or model complexity as needed.
- Containerized services with resource limits to prevent system overload.

## Development Workflow
- **Version Control:** Git is used for source control. Follow best practices for branching and pull requests.
- **Task Management:** Use Taskmaster for AI-driven task tracking and project management.
- **Testing:** Implement unit and integration tests for all modules.
- **Continuous Improvement:** Regularly profile and optimize code for performance on Apple Silicon.
- **Monorepo Structure:** Unified repository for all services, ensuring consistent versioning and dependency management.

## Security Features

### API Key Management
- AES-256-GCM encryption for Delta Exchange API keys
- Per-user encryption keys with key derivation functions
- Authentication tags to prevent tampering
- Masked keys in UI and logs for improved security
- Automatic key rotation and expiry mechanisms

### User Authentication
- Secure authentication system with JWT tokens
- Password hashing with bcrypt
- Role-based access controls
- Protection against common web vulnerabilities

### Data Protection
- HTTPS communication between all services
- Input validation and sanitization
- Rate limiting to prevent abuse
- Audit logging for security-relevant events

## Contributing
Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features. Follow the [CONTRIBUTING.md](CONTRIBUTING.md) guidelines.

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

> For questions or support, please contact the project maintainers. 