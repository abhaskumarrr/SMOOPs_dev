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

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [System Optimization](#system-optimization)
- [Development Workflow](#development-workflow)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
```
SMOOPs_dev/
├── .git/
├── README.md
├── ... (source code and modules)
```

## Installation
### Prerequisites
- macOS (Apple Silicon, M2 recommended)
- Python 3.10+
- [PyTorch](https://pytorch.org/) (Apple Silicon version)
- [Taskmaster](https://github.com/your-org/taskmaster)
- [Ollama](https://ollama.com/)
- Delta Exchange India API credentials (testnet/real net)

### Setup Steps
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd SMOOPs_dev
   ```
2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you install the Apple Silicon version of PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```
4. **Configure environment variables:**
   - Set your Delta Exchange API keys in a `.env` file or export them in your shell.

5. **Install and configure Taskmaster and Ollama locally.**

## Usage
- **Run the trading bot:**
  ```bash
  python main.py
  ```
- **Switch between testnet and real net:**
  Configure the environment variable or config file as per your trading environment.
- **Taskmaster Integration:**
  Use Taskmaster CLI for managing and tracking project tasks.
- **Ollama Integration:**
  Ensure Ollama is running locally for LLM-powered features.

## System Optimization
- All ML models and data pipelines are optimized for the MacBook Air M2 (Apple Silicon).
- Use efficient data loaders, mixed precision, and batch processing to minimize resource usage.
- Monitor system performance and adjust batch sizes or model complexity as needed.

## Development Workflow
- **Version Control:** Git is used for source control. Follow best practices for branching and pull requests.
- **Task Management:** Use Taskmaster for AI-driven task tracking and project management.
- **Testing:** Implement unit and integration tests for all modules.
- **Continuous Improvement:** Regularly profile and optimize code for performance on Apple Silicon.

## Contributing
Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features. Follow the [CONTRIBUTING.md](CONTRIBUTING.md) guidelines.

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

> For more information on writing effective READMEs, see [GitHub Docs: About READMEs](https://docs.github.com/articles/about-readmes) and [README How To Guide](https://github.com/Tinymrsb/READMEhowto). 