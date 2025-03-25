# Bitcoin Trading System using Genetic Algorithm

This project implements an automated Bitcoin trading system using Genetic Algorithm (GA) optimization. The system is designed to evolve and optimize trading strategies for cryptocurrency markets, specifically focusing on Bitcoin trading.

## Project Structure

```
bit_base/
├── Evolution/           # Genetic Algorithm core implementation
│   ├── callbacks/       # Callback functions for GA
│   ├── crossover/      # Crossover operations
│   ├── mutation/       # Mutation operations
│   └── selection/      # Selection strategies
├── Prescriptor/        # Trading strategy prescriptor
├── data/              # Data storage
├── feature_calculations.py  # Feature engineering
├── strategies.py      # Trading strategies implementation
└── train_functions/   # Various training implementations
    ├── train_functions.py
    ├── train_functions_bi.py
    ├── train_functions_gpu.py
    ├── train_functions_rl.py
    └── ...
```

## Key Features

- **Genetic Algorithm Implementation**: Custom GA framework for optimizing trading strategies
- **Multiple Training Approaches**: 
  - Basic training
  - Bi-directional training
  - GPU-accelerated training
  - Reinforcement Learning integration
- **Feature Engineering**: Comprehensive technical indicators and feature calculations
- **Strategy Evolution**: Dynamic evolution of trading strategies using genetic operations
- **Multi-objective Optimization**: Support for multiple trading objectives

## Training Modules

1. **Basic Training** (`train_functions.py`)
   - Standard GA-based strategy optimization

2. **Bi-directional Training** (`train_functions_bi.py`, `train_functions_bi_cul.py`)
   - Enhanced training with bi-directional analysis
   - Cumulative learning capabilities

3. **GPU Acceleration** (`train_functions_gpu.py`)
   - GPU-optimized implementations for faster training

4. **Reinforcement Learning** (`train_functions_rl.py`)
   - RL-integrated approach for strategy optimization

## Evolution Framework

The Evolution module provides a comprehensive framework for genetic algorithm implementation:

- **Callbacks**: Monitor and control the evolution process
- **Crossover**: Various genetic crossover operations
- **Mutation**: Different mutation strategies
- **Selection**: Multiple selection methods including:
  - Single objective selection
  - Multi-objective selection
  - Pareto-based selection

## Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Required trading data in the data/ directory

## Usage

1. Data Preparation:
```python
from data_loader import DataLoader
from dataset import Dataset

# Prepare your trading data
data_loader = DataLoader()
dataset = Dataset()
```

2. Feature Calculation:
```python
from feature_calculations import calculate_features

# Calculate technical indicators and features
features = calculate_features(data)
```

3. Training:
```python
from train_functions import train_model

# Train the trading model
model = train_model(dataset, features)
```

4. Strategy Execution:
```python
from strategies import execute_strategy

# Execute the evolved trading strategy
results = execute_strategy(model, market_data)
```

## Notes

- Ensure proper data format and preprocessing before training
- GPU acceleration requires compatible hardware
- Model checkpoints (.pt, .pth, .pkl) are excluded from version control
- Adjust hyperparameters based on your specific trading requirements

## License

Private repository - All rights reserved

---
**Note**: This is a private repository containing proprietary trading strategies. Please ensure proper security measures when handling the code and trading algorithms. 