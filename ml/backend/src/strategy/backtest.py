import pandas as pd
import numpy as np
from typing import Dict, Any, Callable

class Backtester:
    def __init__(self, data: pd.DataFrame, strategy_func: Callable):
        self.data = data
        self.strategy_func = strategy_func

    def run(self) -> Dict[str, Any]:
        trades = []
        balance = 10000
        equity_curve = [balance]
        for i in range(1, len(self.data)):
            signal = self.strategy_func(self.data.iloc[:i])
            if signal == 'buy':
                entry = self.data['close'].iloc[i]
                exit_price = self.data['close'].iloc[min(i+10, len(self.data)-1)]
                pnl = (exit_price - entry) / entry * balance
                balance += pnl
                trades.append(pnl)
                equity_curve.append(balance)
        win_rate = np.mean([1 if t > 0 else 0 for t in trades]) if trades else 0
        sharpe = np.mean(trades) / (np.std(trades) + 1e-8) if trades else 0
        max_drawdown = np.min((np.array(equity_curve) - np.maximum.accumulate(equity_curve)) / np.maximum.accumulate(equity_curve)) if equity_curve else 0
        return {
            'trades': trades,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'final_balance': balance
        }

# Example usage:
# data = pd.read_csv('BTCUSD_15m.csv')
# def my_strategy(df): ...
# backtester = Backtester(data, my_strategy)
# report = backtester.run()
# print(report) 