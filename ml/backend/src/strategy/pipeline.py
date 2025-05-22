import pandas as pd
import matplotlib.pyplot as plt
from ml.backend.src.strategy.macro_bias import MacroBiasAnalyzer
from ml.backend.src.strategy.smc_detection import SMCDector
from ml.backend.src.strategy.candlestick_patterns import CandlestickPatternDetector
from ml.backend.src.strategy.crt_confluence import CRTConfluence
from ml.backend.src.strategy.orderflow import OrderFlowAnalyzer
from ml.backend.src.strategy.risk_management import RiskManager
from ml.backend.src.strategy.backtest import Backtester
import torch
from ml.backend.src.models.cnn_lstm import CNNLSTMModel
import numpy as np
import mplfinance as mpf
import ta

def main(symbol: str = 'BTCUSD'):
    # Load sample data (replace with your data source)
    ohlcv_1d = pd.read_csv('sample_data/BTCUSD_1d.csv')
    ohlcv_4h = pd.read_csv('sample_data/BTCUSD_4h.csv')
    ohlcv_15m = pd.read_csv('sample_data/BTCUSD_15m.csv')
    orderbook = pd.read_csv('sample_data/orderbook.csv')
    trades = pd.read_csv('sample_data/trades.csv')

    # Macro Bias
    macro = MacroBiasAnalyzer(ohlcv_1d, ohlcv_4h)
    macro_result = macro.analyze()

    # SMC Detection
    smc = SMCDector(ohlcv_15m)
    smc_result = smc.detect_all()

    # Candlestick Patterns
    candle = CandlestickPatternDetector(ohlcv_15m)
    candle_result = candle.detect_all()

    # Order Flow
    orderflow = OrderFlowAnalyzer(orderbook, trades)
    orderflow_result = orderflow.analyze()

    # CRT Confluence
    crt = CRTConfluence(macro_result, smc_result, candle_result, orderflow_result)
    crt_result = crt.compute_confluence()

    # Risk Management
    risk = RiskManager(account_balance=10000)
    risk_result = risk.recommend(stop_loss_pct=0.02, take_profit_pct=0.04)

    # Load trained model
    model_path = 'models/cnnlstm_trained.pt'
    seq_len = 32
    input_size = 5  # open, high, low, close, volume
    model = CNNLSTMModel(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    def model_strategy(df):
        if len(df) < seq_len:
            return 'wait'
        window = df[['open','high','low','close','volume']].values[-seq_len:]
        window = np.expand_dims(window, axis=0)  # (1, seq_len, input_size)
        window_tensor = torch.tensor(window, dtype=torch.float32)
        with torch.no_grad():
            pred = model(window_tensor).item()
        # Example: buy if predicted close > last close (regression)
        last_close = df['close'].values[-1]
        if pred > last_close:
            return 'buy'
        else:
            return 'wait'

    backtester = Backtester(ohlcv_15m, model_strategy)
    backtest_result = backtester.run()

    # Generate candlestick chart with SMA 50 overlay
    ohlcv_1d_mpf = ohlcv_1d.copy()
    ohlcv_1d_mpf = ohlcv_1d_mpf.rename(columns={'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    ohlcv_1d_mpf['Date'] = pd.to_datetime(ohlcv_1d_mpf['Date'])
    ohlcv_1d_mpf.set_index('Date', inplace=True)
    mpf.plot(ohlcv_1d_mpf, type='candle', style='charles', title=f'{symbol} 1D Candlestick', ylabel='Price', volume=True, mav=(50), savefig='candlestick_chart.png')

    # Optional: Plot extracted feature (e.g., 50-period SMA on 1D)
    plt.figure(figsize=(12,6))
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['close'], label='Close Price')
    plt.plot(macro.calculate_moving_averages(ohlcv_1d.copy())['sma_50'], label='SMA 50')
    plt.title(f'{symbol} 1D Close Price and 50-period SMA')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_verification_chart.png')
    plt.close()

    # --- Feature Engineering ---
    features = {}
    # SMA 50, SMA 200
    ohlcv_1d['sma_50'] = ta.trend.sma_indicator(ohlcv_1d['close'], window=50)
    ohlcv_1d['sma_200'] = ta.trend.sma_indicator(ohlcv_1d['close'], window=200)
    # EMA 20
    ohlcv_1d['ema_20'] = ta.trend.ema_indicator(ohlcv_1d['close'], window=20)
    # RSI
    ohlcv_1d['rsi'] = ta.momentum.rsi(ohlcv_1d['close'], window=14)
    # MACD
    ohlcv_1d['macd'] = ta.trend.macd(ohlcv_1d['close'])
    ohlcv_1d['macd_signal'] = ta.trend.macd_signal(ohlcv_1d['close'])
    # OBV
    ohlcv_1d['obv'] = ta.volume.on_balance_volume(ohlcv_1d['close'], ohlcv_1d['volume'])
    # Stochastic Oscillator
    ohlcv_1d['stoch_k'] = ta.momentum.stoch(ohlcv_1d['high'], ohlcv_1d['low'], ohlcv_1d['close'])
    ohlcv_1d['stoch_d'] = ta.momentum.stoch_signal(ohlcv_1d['high'], ohlcv_1d['low'], ohlcv_1d['close'])
    features['sma_50'] = ohlcv_1d['sma_50']
    features['sma_200'] = ohlcv_1d['sma_200']
    features['ema_20'] = ohlcv_1d['ema_20']
    features['rsi'] = ohlcv_1d['rsi']
    features['macd'] = ohlcv_1d['macd']
    features['macd_signal'] = ohlcv_1d['macd_signal']
    features['obv'] = ohlcv_1d['obv']
    features['stoch_k'] = ohlcv_1d['stoch_k']
    features['stoch_d'] = ohlcv_1d['stoch_d']

    # --- Plot all features ---
    # SMA/EMA
    plt.figure(figsize=(12,6))
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['close'], label='Close')
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['sma_50'], label='SMA 50')
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['sma_200'], label='SMA 200')
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['ema_20'], label='EMA 20')
    plt.title(f'{symbol} 1D Close, SMA, EMA')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_sma_ema.png')
    plt.close()
    # RSI
    plt.figure(figsize=(12,4))
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['rsi'], label='RSI')
    plt.axhline(70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(30, color='g', linestyle='--', alpha=0.5)
    plt.title(f'{symbol} 1D RSI')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_rsi.png')
    plt.close()
    # MACD
    plt.figure(figsize=(12,4))
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['macd'], label='MACD')
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['macd_signal'], label='MACD Signal')
    plt.title(f'{symbol} 1D MACD')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_macd.png')
    plt.close()
    # OBV
    plt.figure(figsize=(12,4))
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['obv'], label='OBV')
    plt.title(f'{symbol} 1D OBV')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_obv.png')
    plt.close()
    # Stochastic Oscillator
    plt.figure(figsize=(12,4))
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['stoch_k'], label='%K')
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['stoch_d'], label='%D')
    plt.axhline(80, color='r', linestyle='--', alpha=0.5)
    plt.axhline(20, color='g', linestyle='--', alpha=0.5)
    plt.title(f'{symbol} 1D Stochastic Oscillator')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_stoch.png')
    plt.close()

    return {
        'symbol': symbol,
        'macro_bias': macro_result,
        'smc': smc_result,
        'candlestick_patterns': candle_result,
        'orderflow': orderflow_result,
        'confluence': crt_result,
        'risk': risk_result,
        'backtest': backtest_result,
        'feature_chart': 'feature_verification_chart.png',
        'candlestick_chart': 'candlestick_chart.png',
        'features': features,
        'feature_charts': [
            'feature_sma_ema.png',
            'feature_rsi.png',
            'feature_macd.png',
            'feature_obv.png',
            'feature_stoch.png',
        ],
    }

if __name__ == "__main__":
    result = main('BTCUSD')
    for k, v in result.items():
        print(f"{k}: {v}")
    print("Feature chart saved as feature_verification_chart.png") 