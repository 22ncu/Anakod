import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import time
import logging
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import MACD, ADXIndicator, SMAIndicator, EMAIndicator, IchimokuIndicator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from sklearn.preprocessing import MinMaxScaler

def fetch_data(binance, symbol='BCH/USDT', timeframe='1m', days=1):
    logging.info(f"Veri çekme başlatılıyor: {symbol}, Timeframe: {timeframe}, Gün: {days}")
    since = binance.milliseconds() - days * 24 * 60 * 60 * 1000
    all_bars = []
    total_bars = math.ceil((days * 24 * 60) / 1000)

    for i in tqdm(range(total_bars), desc="Veri Çekme İlerlemesi"):
        try:
            bars = binance.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not bars:
                logging.warning("Hiç veri bulunamadı. Veri çekme işlemi sonlandırılıyor.")
                break
            all_bars += bars
            since = bars[-1][0] + 60 * 1000
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Veri çekme sırasında hata: {e}")
            break

    if not all_bars:
        logging.error("Hiç veri çekilemedi.")
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    logging.info(f"Veri başarıyla çekildi. Toplam veri noktası: {len(df)}")
    return df

def calculate_indicators(df):
    logging.info("Teknik göstergelerin hesaplanması başlatılıyor.")
    try:
        # Bollinger Bands
        indicator_bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_mavg'] = indicator_bb.bollinger_mavg()
        df['bb_high'] = indicator_bb.bollinger_hband()
        df['bb_low'] = indicator_bb.bollinger_lband()
        
        # MACD
        indicator_macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = indicator_macd.macd()
        df['macd_signal'] = indicator_macd.macd_signal()
        df['macd_diff'] = indicator_macd.macd_diff()
        
        # RSI
        indicator_rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = indicator_rsi.rsi()
        
        # ADX
        indicator_adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['adx'] = indicator_adx.adx()
        
        # ATR
        indicator_atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = indicator_atr.average_true_range()
        
        # Stochastic Oscillator
        indicator_stochastic = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
        df['stoch_k'] = indicator_stochastic.stoch()
        df['stoch_d'] = indicator_stochastic.stoch_signal()
        
        # ROC (Rate of Change) Indicator
        indicator_roc = ROCIndicator(close=df['close'], window=10)
        df['roc'] = indicator_roc.roc()
        
        # SMA
        indicator_sma = SMAIndicator(close=df['close'], window=30)
        df['sma_30'] = indicator_sma.sma_indicator()
        
        # EMA
        indicator_ema = EMAIndicator(close=df['close'], window=30)
        df['ema_30'] = indicator_ema.ema_indicator()
        
        # Ichimoku Indicator
        indicator_ichimoku = IchimokuIndicator(high=df['high'], low=df['low'], window1=9, window2=26, window3=52)
        df['ichimoku_a'] = indicator_ichimoku.ichimoku_a()
        df['ichimoku_b'] = indicator_ichimoku.ichimoku_b()
        
        # On-Balance Volume (OBV)
        indicator_obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['obv'] = indicator_obv.on_balance_volume()
        
        # Chaikin Money Flow (CMF)
        indicator_cmf = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=20)
        df['cmf'] = indicator_cmf.chaikin_money_flow()
        
        logging.info("Teknik göstergeler başarıyla hesaplandı.")
    except Exception as e:
        logging.error(f"Göstergelerin hesaplanması sırasında hata: {e}")
        raise
    return df

def detect_candlestick_patterns(df):
    logging.info("Candlestick formasyonlarının tespiti başlatılıyor.")
    try:
        df['body'] = df['close'] - df['open']
        df['range'] = df['high'] - df['low']
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        
        # Hammer
        df['hammer'] = np.where((df['lower_shadow'] > 2 * df['body'].abs()) & 
                                (df['upper_shadow'] < 0.1 * df['body'].abs()), 1, 0)
        
        # Doji
        df['doji'] = np.where(df['body'].abs() <= 0.1 * df['range'], 1, 0)
        
        logging.info("Candlestick formasyonları başarıyla tespit edildi.")
    except Exception as e:
        logging.error(f"Candlestick formasyonlarının tespiti sırasında hata: {e}")
        raise
    return df

def calculate_fibonacci_levels(df, period=50):
    logging.info("Fibonacci seviyelerinin hesaplanması başlatılıyor.")
    try:
        recent_high = df['high'].rolling(window=period).max()
        recent_low = df['low'].rolling(window=period).min()
        
        df['fib_0'] = recent_high
        df['fib_23.6'] = recent_high - 0.236 * (recent_high - recent_low)
        df['fib_38.2'] = recent_high - 0.382 * (recent_high - recent_low)
        df['fib_50'] = recent_high - 0.5 * (recent_high - recent_low)
        df['fib_61.8'] = recent_high - 0.618 * (recent_high - recent_low)
        df['fib_76.4'] = recent_high - 0.764 * (recent_high - recent_low)
        df['fib_100'] = recent_low
        
        logging.info("Fibonacci seviyeleri başarıyla hesaplandı.")
    except Exception as e:
        logging.error(f"Fibonacci seviyelerinin hesaplanması sırasında hata: {e}")
        raise
    return df

def scale_data(df):
    logging.info("Verilerin ölçeklendirilmesi başlatılıyor.")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close', 'volume', 'open', 'high', 'low']])
    logging.info("Veriler başarıyla ölçeklendirildi.")
    return scaled_data, scaler

def prepare_lstm_data(scaled_data, lookback=60):
    logging.info("LSTM verisinin hazırlanması başlatılıyor.")
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0])  # Sadece kapanış fiyatını tahmin ediyoruz
    X, y = np.array(X), np.array(y)
    logging.info(f"LSTM verisi hazırlandı. X shape: {X.shape}, y shape: {y.shape}")
    return X, y