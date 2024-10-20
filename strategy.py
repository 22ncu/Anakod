# strategy.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

class TradingStrategy:
    def __init__(self, model, scaler, lookback=60):
        self.model = model
        self.scaler = scaler
        self.lookback = lookback
        self.positions = []
        logging.info("TradingStrategy sınıfı başlatıldı.")
    def generate_signals(self, df):
        logging.info("Al-sat sinyalleri üretiliyor.")
        
        # Scaler'ın beklediği özellikleri doğru sırada seç
        required_features = self.scaler.feature_names_in_
        available_features = [feat for feat in required_features if feat in df.columns]
        
        if len(available_features) != len(required_features):
            missing_features = set(required_features) - set(available_features)
            logging.warning(f"Bazı özellikler eksik. Eksik özellikler: {missing_features}")
        
        # Sadece mevcut özellikleri seç
        df_selected = df[available_features]
        
        # Ölçeklendirme uygula
        scaled_data = self.scale_data(df_selected)
            
        # Eksik özellikleri NaN değerlerle doldur
        for feat in required_features:
            if feat not in df.columns:
                df[feat] = np.nan
        
        # Özellikleri doğru sırada seç ve ölçeklendir
        df_selected = df[required_features]
        scaled_data = self.scale_data(df_selected)
        X, _ = self.prepare_lstm_data(scaled_data)
        predictions = self.model.predict(X)
        predictions = self.scaler.inverse_transform(np.repeat(predictions.reshape(-1, 1), 5, axis=1))[:, 0]
        df['predicted_close'] = np.concatenate((np.full(self.lookback, np.nan), predictions))
        df['signal'] = np.where(df['predicted_close'] > df['close'], 1, 0)  # 1: Al, 0: Sat
        logging.info("Al-sat sinyalleri üretildi.")
        return df

    def execute_trades(self, df, initial_balance=10000, commission=0.001):
        logging.info("Al-sat işlemleri simüle ediliyor.")
        balance = initial_balance
        position = 0
        entry_price = 0
        
        for i in range(len(df)):
            if df['signal'].iloc[i] == 1 and position == 0:  # Al sinyali
                shares = balance / df['close'].iloc[i]
                cost = shares * df['close'].iloc[i] * (1 + commission)
                if balance >= cost:
                    balance -= cost
                    position = shares
                    entry_price = df['close'].iloc[i]
                    self.positions.append(('BUY', df.index[i], df['close'].iloc[i], shares, balance))
                    logging.info(f"ALIŞ: Tarih={df.index[i]}, Fiyat={df['close'].iloc[i]}, Adet={shares}, Bakiye={balance}")
            
            elif df['signal'].iloc[i] == 0 and position > 0:  # Sat sinyali
                sale_value = position * df['close'].iloc[i] * (1 - commission)
                balance += sale_value
                self.positions.append(('SELL', df.index[i], df['close'].iloc[i], position, balance))
                logging.info(f"SATIŞ: Tarih={df.index[i]}, Fiyat={df['close'].iloc[i]}, Adet={position}, Bakiye={balance}")
                position = 0
        
        if position > 0:
            sale_value = position * df['close'].iloc[-1] * (1 - commission)
            balance += sale_value
            self.positions.append(('SELL', df.index[-1], df['close'].iloc[-1], position, balance))
            logging.info(f"SON SATIŞ: Tarih={df.index[-1]}, Fiyat={df['close'].iloc[-1]}, Adet={position}, Bakiye={balance}")
        
        total_return = (balance - initial_balance) / initial_balance
        logging.info(f"Simülasyon tamamlandı. Toplam Getiri: {total_return:.2%}")
        return balance, total_return

    def analyze_performance(self):
        if not self.positions:
            logging.warning("Analiz için pozisyon bulunmuyor.")
            return

        df_positions = pd.DataFrame(self.positions, columns=['action', 'date', 'price', 'shares', 'balance'])
        df_positions['cumulative_return'] = (df_positions['balance'] - df_positions['balance'].iloc[0]) / df_positions['balance'].iloc[0]

        total_trades = len(df_positions)
        winning_trades = len(df_positions[df_positions['balance'] > df_positions['balance'].shift(1)])
        losing_trades = total_trades - winning_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_return = df_positions['cumulative_return'].iloc[-1]
        max_drawdown = (df_positions['balance'].cummax() - df_positions['balance']) / df_positions['balance'].cummax()

        logging.info(f"Performans Analizi:")
        logging.info(f"Toplam İşlem Sayısı: {total_trades}")
        logging.info(f"Kazançlı İşlem Sayısı: {winning_trades}")
        logging.info(f"Zararlı İşlem Sayısı: {losing_trades}")
        logging.info(f"Kazanç Oranı: {win_rate:.2%}")
        logging.info(f"Toplam Getiri: {total_return:.2%}")
        logging.info(f"Maksimum Drawdown: {max_drawdown.max():.2%}")

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': max_drawdown.max()
        }

    def scale_data(self, data):
        # Scaler'ın beklediği tüm özellikleri içeren boş bir DataFrame oluştur
        empty_df = pd.DataFrame(columns=self.scaler.feature_names_in_)
        
        # Mevcut verileri boş DataFrame'e ekle
        for col in data.columns:
            if col in empty_df.columns:
                empty_df[col] = data[col]
        
        # Eksik sütunları 0 ile doldur
        empty_df = empty_df.fillna(0)
        
        # Ölçeklendirmeyi uygula
        scaled_data = self.scaler.transform(empty_df)
        
        return scaled_data

    def prepare_lstm_data(self, data):
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback), :])
            y.append(data[i + self.lookback, 3])  # 3: Close price index
        return np.array(X), np.array(y)

class RiskManager:
    def __init__(self, initial_balance, max_position_size=0.1, stop_loss_pct=0.02, take_profit_pct=0.05):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        logging.info("RiskManager sınıfı başlatıldı.")

    def calculate_position_size(self, current_price):
        max_position_value = self.current_balance * self.max_position_size
        position_size = max_position_value / current_price
        logging.info(f"Hesaplanan pozisyon büyüklüğü: {position_size}")
        return position_size

    def check_stop_loss(self, entry_price, current_price, position_type='long'):
        if position_type == 'long':
            stop_loss_hit = current_price <= entry_price * (1 - self.stop_loss_pct)
        else:  # short position
            stop_loss_hit = current_price >= entry_price * (1 + self.stop_loss_pct)
        
        if stop_loss_hit:
            logging.warning(f"Stop loss tetiklendi. Giriş fiyatı: {entry_price}, Mevcut fiyat: {current_price}")
        return stop_loss_hit

    def check_take_profit(self, entry_price, current_price, position_type='long'):
        if position_type == 'long':
            take_profit_hit = current_price >= entry_price * (1 + self.take_profit_pct)
        else:  # short position
            take_profit_hit = current_price <= entry_price * (1 - self.take_profit_pct)
        
        if take_profit_hit:
            logging.info(f"Take profit hedefine ulaşıldı. Giriş fiyatı: {entry_price}, Mevcut fiyat: {current_price}")
        return take_profit_hit

    def update_balance(self, pnl):
        self.current_balance += pnl
        logging.info(f"Bakiye güncellendi. Yeni bakiye: {self.current_balance}")

    def calculate_kelly_criterion(self, win_rate, avg_win, avg_loss):
        if avg_loss == 0:
            return 0
        kelly_percentage = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return max(0, min(kelly_percentage, self.max_position_size))

    def adjust_position_size_kelly(self, win_rate, avg_win, avg_loss, current_price):
        kelly_pct = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        position_size = (self.current_balance * kelly_pct) / current_price
        logging.info(f"Kelly kriterine göre ayarlanmış pozisyon büyüklüğü: {position_size}")
        return position_size

    def calculate_value_at_risk(self, returns, confidence_level=0.95):
        var = np.percentile(returns, (1 - confidence_level) * 100)
        logging.info(f"Hesaplanan Value at Risk (VaR): {var}")
        return var

    def adjust_max_position_size(self, var):
        self.max_position_size = min(self.max_position_size, abs(1 / var))
        logging.info(f"Maksimum pozisyon büyüklüğü VaR'a göre ayarlandı: {self.max_position_size}")

class PortfolioManager:
    def __init__(self, symbols, initial_weights=None):
        self.symbols = symbols
        self.weights = initial_weights if initial_weights else self._equal_weight()
        logging.info("PortfolioManager sınıfı başlatıldı.")

    def _equal_weight(self):
        return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

    def update_weights(self, new_weights):
        if set(new_weights.keys()) != set(self.symbols):
            raise ValueError("Yeni ağırlıklar tüm sembolleri içermelidir.")
        if not np.isclose(sum(new_weights.values()), 1.0):
            raise ValueError("Ağırlıkların toplamı 1 olmalıdır.")
        self.weights = new_weights
        logging.info("Portföy ağırlıkları güncellendi.")

    def rebalance(self, current_values):
        total_value = sum(current_values.values())
        target_values = {symbol: total_value * self.weights[symbol] for symbol in self.symbols}
        rebalance_orders = {}
        for symbol in self.symbols:
            diff = target_values[symbol] - current_values[symbol]
            if abs(diff) > 0.01 * current_values[symbol]:  # %1'den fazla sapma varsa rebalance et
                rebalance_orders[symbol] = diff
        logging.info(f"Rebalance emirleri oluşturuldu: {rebalance_orders}")
        return rebalance_orders

    def calculate_portfolio_return(self, symbol_returns):
        portfolio_return = sum(self.weights[symbol] * symbol_returns[symbol] for symbol in self.symbols)
        logging.info(f"Hesaplanan portföy getirisi: {portfolio_return}")
        return portfolio_return

    def calculate_portfolio_volatility(self, returns_df):
        cov_matrix = returns_df.cov()
        portfolio_variance = np.dot(np.dot(np.array(list(self.weights.values())), cov_matrix), np.array(list(self.weights.values())).T)
        portfolio_volatility = np.sqrt(portfolio_variance)
        logging.info(f"Hesaplanan portföy volatilitesi: {portfolio_volatility}")
        return portfolio_volatility
