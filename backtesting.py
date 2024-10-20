import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from strategy import TradingStrategy
from risk_management import RiskManager, PortfolioManager
import logging
import itertools  # Parametre kombinasyonları için gerekli

class Backtester:
    def __init__(self, data, model, scaler, initial_balance=10000, commission=0.001):
        self.data = data
        self.model = model
        self.scaler = scaler
        self.initial_balance = initial_balance
        self.commission = commission
        self.strategy = TradingStrategy(model, scaler)
        self.risk_manager = RiskManager(initial_balance)
        self.portfolio_manager = PortfolioManager([data.columns[0]])  # Tek bir sembol için
        self.results = None
        logging.info("Backtester sınıfı başlatıldı.")

    def run(self):
        logging.info("Backtesting başlatılıyor.")
        df = self.strategy.generate_signals(self.data)
        
        balance = self.initial_balance
        position = 0
        entry_price = 0
        trades = []
        
        for i in tqdm(range(len(df)), desc="Backtesting Progress"):
            current_price = df['close'].iloc[i]
            
            if df['signal'].iloc[i] == 1 and position == 0:  # Al sinyali
                position_size = self.risk_manager.calculate_position_size(current_price)
                cost = position_size * current_price * (1 + self.commission)
                if balance >= cost:
                    balance -= cost
                    position = position_size
                    entry_price = current_price
                    trades.append(('BUY', df.index[i], current_price, position_size, balance))
                    logging.info(f"ALIŞ: Tarih={df.index[i]}, Fiyat={current_price}, Adet={position_size}, Bakiye={balance}")
            
            elif (df['signal'].iloc[i] == 0 and position > 0) or \
                 self.risk_manager.check_stop_loss(entry_price, current_price) or \
                 self.risk_manager.check_take_profit(entry_price, current_price):  # Sat sinyali veya stop-loss/take-profit
                sale_value = position * current_price * (1 - self.commission)
                balance += sale_value
                pnl = sale_value - (position * entry_price)
                self.risk_manager.update_balance(pnl)
                trades.append(('SELL', df.index[i], current_price, position, balance))
                logging.info(f"SATIŞ: Tarih={df.index[i]}, Fiyat={current_price}, Adet={position}, Bakiye={balance}, PnL={pnl}")
                position = 0
        
        # Son pozisyonu kapat
        if position > 0:
            sale_value = position * df['close'].iloc[-1] * (1 - self.commission)
            balance += sale_value
            pnl = sale_value - (position * entry_price)
            self.risk_manager.update_balance(pnl)
            trades.append(('SELL', df.index[-1], df['close'].iloc[-1], position, balance))
            logging.info(f"SON SATIŞ: Tarih={df.index[-1]}, Fiyat={df['close'].iloc[-1]}, Adet={position}, Bakiye={balance}, PnL={pnl}")
        
        self.results = pd.DataFrame(trades, columns=['action', 'date', 'price', 'shares', 'balance'])
        self.results['cumulative_return'] = (self.results['balance'] - self.initial_balance) / self.initial_balance
        
        logging.info("Backtesting tamamlandı.")
        return self.results

    def calculate_metrics(self):
        if self.results is None:
            raise ValueError("Önce backtesting çalıştırılmalıdır.")
        
        total_trades = len(self.results)
        winning_trades = len(self.results[self.results['balance'] > self.results['balance'].shift(1)])
        losing_trades = total_trades - winning_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_return = self.results['cumulative_return'].iloc[-1]
        
        # Sharpe Ratio hesaplama
        returns = self.results['balance'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        
        # Maximum Drawdown hesaplama
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()

        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        logging.info("Performans metrikleri hesaplandı.")
        return metrics

    def plot_results(self):
        if self.results is None:
            raise ValueError("Önce backtesting çalıştırılmalıdır.")
        
        plt.figure(figsize=(12, 8))
        plt.plot(self.results['date'], self.results['cumulative_return'])
        plt.title('Kümülatif Getiri')
        plt.xlabel('Tarih')
        plt.ylabel('Kümülatif Getiri')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        logging.info("Sonuçlar görselleştirildi.")

    def optimize_parameters(self, param_grid):
        logging.info("Parametre optimizasyonu başlatılıyor.")
        best_sharpe = -np.inf
        best_params = {}

        for params in tqdm(self._generate_param_combinations(param_grid), desc="Optimization Progress"):
            self.strategy.update_parameters(params)
            self.run()
            metrics = self.calculate_metrics()
            
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_params = params
        
        logging.info(f"En iyi parametreler bulundu: {best_params}")
        return best_params

    def _generate_param_combinations(self, param_grid):
        keys = param_grid.keys()
        values = param_grid.values()
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

# Kullanım örneği
if __name__ == "__main__":
    # Veri yükleme, model eğitimi vb. işlemler burada yapılır
    data = pd.read_csv("historical_data.csv", index_col=0, parse_dates=True)
    model = load_model("trained_model.h5")
    scaler = load_scaler("fitted_scaler.pkl")

    backtester = Backtester(data, model, scaler)
    results = backtester.run()
    metrics = backtester.calculate_metrics()
    backtester.plot_results()

    print("Performans Metrikleri:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Parametre optimizasyonu
    param_grid = {
        'lookback': [30, 60, 90],
        'threshold': [0.5, 0.6, 0.7]
    }
    best_params = backtester.optimize_parameters(param_grid)
    print(f"En iyi parametreler: {best_params}")
