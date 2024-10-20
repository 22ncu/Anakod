import numpy as np
import pandas as pd
import logging

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
        # VaR'a göre maksimum pozisyon büyüklüğünü ayarla
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