import pandas as pd
import numpy as np
import ta
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class MarketAnalyzer:
    def __init__(self, symbol, timeframe='1d'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = None
        logging.info(f"MarketAnalyzer sınıfı başlatıldı. Sembol: {symbol}, Zaman dilimi: {timeframe}")

    def fetch_data(self, start_date, end_date):
        logging.info(f"{self.symbol} için veri çekiliyor. Başlangıç: {start_date}, Bitiş: {end_date}")
        try:
            self.data = yf.download(self.symbol, start=start_date, end=end_date, interval=self.timeframe)
            logging.info(f"Veri başarıyla çekildi. Satır sayısı: {len(self.data)}")
        except Exception as e:
            logging.error(f"Veri çekme hatası: {str(e)}")
        return self.data

    def calculate_technical_indicators(self):
        if self.data is None:
            raise ValueError("Önce veri çekilmelidir.")
        
        logging.info("Teknik göstergeler hesaplanıyor.")
        # Trend Göstergeleri
        self.data['SMA_20'] = ta.trend.sma_indicator(self.data['Close'], window=20)
        self.data['EMA_20'] = ta.trend.ema_indicator(self.data['Close'], window=20)
        self.data['MACD'] = ta.trend.macd_diff(self.data['Close'])
        self.data['ADX'] = ta.trend.adx(self.data['High'], self.data['Low'], self.data['Close'])

        # Momentum Göstergeleri
        self.data['RSI'] = ta.momentum.rsi(self.data['Close'])
        self.data['Stoch_K'] = ta.momentum.stoch(self.data['High'], self.data['Low'], self.data['Close'])
        self.data['Stoch_D'] = ta.momentum.stoch_signal(self.data['High'], self.data['Low'], self.data['Close'])

        # Volatilite Göstergeleri
        self.data['ATR'] = ta.volatility.average_true_range(self.data['High'], self.data['Low'], self.data['Close'])
        self.data['Bollinger_High'] = ta.volatility.bollinger_hband(self.data['Close'])
        self.data['Bollinger_Low'] = ta.volatility.bollinger_lband(self.data['Close'])

        # Hacim Göstergeleri
        self.data['OBV'] = ta.volume.on_balance_volume(self.data['Close'], self.data['Volume'])
        self.data['CMF'] = ta.volume.chaikin_money_flow(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'])

        logging.info("Teknik göstergeler başarıyla hesaplandı.")
        return self.data

    # def perform_sentiment_analysis(self):
        logging.info(f"{self.symbol} için duygu analizi yapılıyor.")
        # Bu fonksiyon, haber makalelerini veya sosyal medya gönderilerini analiz ederek
        # piyasa duyarlılığını ölçebilir. Basit bir örnek olarak, bir API'den veri çekiyoruz.
        try:
            url = f"https://newsapi.org/v2/everything?q={self.symbol}&apiKey=YOUR_API_KEY"
            response = requests.get(url)
            news = response.json()['articles']
            
            # Basit bir duyarlılık puanı hesaplama (gerçek uygulamada NLP kullanılabilir)
            sentiment_score = sum([1 if 'positive' in article['description'].lower() else -1 if 'negative' in article['description'].lower() else 0 for article in news])
            logging.info(f"Duyarlılık puanı hesaplandı: {sentiment_score}")
            return sentiment_score
        except Exception as e:
            logging.error(f"Duygu analizi hatası: {str(e)}")
            return None

    def analyze_market_cycles(self):
        if self.data is None:
            raise ValueError("Önce veri çekilmelidir.")
        
        logging.info("Piyasa döngüleri analiz ediliyor.")
        # Basit bir döngü analizi: Fiyatın 200 günlük hareketli ortalamanın üstünde veya altında olması
        self.data['SMA_200'] = ta.trend.sma_indicator(self.data['Close'], window=200)
        self.data['Market_Cycle'] = np.where(self.data['Close'] > self.data['SMA_200'], 'Bull', 'Bear')
        
        current_cycle = self.data['Market_Cycle'].iloc[-1]
        cycle_duration = len(self.data[self.data['Market_Cycle'] == current_cycle].tail())
        
        logging.info(f"Mevcut piyasa döngüsü: {current_cycle}, Süre: {cycle_duration} gün")
        return current_cycle, cycle_duration

    def perform_correlation_analysis(self, other_symbols):
        logging.info(f"{self.symbol} ve diğer semboller arasında korelasyon analizi yapılıyor.")
        all_data = pd.DataFrame()
        all_data[self.symbol] = self.data['Close']
        
        for symbol in other_symbols:
            try:
                other_data = yf.download(symbol, start=self.data.index[0], end=self.data.index[-1])
                all_data[symbol] = other_data['Close']
            except Exception as e:
                logging.error(f"{symbol} için veri çekme hatası: {str(e)}")
        
        correlation_matrix = all_data.pct_change().corr()
        logging.info("Korelasyon matrisi oluşturuldu.")
        return correlation_matrix

    def visualize_data(self):
        if self.data is None:
            raise ValueError("Önce veri çekilmelidir.")
        
        logging.info("Veri görselleştiriliyor.")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 20))

        # Fiyat ve hareketli ortalamalar
        ax1.plot(self.data.index, self.data['Close'], label='Kapanış Fiyatı')
        ax1.plot(self.data.index, self.data['SMA_20'], label='20 Günlük SMA')
        ax1.plot(self.data.index, self.data['EMA_20'], label='20 Günlük EMA')
        ax1.set_title('Fiyat ve Hareketli Ortalamalar')
        ax1.legend()

        # RSI
        ax2.plot(self.data.index, self.data['RSI'])
        ax2.axhline(y=70, color='r', linestyle='--')
        ax2.axhline(y=30, color='g', linestyle='--')
        ax2.set_title('RSI Göstergesi')

        # Hacim
        ax3.bar(self.data.index, self.data['Volume'])
        ax3.set_title('İşlem Hacmi')

        plt.tight_layout()
        plt.show()
        logging.info("Veri görselleştirme tamamlandı.")

    def identify_support_resistance(self, window=20):
        logging.info("Destek ve direnç seviyeleri belirleniyor.")
        self.data['Support'] = self.data['Low'].rolling(window=window, center=True).min()
        self.data['Resistance'] = self.data['High'].rolling(window=window, center=True).max()
        
        current_support = self.data['Support'].iloc[-1]
        current_resistance = self.data['Resistance'].iloc[-1]
        
        logging.info(f"Mevcut destek seviyesi: {current_support}, Mevcut direnç seviyesi: {current_resistance}")
        return current_support, current_resistance

    def perform_fundamental_analysis(self):
        logging.info(f"{self.symbol} için temel analiz yapılıyor.")
        try:
            stock = yf.Ticker(self.symbol)
            info = stock.info
            
            pe_ratio = info.get('trailingPE', None)
            peg_ratio = info.get('pegRatio', None)
            dividend_yield = info.get('dividendYield', None)
            book_value = info.get('bookValue', None)
            
            fundamental_data = {
                'P/E Ratio': pe_ratio,
                'PEG Ratio': peg_ratio,
                'Dividend Yield': dividend_yield,
                'Book Value': book_value
            }
            
            logging.info(f"Temel analiz verileri elde edildi: {fundamental_data}")
            return fundamental_data
        except Exception as e:
            logging.error(f"Temel analiz hatası: {str(e)}")
            return None

# Kullanım örneği
if __name__ == "__main__":
    analyzer = MarketAnalyzer("AAPL")
    data = analyzer.fetch_data("2020-01-01", "2023-01-01")
    analyzer.calculate_technical_indicators()
    analyzer.visualize_data()
    
    sentiment_score = analyzer.perform_sentiment_analysis()
    print(f"Duyarlılık Puanı: {sentiment_score}")
    
    market_cycle, cycle_duration = analyzer.analyze_market_cycles()
    print(f"Piyasa Döngüsü: {market_cycle}, Süre: {cycle_duration} gün")
    
    correlation_matrix = analyzer.perform_correlation_analysis(["GOOGL", "MSFT", "AMZN"])
    print("Korelasyon Matrisi:")
    print(correlation_matrix)
    
    support, resistance = analyzer.identify_support_resistance()
    print(f"Destek Seviyesi: {support}, Direnç Seviyesi: {resistance}")
    
    fundamental_data = analyzer.perform_fundamental_analysis()
    print("Temel Analiz Verileri:")
    print(fundamental_data)