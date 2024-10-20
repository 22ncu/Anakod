import logging
import ccxt
import pandas as pd
import datetime
import os
from sklearn.model_selection import train_test_split
from data import fetch_data, calculate_indicators, detect_candlestick_patterns, calculate_fibonacci_levels, scale_data, prepare_lstm_data
from model import create_lstm_model, train_model
from backtesting import Backtester
from market_analiysis import MarketAnalyzer
from risk_management import RiskManager, PortfolioManager
from utils import save_data, load_data, save_model, load_saved_model, save_scaler, load_scaler, save_last_run_date, load_last_run_date

def main():
    # Loglama ayarları
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Binance API nesnesi oluşturuluyor
    binance = ccxt.binance()

    # Veri çekme ve hazırlama parametreleri
    symbol = 'BCH/USDT'
    timeframe = '1m'
    days = 1

    # Son çalışma tarihini kontrol et
    last_run_date = load_last_run_date()
    current_date = datetime.datetime.now()

    if last_run_date is None or (current_date - last_run_date).days >= 1:
        logging.info(f"Veri çekme işlemi başlatılıyor: {symbol}, Timeframe: {timeframe}, Gün: {days}")
        df = fetch_data(binance, symbol, timeframe, days)
        if df.empty:
            logging.error("Veri çekilemedi. Program sonlandırılıyor.")
            return
        save_data(df)
    else:
        logging.info("Son çalışmadan bu yana 1 gün geçmedi. Mevcut veriler kullanılacak.")
        df = load_data()

    print(df.columns)
    logging.info(f"DataFrame sütunları: {df.columns}")

    # Göstergelerin hesaplanması
    df = calculate_indicators(df)
    df = detect_candlestick_patterns(df)
    df = calculate_fibonacci_levels(df)

    # Veri ölçeklendirme ve LSTM için hazırlama
    scaler = load_scaler()
    if scaler is None:
        scaled_data, scaler = scale_data(df)
        save_scaler(scaler)
    else:
        scaled_data = scaler.transform(df[['close', 'volume', 'open', 'high', 'low']])  # Kullanılan sütunları doğru şekilde belirtin

    X, y = prepare_lstm_data(scaled_data)

    # Veriyi eğitim ve test seti olarak ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model oluşturma ve eğitme
    model = load_saved_model()
    if model is None:
        model = create_lstm_model((X_train.shape[1], X_train.shape[2]))

    trained_model = train_model(model, X_train, y_train, X_test, y_test)
    save_model(trained_model)

    # Backtesting
    initial_balance = 10000
    commission = 0.001
    backtester = Backtester(df, trained_model, scaler, initial_balance, commission)
    results = backtester.run()
    metrics = backtester.calculate_metrics()

    logging.info("Backtesting sonuçları:")
    for key, value in metrics.items():
        logging.info(f"{key}: {value}")

    backtester.plot_results()

    # Market analizi
    analyzer = MarketAnalyzer(symbol, timeframe)
    analyzer.data = df  # Önceden çekilen veriyi kullan
    analyzer.calculate_technical_indicators()
    analyzer.visualize_data()

    sentiment_score = analyzer.perform_sentiment_analysis()
    logging.info(f"Duyarlılık Puanı: {sentiment_score}")

    market_cycle, cycle_duration = analyzer.analyze_market_cycles()
    logging.info(f"Piyasa Döngüsü: {market_cycle}, Süre: {cycle_duration} gün")

    correlation_matrix = analyzer.perform_correlation_analysis(["ETH/USDT", "BTC/USDT"])
    logging.info("Korelasyon Matrisi:")
    logging.info(correlation_matrix)

    support, resistance = analyzer.identify_support_resistance()
    logging.info(f"Destek Seviyesi: {support}, Direnç Seviyesi: {resistance}")

    fundamental_data = analyzer.perform_fundamental_analysis()
    logging.info("Temel Analiz Verileri:")
    logging.info(fundamental_data)

    # Risk yönetimi
    risk_manager = RiskManager(initial_balance)
    portfolio_manager = PortfolioManager([symbol])

    # Örnek risk hesaplamaları
    position_size = risk_manager.calculate_position_size(df['close'].iloc[-1])
    logging.info(f"Önerilen pozisyon büyüklüğü: {position_size}")

    var = risk_manager.calculate_value_at_risk(df['close'].pct_change().dropna())
    logging.info(f"Value at Risk (VaR): {var}")

    # Portföy yönetimi örneği
    portfolio_return = portfolio_manager.calculate_portfolio_return({symbol: df['close'].pct_change().mean()})
    logging.info(f"Portföy getirisi: {portfolio_return}")

    # Son çalışma tarihini kaydet
    save_last_run_date(current_date)

if __name__ == "__main__":
    main()
