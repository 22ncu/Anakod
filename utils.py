import os
import logging
import pandas as pd
import datetime
import joblib
from tensorflow.keras.models import load_model
from keras.utils import custom_object_scope
from model import AttentionLayer  # AttentionLayer burada tanımlı olmalı

def save_data(df, filename='data.csv'):
    """
    Veriyi CSV dosyasına kaydeder.
    """
    df.to_csv(filename, index=False)
    logging.info(f"Veriler {filename} olarak kaydedildi.")

def load_data(filename='data.csv'):
    """
    CSV dosyasından veriyi yükler.
    """
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logging.info(f"Veriler {filename} dosyasından yüklendi.")
        return df
    return None

def save_model(model, filename='model.h5'):
    """
    Verilen modeli dosyaya kaydeder.
    """
    try:
        model.save(filename)
        logging.info(f"Model {filename} olarak kaydedildi.")
    except Exception as e:
        logging.error(f"Model kaydedilirken hata oluştu: {e}")

def load_saved_model(filename='model.h5'):
    """
    Kaydedilmiş modeli dosyadan yükler. Eğer AttentionLayer gibi özel katmanlar varsa tanıtır.
    """
    if os.path.exists(filename):
        try:
            # AttentionLayer gibi özel katmanları tanıtmak için custom_object_scope kullanıyoruz
            with custom_object_scope({'AttentionLayer': AttentionLayer}):
                model = load_model(filename)
            logging.info(f"Model {filename} dosyasından yüklendi.")
            return model
        except Exception as e:
            logging.error(f"Model yüklenirken hata oluştu: {e}")
            return None
    else:
        logging.info(f"Model dosyası {filename} bulunamadı.")
        return None

def save_scaler(scaler, filename='scaler.joblib'):
    """
    Scaler'ı dosyaya kaydeder.
    """
    joblib.dump(scaler, filename)
    logging.info(f"Scaler {filename} olarak kaydedildi.")

def load_scaler(filename='scaler.joblib'):
    """
    Dosyadan scaler'ı yükler.
    """
    if os.path.exists(filename):
        scaler = joblib.load(filename)
        logging.info(f"Scaler {filename} dosyasından yüklendi.")
        return scaler
    return None

def save_last_run_date(date):
    """
    Son çalışma tarihini dosyaya kaydeder.
    """
    with open('last_run_date.txt', 'w') as f:
        f.write(date.strftime('%Y-%m-%d'))
    logging.info(f"Son çalışma tarihi kaydedildi: {date}")

def load_last_run_date():
    """
    Dosyadan son çalışma tarihini yükler.
    """
    if os.path.exists('last_run_date.txt'):
        with open('last_run_date.txt', 'r') as f:
            date_str = f.read().strip()
        return datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return None
