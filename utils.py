# utils.py
import os
import logging
import pandas as pd
import datetime
import joblib
from tensorflow.keras.models import load_model

def save_data(df, filename='data.csv'):
    df.to_csv(filename, index=False)
    logging.info(f"Veriler {filename} olarak kaydedildi.")

def load_data(filename='data.csv'):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logging.info(f"Veriler {filename} dosyasından yüklendi.")
        return df
    return None

def save_model(model, filename='model.h5'):
    model.save(filename)
    logging.info(f"Model {filename} olarak kaydedildi.")

def load_saved_model(filename='model.h5'):
    if os.path.exists(filename):
        model = load_model(filename)
        logging.info(f"Model {filename} dosyasından yüklendi.")
        return model
    return None

def save_scaler(scaler, filename='scaler.joblib'):
    joblib.dump(scaler, filename)
    logging.info(f"Scaler {filename} olarak kaydedildi.")

def load_scaler(filename='scaler.joblib'):
    if os.path.exists(filename):
        scaler = joblib.load(filename)
        logging.info(f"Scaler {filename} dosyasından yüklendi.")
        return scaler
    return None

def save_last_run_date(date):
    with open('last_run_date.txt', 'w') as f:
        f.write(date.strftime('%Y-%m-%d'))
    logging.info(f"Son çalışma tarihi kaydedildi: {date}")

def load_last_run_date():
    if os.path.exists('last_run_date.txt'):
        with open('last_run_date.txt', 'r') as f:
            date_str = f.read().strip()
        return datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return None