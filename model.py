import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt
import logging
import os

# Attention Layer sınıfı
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        u_it = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a_it = tf.nn.softmax(tf.tensordot(u_it, self.u, axes=1), axis=1)
        weighted_input = x * tf.expand_dims(a_it, -1)
        return tf.reduce_sum(weighted_input, axis=1)

# Verilerin ölçeklendirilmesi (scaling)
def scale_data(df, scaler=None):
    logging.info("Verilerin ölçeklendirilmesi başlatılıyor.")
    features = ['close', 'bb_mavg', 'bb_high', 'bb_low', 'macd', 'macd_signal', 'macd_diff',
                'rsi', 'adx', 'atr', 'stoch_k', 'stoch_d', 'roc', 'sma_30', 'ema_30',
                'ichimoku_a', 'ichimoku_b', 'obv', 'cmf']
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[features])
        logging.info("Scaler başarıyla fit edildi.")
    else:
        scaled_data = scaler.transform(df[features])
        logging.info("Scaler mevcut veriye uygulandı.")
    return scaled_data, scaler

# LSTM için verinin hazırlanması
def prepare_lstm_data(data, lookback=60):
    logging.info("LSTM verisinin hazırlanması başlatılıyor.")
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])  # Sadece kapanış fiyatını tahmin ediyoruz
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], data.shape[1]))
    logging.info(f"LSTM verisi hazırlandı. X shape: {X.shape}, y shape: {y.shape}")
    return X, y

# LSTM ve Attention mekanizması ile model oluşturma
def create_lstm_model(input_shape):
    logging.info("LSTM ve Attention mekanizması ile geliştirilmiş modelin oluşturulması başlatılıyor.")
    
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(AttentionLayer())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    logging.info("LSTM ve Attention mekanizması ile geliştirilmiş model başarıyla oluşturuldu.")
    
    return model

# Keras Tuner ile hiperparametre optimizasyonu yapılması
def build_model(hp, input_shape):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units_1', min_value=32, max_value=128, step=32),
        return_sequences=True,
        input_shape=input_shape
    ))
    model.add(Dropout(rate=hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
    model.add(LSTM(
        units=hp.Int('units_2', min_value=32, max_value=128, step=32),
        return_sequences=False
    ))
    model.add(Dropout(rate=hp.Float('dropout_2', 0.1, 0.5, step=0.1)))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Modelin eğitilmesi
def train_model(model, X_train, y_train, X_test, y_test):
    logging.info("Model eğitimi başlatılıyor.")
    
    # Checkpoint dosyasını belirleme
    checkpoint_filepath = 'model_checkpoint.h5'
    
    # Model checkpoint ve early stopping callback'lerini ayarlama
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,  # Sadece model ağırlıklarını kaydeder
        monitor='val_loss',
        mode='min',
        save_best_only=False  # Tüm epoch'ları kaydet
    )
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Checkpoint'ten devam etme
    if os.path.exists(checkpoint_filepath):
        logging.info("Önceki model checkpoint'i yüklendi.")
        model.load_weights(checkpoint_filepath)
    
    # Model eğitimi
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, input_shape=(X_train.shape[1], X_train.shape[2])),
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        directory='lstm_tuning',
        project_name='lstm_optimization'
    )

    tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
    logging.info("Hiperparametre optimizasyonu tamamlandı.")

    best_model = tuner.get_best_models(num_models=1)[0]
    logging.info("En iyi model seçildi.")

    best_model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[checkpoint_callback, early_stopping])
    
    logging.info("Model eğitimi tamamlandı.")
    return best_model
