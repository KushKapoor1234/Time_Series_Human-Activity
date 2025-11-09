# src/models/rnn.py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

def build_lstm(n_steps, n_feat, n_cls, lr=1e-3, units=128):
    m = Sequential([
        LSTM(units, input_shape=(n_steps,n_feat), return_sequences=False),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_cls, activation='softmax')
    ])
    m.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr), metrics=['accuracy'])
    return m

def build_gru(n_steps, n_feat, n_cls, lr=1e-3, units=128):
    m = Sequential([
        GRU(units, input_shape=(n_steps,n_feat), return_sequences=False),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_cls, activation='softmax')
    ])
    m.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr), metrics=['accuracy'])
    return m
