# src/models/cnn1d.py
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Dense, Dropout, MaxPooling1D

def residual_block(x, filters, kernel_size=3, stride=1):
    y = Conv1D(filters, kernel_size, padding='same', strides=stride)(x)
    y = BatchNormalization()(y); y = ReLU()(y)
    y = Conv1D(filters, kernel_size, padding='same', strides=1)(y)
    y = BatchNormalization()(y)
    if x.shape[-1] != filters:
        x = Conv1D(filters, 1, padding='same')(x)
    out = Add()([x, y])
    out = ReLU()(out)
    return out

def build_cnn1d(n_steps, n_feat, n_cls, lr=1e-3):
    inp = Input(shape=(n_steps, n_feat))
    x = Conv1D(64, 3, padding='same')(inp)
    x = BatchNormalization()(x); x = ReLU()(x)
    x = residual_block(x, 64, 3)
    x = MaxPooling1D(2)(x)
    x = residual_block(x, 128, 3)
    x = MaxPooling1D(2)(x)
    x = residual_block(x, 256, 3)
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(n_cls, activation='softmax')(x)
    m = Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    m.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    return m
