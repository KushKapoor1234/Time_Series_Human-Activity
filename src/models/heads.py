# src/models/heads.py
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense, BatchNormalization, ReLU, Dropout

def encoder_1d(n_steps, n_feat, width=128):
    inp = Input(shape=(n_steps, n_feat))
    x = Conv1D(64,3,padding='same',activation='relu')(inp); x=BatchNormalization()(x)
    x = Conv1D(64,3,padding='same',activation='relu')(x); x=BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(width,3,padding='same',activation='relu')(x); x=BatchNormalization()(x)
    return inp, x

def build_multitask_cnn(n_steps, n_feat, n_cls, forecast_steps=10, lr=1e-3):
    inp, feat = encoder_1d(n_steps, n_feat, width=128)
    # classification head
    x = GlobalAveragePooling1D()(feat)
    x = Dense(128, activation='relu')(x); x = Dropout(0.3)(x)
    cls = Dense(n_cls, activation='softmax', name='cls')(x)
    # simple forecasting head: predict next-step per feature (here: last hidden -> dense)
    # output shape: (batch, forecast_steps * n_feat)
    f = GlobalAveragePooling1D()(feat)
    f = Dense(256, activation='relu')(f)
    fore = Dense(forecast_steps*n_feat, activation='linear', name='forecast')(f)
    m = Model(inp, [cls, fore])
    m.compile(
        loss={'cls':'categorical_crossentropy','forecast':'mse'},
        loss_weights={'cls':1.0,'forecast':0.3},
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics={'cls':'accuracy'}
    )
    return m
