# src/models/tcn.py
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Dropout, Add, GlobalAveragePooling1D, Dense

def tcn_block(x, filters, kernel_size=3, dilation=1, dropout=0.2):
    y = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation)(x)
    y = BatchNormalization()(y); y = ReLU()(y); y = Dropout(dropout)(y)
    y = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation)(y)
    y = BatchNormalization()(y)
    if x.shape[-1] != filters:
        x = Conv1D(filters, 1, padding='same')(x)
    y = Add()([x,y]); y = ReLU()(y)
    return y

def build_tcn(n_steps, n_feat, n_cls, lr=1e-3, stacks=2, filters=64):
    inp = Input(shape=(n_steps, n_feat))
    x = inp
    for s in range(stacks):
        for d in [1,2,4,8]:
            x = tcn_block(x, filters=filters, kernel_size=3, dilation=d, dropout=0.2)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(n_cls, activation='softmax')(x)
    m = Model(inp, out)
    m.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr), metrics=['accuracy'])
    return m
