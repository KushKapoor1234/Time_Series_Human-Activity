# src/data.py
import os
import numpy as np
import pandas as pd

DATA_PATH = 'UCI HAR Dataset/'
SIGNALS = [
    "body_acc_x","body_acc_y","body_acc_z",
    "body_gyro_x","body_gyro_y","body_gyro_z",
    "total_acc_x","total_acc_y","total_acc_z"
]

def _load_matrix(path):
    return pd.read_csv(path, header=None, delim_whitespace=True).values

def load_signals(group):
    base = os.path.join(DATA_PATH, group, "Inertial Signals")
    mats = [_load_matrix(os.path.join(base, f"{s}_{group}.txt")) for s in SIGNALS]
    X = np.dstack(mats).astype('float32')  # (N, 128, 9)
    return X

def load_labels(group):
    y = pd.read_csv(os.path.join(DATA_PATH, group, f"y_{group}.txt"), header=None).values.flatten().astype(int) - 1
    return y

def load_subjects(group):
    sub = pd.read_csv(os.path.join(DATA_PATH, group, f"subject_{group}.txt"), header=None).values.flatten().astype(int)
    return sub

def get_train_test():
    Xtr = load_signals('train'); ytr = load_labels('train'); str_ = load_subjects('train')
    Xte = load_signals('test'); yte = load_labels('test'); ste_ = load_subjects('test')
    return Xtr, ytr, str_, Xte, yte, ste_

# ----------------------------
# Augmentations (numpy)
# ----------------------------
def jitter(X, sigma=0.01, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return X + rng.normal(0.0, sigma, size=X.shape).astype('float32')

def scaling(X, sigma=0.1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    factor = rng.normal(1.0, sigma, size=(X.shape[0], 1, X.shape[2])).astype('float32')
    return X * factor

def magnitude_warp(X, sigma=0.2, knot=4, rng=None):
    # simple piecewise linear scaling across time
    rng = np.random.default_rng() if rng is None else rng
    N, T, F = X.shape
    res = X.copy()
    for i in range(N):
        for f in range(F):
            knots = rng.normal(1.0, sigma, size=(knot,))
            xs = np.linspace(0, T-1, num=knot)
            xp = np.arange(T)
            curve = np.interp(xp, xs, knots)
            res[i,:,f] = res[i,:,f] * curve
    return res.astype('float32')

def time_shift_one(X, shift_max=8, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    s = rng.integers(-shift_max, shift_max+1)
    return np.roll(X, shift=s, axis=1)

def random_mask(X, p=0.05, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    mask = rng.random(X.shape) < p
    X2 = X.copy()
    X2[mask] = 0.0
    return X2

# ----------------------------
# Keras Sequence data generator
# ----------------------------
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """
    Numpy -> on-the-fly augmentations DataGenerator for model.fit
    - X: (N, T, F)
    - y: (N,) integer labels
    Supports:
      - z-score normalization (mu, sd from training)
      - jitter, scaling, magnitude warp, time shift, masking
      - optional mixup
    """
    def __init__(self, X, y, batch_size=64, shuffle=True, mu=None, sd=None,
                 augment=True, jitter_p=0.5, scaling_p=0.5, magwarp_p=0.2,
                 shift_p=0.5, mask_p=0.2, mixup_alpha=0.0, rng=None):
        self.X = X
        self.y = y
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.mu = mu
        self.sd = sd
        self.augment = augment
        self.jitter_p = jitter_p
        self.scaling_p = scaling_p
        self.magwarp_p = magwarp_p
        self.shift_p = shift_p
        self.mask_p = mask_p
        self.mixup_alpha = float(mixup_alpha)
        self.rng = np.random.default_rng() if rng is None else rng
        self.indexes = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def _apply_augment(self, batch_X):
        # batch_X shape (B, T, F)
        B = batch_X.shape[0]
        out = batch_X.copy()
        for i in range(B):
            if self.rng.random() < self.jitter_p:
                out[i] = jitter(out[i][None, ...], sigma=0.01, rng=self.rng)[0]
            if self.rng.random() < self.scaling_p:
                out[i] = scaling(out[i][None, ...], sigma=0.1, rng=self.rng)[0]
            if self.rng.random() < self.magwarp_p:
                out[i] = magnitude_warp(out[i][None, ...], sigma=0.2, rng=self.rng)[0]
            if self.rng.random() < self.shift_p:
                out[i] = time_shift_one(out[i], shift_max=8, rng=self.rng)
            if self.rng.random() < self.mask_p:
                out[i] = random_mask(out[i][None, ...], p=0.05, rng=self.rng)[0]
        return out

    def _mixup(self, Xb, yb):
        # Xb: (B,T,F), yb: one-hot (B,C)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=(len(Xb),))
        lam = lam.astype('float32')
        idx = self.rng.permutation(len(Xb))
        X2 = Xb[idx]
        y2 = yb[idx]
        Xm = Xb * lam[:,None,None] + X2 * (1.0 - lam)[:,None,None]
        ym = yb * lam[:,None] + y2 * (1.0 - lam)[:,None]
        return Xm, ym

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        Xb = self.X[batch_idx].astype('float32')
        yb = self.y[batch_idx]
        # normalize if requested
        if (self.mu is not None) and (self.sd is not None):
            Xb = (Xb - self.mu) / (self.sd + 1e-8)
        # augment
        if self.augment:
            Xb = self._apply_augment(Xb)
        # one-hot encode y
        from tensorflow.keras.utils import to_categorical
        nb_classes = int(np.max(self.y) + 1)
        yb_oh = to_categorical(yb, num_classes=nb_classes).astype('float32')
        # mixup if enabled
        if self.mixup_alpha > 0.0:
            Xb, yb_oh = self._mixup(Xb, yb_oh)
        return Xb, yb_oh
