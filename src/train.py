# src/train.py
import os, argparse, numpy as np, tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from . import data as D
from .utils import set_seeds, ensure_dir, zscore_train_stats, zscore_apply, plot_training_curves

from .models.cnn1d import build_cnn1d
from .models.tcn import build_tcn
from .models.rnn import build_lstm, build_gru
from .models.heads import build_multitask_cnn

LABELS = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']

def get_model(name, n_steps, n_feat, n_cls, lr, multitask=False, forecast_steps=10):
    if multitask:
        return build_multitask_cnn(n_steps, n_feat, n_cls, forecast_steps=forecast_steps, lr=lr)
    name = name.lower()
    if name=='cnn': return build_cnn1d(n_steps, n_feat, n_cls, lr)
    if name=='tcn': return build_tcn(n_steps, n_feat, n_cls, lr)
    if name=='lstm': return build_lstm(n_steps, n_feat, n_cls, lr)
    if name=='gru': return build_gru(n_steps, n_feat, n_cls, lr)
    raise ValueError(f"Unknown model {name}")

# Replace train_main in src/train.py with this implementation
def train_main(args=None):
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='cnn', choices=['cnn','tcn','lstm','gru'])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--art_dir', default='artifacts')
    p.add_argument('--multitask', action='store_true')
    p.add_argument('--forecast_steps', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val_split', type=float, default=0.2)
    p.add_argument('--mixup', action='store_true', help='Enable mixup augmentation')
    p.add_argument('--mixup_alpha', type=float, default=0.2)
    p.add_argument('--augment', action='store_true', help='Enable augmentations in DataGenerator')
    a = p.parse_args(args) if args is not None else p.parse_args()

    set_seeds(a.seed)
    ensure_dir(a.art_dir)

    # Load data
    Xtr, ytr, _, Xte, yte, _ = D.get_train_test()
    mu, sd = zscore_train_stats(Xtr)

    # deterministic train/val split
    n_train = Xtr.shape[0]
    val_n = int(n_train * a.val_split)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(n_train)
    val_idx = perm[:val_n]; train_idx = perm[val_n:]

    X_train_full = Xtr[train_idx]; y_train_full = ytr[train_idx]
    X_val = Xtr[val_idx]; y_val = ytr[val_idx]

    n_steps, n_feat = Xtr.shape[1], Xtr.shape[2]
    n_cls = int(np.max(ytr) + 1)

    # DataGenerators
    from .data import DataGenerator
    train_gen = DataGenerator(
        X_train_full, y_train_full,
        batch_size=a.batch,
        shuffle=True,
        mu=mu, sd=sd,
        augment=a.augment,
        mixup_alpha=(a.mixup_alpha if a.mixup else 0.0)
    )
    val_gen = DataGenerator(
        X_val, y_val,
        batch_size=a.batch,
        shuffle=False,
        mu=mu, sd=sd,
        augment=False,
        mixup_alpha=0.0
    )

    # Build model
    model = get_model(a.model, n_steps, n_feat, n_cls, a.lr, multitask=False)
    model.summary()

    # Callbacks and paths
    ckpt = os.path.join(a.art_dir, f"best_{a.model}.keras")
    final = os.path.join(a.art_dir, f"final_{a.model}.keras")
    curves = os.path.join(a.art_dir, f"curves_{a.model}.png")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]

    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.arange(n_cls)
    cw = compute_class_weight('balanced', classes=classes, y=y_train_full)
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    print("Class weights:", class_weight)

    # Fit using generators (NO use_multiprocessing/workers kwargs)
    steps_per_epoch = int(np.ceil(len(train_gen)))
    validation_steps = int(np.ceil(len(val_gen)))
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=a.epochs,
        callbacks=callbacks,
        verbose=2,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        class_weight=class_weight
    )

    # Save final model and curves
    model.save(final)
    plot_training_curves(history, curves)
    print("Saved:", ckpt, final, curves)


if __name__ == '__main__':
    train_main()
