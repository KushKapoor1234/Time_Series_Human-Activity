# src/robustness.py
import os, argparse, numpy as np, tensorflow as tf
from . import data as D
from .utils import set_seeds, ensure_dir, zscore_train_stats, zscore_apply

def sweep_main(args=None):
    p=argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--art_dir', default='artifacts/robustness')
    p.add_argument('--seed', type=int, default=42)
    a=p.parse_args(args) if args is not None else p.parse_args()

    set_seeds(a.seed); ensure_dir(a.art_dir)
    Xtr,ytr,_,Xte,yte,_ = D.get_train_test()
    mu, sd = zscore_train_stats(Xtr); Xte=zscore_apply(Xte, mu, sd)
    model = tf.keras.models.load_model(a.model_path, compile=True)

    # Noise sweep
    noise_sigmas=[0.0, 0.01, 0.02, 0.05, 0.1]
    rows=[]
    for s in noise_sigmas:
        Xn = D.add_gaussian_noise(Xte, sigma=s)
        pred = model.predict(Xn, verbose=0)
        if isinstance(pred,(list,tuple)): pred=pred[0]
        acc = (pred.argmax(axis=1)==yte).mean()
        rows.append(['noise', s, acc])
    np.savetxt(os.path.join(a.art_dir,"noise_sweep.csv"), np.array(rows, dtype=object), fmt='%s', delimiter=',')

    # Missingness sweep
    ps=[0.0,0.05,0.1,0.2,0.3]
    rows=[]
    for p_m in ps:
        Xm = D.random_missingness(Xte, p=p_m)
        pred = model.predict(Xm, verbose=0)
        if isinstance(pred,(list,tuple)): pred=pred[0]
        acc = (pred.argmax(axis=1)==yte).mean()
        rows.append(['missing', p_m, acc])
    np.savetxt(os.path.join(a.art_dir,"missingness_sweep.csv"), np.array(rows, dtype=object), fmt='%s', delimiter=',')

    # Window shift sweep
    shifts=[-16,-8,-4,0,4,8,16]
    rows=[]
    for sh in shifts:
        Xs = D.time_shift(Xte, shift=sh)
        pred = model.predict(Xs, verbose=0)
        if isinstance(pred,(list,tuple)): pred=pred[0]
        acc = (pred.argmax(axis=1)==yte).mean()
        rows.append(['shift', sh, acc])
    np.savetxt(os.path.join(a.art_dir,"window_shift.csv"), np.array(rows, dtype=object), fmt='%s', delimiter=',')
    print("Saved robustness sweeps.")

if __name__=='__main__':
    sweep_main()
