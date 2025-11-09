# src/eval.py
import os, argparse, numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from . import data as D
from .utils import set_seeds, ensure_dir, zscore_train_stats, zscore_apply, plot_confusion_matrix, reliability_diagram

LABELS = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']

def save_pr_roc(y_true, probs, out_dir):
    ensure_dir(out_dir)
    y_true_oh = np.eye(len(LABELS))[y_true]
    # PR
    for c in range(len(LABELS)):
        p, r, _ = precision_recall_curve(y_true_oh[:,c], probs[:,c])
        np.savetxt(os.path.join(out_dir, f"pr_class{c}.csv"), np.c_[r,p], delimiter=',', header="recall,precision", comments='')
    # ROC
    for c in range(len(LABELS)):
        fpr, tpr, _ = roc_curve(y_true_oh[:,c], probs[:,c])
        np.savetxt(os.path.join(out_dir, f"roc_class{c}.csv"), np.c_[fpr,tpr], delimiter=',', header="fpr,tpr", comments='')

def eval_main(args=None):
    p=argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--art_dir', default='artifacts')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--loso', action='store_true')
    a=p.parse_args(args) if args is not None else p.parse_args()

    set_seeds(a.seed); ensure_dir(a.art_dir)

    if a.loso:
        # LOSO over train split (UCI HAR has pre-defined train/test subjects; here we show LOSO over train)
        Xtr,ytr,str_,_,_,_ = D.get_train_test()
        mu, sd = zscore_train_stats(Xtr)
        res=[]
        for sid, tr_idx, te_idx in D.loso_indices(str_):
            X_tr, y_tr = Xtr[tr_idx], ytr[tr_idx]
            X_te, y_te = Xtr[te_idx], ytr[te_idx]
            X_tr = zscore_apply(X_tr, mu, sd); X_te = zscore_apply(X_te, mu, sd)
            model = tf.keras.models.load_model(a.model_path, compile=True)
            y_prob = model.predict(X_te, verbose=0)
            y_pred = np.argmax(y_prob, axis=1)
            acc = (y_pred==y_te).mean()
            res.append([int(sid), float(acc)])
        np.savetxt(os.path.join(a.art_dir, "loso_summary.csv"), np.array(res), delimiter=',', header="subject,accuracy", comments='')
        print("Saved LOSO summary.")
        return

    # Standard test eval
    Xtr,ytr,_,Xte,yte,_ = D.get_train_test()
    mu, sd = zscore_train_stats(Xtr)
    Xte = zscore_apply(Xte, mu, sd)
    model = tf.keras.models.load_model(a.model_path, compile=True)
    probs = model.predict(Xte, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    rep = classification_report(yte, y_pred, target_names=LABELS, digits=4)
    print(rep)
    with open(os.path.join(a.art_dir, "eval_report.txt"), "w") as f: f.write(rep)

    cm = confusion_matrix(yte, y_pred)
    plot_confusion_matrix(cm, LABELS, os.path.join(a.art_dir,"cm_raw.png"), normalize=False)
    plot_confusion_matrix(cm, LABELS, os.path.join(a.art_dir,"cm_norm.png"), normalize=True)
    save_pr_roc(yte, probs, os.path.join(a.art_dir, "curves"))
    _, _, _, ece = reliability_diagram(probs, yte, n_bins=15)
    with open(os.path.join(a.art_dir, "calibration.txt"), "w") as f: f.write(f"ECE: {ece:.6f}\n")
    print("Artifacts saved in", a.art_dir)

if __name__=='__main__':
    eval_main()
