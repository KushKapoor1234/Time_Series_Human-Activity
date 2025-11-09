# src/utils.py
import os, numpy as np, tensorflow as tf, matplotlib.pyplot as plt, time

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed); tf.random.set_seed(seed)
    os.environ.setdefault('TF_DETERMINISTIC_OPS','1')

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def zscore_train_stats(X):
    mu = X.mean(axis=(0,1), keepdims=True)
    sd = X.std(axis=(0,1), keepdims=True) + 1e-8
    return mu, sd

def zscore_apply(X, mu, sd): return (X - mu) / sd

def plot_training_curves(history, outpath):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history.get('accuracy',[]), label='train')
    plt.plot(history.history.get('val_accuracy',[]), label='val')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(history.history.get('loss',[]), label='train')
    plt.plot(history.history.get('val_loss',[]), label='val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_confusion_matrix(cm, labels, outpath, normalize=False):
    from matplotlib import pyplot as plt
    import numpy as np
    if normalize:
        with np.errstate(invalid='ignore'):
            cm = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(7,6)); im=plt.imshow(cm, cmap='viridis')
    plt.title('Confusion Matrix' + (' (normalized)' if normalize else '')); plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks=np.arange(len(labels)); plt.xticks(ticks, labels, rotation=45, ha='right'); plt.yticks(ticks, labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i,j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            plt.text(j,i,txt, ha='center', va='center', color='white' if val>cm.max()/2 else 'black', fontsize=8)
    plt.ylabel('True'); plt.xlabel('Pred'); plt.tight_layout(); plt.savefig(outpath); plt.close()

def reliability_diagram(probs, y_true, n_bins=10):
    """Returns (bin_acc, bin_conf, bin_count). probs are softmax (N,C), y_true int labels."""
    import numpy as np
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(conf, bins) - 1
    bin_acc, bin_conf, bin_cnt = [], [], []
    for b in range(n_bins):
        m = (idx==b)
        if m.sum()==0: 
            bin_acc.append(0.0); bin_conf.append((bins[b]+bins[b+1])/2); bin_cnt.append(0)
        else:
            bin_acc.append(correct[m].mean()); bin_conf.append(conf[m].mean()); bin_cnt.append(int(m.sum()))
    ece = np.sum([abs(a-c)*cnt for a,c,cnt in zip(bin_acc,bin_conf,bin_cnt)]) / max(1,sum(bin_cnt))
    return np.array(bin_acc), np.array(bin_conf), np.array(bin_cnt), float(ece)

def timeit_inference(model, x, runs=30, batch_size=1):
    import time, numpy as np
    x = x[:batch_size]
    # warmup
    for _ in range(5): model.predict(x, verbose=0)
    t=[]
    for _ in range(runs):
        t0=time.time(); model.predict(x, verbose=0); t.append((time.time()-t0)*1000.0)
    return np.mean(t), np.std(t)
