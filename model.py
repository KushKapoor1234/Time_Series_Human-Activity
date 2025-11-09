# har_model.py
import os, sys, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

DATA_PATH = 'UCI HAR Dataset/'
SIGNALS = ["body_acc_x","body_acc_y","body_acc_z","body_gyro_x","body_gyro_y","body_gyro_z","total_acc_x","total_acc_y","total_acc_z"]
LABELS = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']

ART_DIR = "artifacts"; os.makedirs(ART_DIR, exist_ok=True)
FINAL = os.path.join(ART_DIR, "final_har_cnn.keras")
CHECKPOINT = os.path.join(ART_DIR, "best_har_cnn.keras")

CURVES = os.path.join(ART_DIR, "training_curves.png")
CM_PNG = os.path.join(ART_DIR, "confusion_matrix.png")
REPORT = os.path.join(ART_DIR, "classification_report.txt")

def set_seeds(seed=42):
    import os, numpy as np, tensorflow as tf
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed); tf.random.set_seed(seed)
set_seeds(42)

def load_signal_data(group):
    base = os.path.join(DATA_PATH, group, "Inertial Signals")
    mats=[]
    for s in SIGNALS:
        fn = os.path.join(base, f"{s}_{group}.txt")
        if not os.path.isfile(fn):
            raise FileNotFoundError(fn)
        mats.append(pd.read_csv(fn, header=None, delim_whitespace=True).values)
    return np.dstack(mats)

def load_labels(group):
    fn = os.path.join(DATA_PATH, group, f"y_{group}.txt")
    y = pd.read_csv(fn, header=None).values.flatten().astype(int) - 1
    return y

def build_model(n_steps, n_feat, n_cls):
    m = Sequential()
    m.add(Conv1D(64, 3, activation='relu', padding='same', input_shape=(n_steps, n_feat)))
    m.add(BatchNormalization()); m.add(Conv1D(64, 3, activation='relu', padding='same')); m.add(BatchNormalization())
    m.add(Dropout(0.4)); m.add(MaxPooling1D(2))
    m.add(Conv1D(128, 3, activation='relu', padding='same')); m.add(BatchNormalization()); m.add(Dropout(0.3))
    m.add(GlobalAveragePooling1D()); m.add(Dense(100, activation='relu')); m.add(Dropout(0.3))
    m.add(Dense(n_cls, activation='softmax'))
    m.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'])
    return m

def plot_curves(h, path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(h.history['accuracy']); plt.plot(h.history['val_accuracy'])
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.legend(['train','val']); plt.grid(True)
    plt.subplot(1,2,2); plt.plot(h.history['loss']); plt.plot(h.history['val_loss'])
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(['train','val']); plt.grid(True)
    plt.tight_layout(); plt.savefig(path); plt.close()

def plot_cm(cm, labels, path):
    import numpy as np, matplotlib.pyplot as plt
    plt.figure(figsize=(7,6))
    im=plt.imshow(cm, cmap='viridis'); plt.title('Confusion Matrix'); plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks=np.arange(len(labels)); plt.xticks(ticks, labels, rotation=45, ha='right'); plt.yticks(ticks, labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,str(cm[i,j]), ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black', fontsize=8)
    plt.ylabel('True'); plt.xlabel('Pred'); plt.tight_layout(); plt.savefig(path); plt.close()

def main():
    print("Loading data...")
    Xtr = load_signal_data('train').astype('float32'); Xte = load_signal_data('test').astype('float32')
    ytr = load_labels('train'); yte = load_labels('test')
    n_cls=len(LABELS)

    # z-score per feature using train stats
    mu = Xtr.mean(axis=(0,1), keepdims=True); sd = Xtr.std(axis=(0,1), keepdims=True)+1e-8
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    ytr_cat = tf.keras.utils.to_categorical(ytr, n_cls); yte_cat = tf.keras.utils.to_categorical(yte, n_cls)

    n, T, F = Xtr.shape
    model = build_model(T, F, n_cls); model.summary()

    cbs=[EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
         ModelCheckpoint(CHECKPOINT, monitor='val_loss', save_best_only=True, verbose=1),
         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)]
    print("Training...")
    hist = model.fit(Xtr, ytr_cat, epochs=50, batch_size=64, validation_split=0.2, callbacks=cbs, verbose=2)
    model.save(FINAL); print("Saved:", FINAL)
    plot_curves(hist, CURVES); print("Saved:", CURVES)

    print("Evaluating...")
    loss, acc = model.evaluate(Xte, yte_cat, verbose=0); print(f"Test Loss {loss:.4f}  Acc {acc:.4f}")
    yp = np.argmax(model.predict(Xte, verbose=0), axis=1)
    rep = classification_report(yte, yp, target_names=LABELS, digits=4); print(rep)
    with open(REPORT, "w") as f:
        f.write(f"Loss: {loss:.6f}\nAcc: {acc:.6f}\n\n{rep}")
    cm = confusion_matrix(yte, yp); plot_cm(cm, LABELS, CM_PNG)
    print("Artifacts in ./artifacts")
if __name__=="__main__":
    try: main()
    except Exception as e:
        print("ERROR:", e); sys.exit(1)
