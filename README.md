# HAR_Project — Time-Series Human Activity Recognition (UCI HAR)

> **One-line:** Reproducible end-to-end time-series pipeline using raw smartphone inertial signals (9 channels × 128 timesteps) with a residual 1D-CNN, on-the-fly augmentations, explainability, robustness tests, and export to ONNX/TFLite.

---

## Project status / headline results

* **Dataset:** UCI Human Activity Recognition Using Smartphones (raw inertial signals). Link: [https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
* **Final model:** Residual 1D-CNN with label smoothing and mixup/augmentation training.
* **Test accuracy (official test split):** **94.44%**
* **Per-class performance (classification report):**

| Activity           | Precision | Recall |   F1-score | Support |
| ------------------ | --------: | -----: | ---------: | ------: |
| WALKING            |    1.0000 | 1.0000 |     1.0000 |     496 |
| WALKING_UPSTAIRS   |    0.9467 | 0.9427 |     0.9447 |     471 |
| WALKING_DOWNSTAIRS |    0.9459 | 1.0000 |     0.9722 |     420 |
| SITTING            |    0.8996 | 0.8391 |     0.8683 |     491 |
| STANDING           |    0.8763 | 0.9192 |     0.8972 |     532 |
| LAYING             |    1.0000 | 0.9721 |     0.9858 |     537 |
| **accuracy**       |           |        | **0.9444** |    2947 |
| **macro avg**      |    0.9448 | 0.9455 |     0.9447 |    2947 |
| **weighted avg**   |    0.9447 | 0.9444 |     0.9441 |    2947 |

> Artifacts and logs were written to the `artifacts/` folder during training and evaluation.

---

## Contents of this README

1. Project overview
2. Directory / file structure
3. Requirements & setup
4. Quickstart (Make + scripts)
5. Data loading & preprocessing
6. Model architectures and training choices
7. Augmentations and data generator
8. Evaluation, explainability, and robustness
9. Export & deployment
10. Artifacts produced
11. Reproducibility notes
12. How to cite / acknowledgements
13. Short CV bullets

---

## 1) Project overview

This project is an end-to-end pipeline to train and evaluate deep learning models on the UCI HAR dataset using raw time-series (per-sample windows of 128 timesteps × 9 channels). The focus is on: reproducible experiments, realistic augmentations, model interpretability, and deployment readiness (ONNX/TFLite). The pipeline supports multiple architectures (CNN / TCN / LSTM / GRU) and includes: training, evaluation, explainability, robustness sweeps, and export.

## 2) Directory / file structure

```
HAR_Project/
├─ UCI HAR Dataset/                  # dataset (unzipped)
├─ Makefile                          # high-level targets: train, eval, explain, robustness, export
├─ run_all.sh                        # fallback runner if Make isn't available
├─ har_model.py                      # one-file demo (standalone)
├─ requirements.txt                  # (optional) pinned packages
├─ scripts/
│  ├─ train.py
│  ├─ eval.py
│  ├─ explain.py
│  ├─ benchmark.py
│  └─ export.py
├─ src/
│  ├─ __init__.py
│  ├─ data.py                         # loaders, augmentations, DataGenerator
│  ├─ utils.py                        # helper funcs and plotting
│  ├─ train.py                        # training pipeline (CLI-driven)
│  ├─ eval.py                         # evaluation helpers and CLI
│  ├─ explain.py                      # saliency/perturbation explainers
│  ├─ robustness.py                   # noise/mask/shift sweeps
│  ├─ export.py                       # ONNX/TFLite conversions & latency
│  └─ models/
│     ├─ __init__.py
│     ├─ cnn1d.py                     # residual 1D-CNN (final)
│     ├─ tcn.py                       # TCN baseline
│     ├─ rnn.py                       # LSTM/GRU baselines
│     └─ heads.py                     # multitask / forecasting head
├─ artifacts/                         # model artifacts and plots generated at runtime
└─ tests/
   ├─ test_data_loading.py
   └─ test_splits_and_shapes.py
```

---

## 3) Requirements & setup

Recommended: a Python 3.10–3.12 virtual environment. TensorFlow 2.17 is used and is compatible with the code here.

Install (example):

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install "tensorflow==2.17.*" "protobuf>=4.24,<5" numpy pandas scikit-learn matplotlib
# optional exporters
pip install "tf2onnx>=1.17.0,<2" onnx onnxruntime
```

**Notes**: `tf2onnx` versions earlier than 1.17 expect older protobuf; the setup above is tested to avoid protobuf conflicts.

---

## 4) Quickstart (Make + scripts)

**From project root (where `Makefile` is):**

```bash
# Install deps (optional)
make setup

# Train the default CNN (writes artifacts to ./artifacts)
make train

# Evaluate the saved model (uses artifacts/final_cnn.keras by default)
make eval

# Run explainability on a sample (sample index 0)
make explain SAMPLE_IDX=0

# Run robustness sweeps (noise/missingness/window-shift)
make robustness

# Export to ONNX/TFLite and run a simple latency benchmark
make export

# Or run everything (fallback)
./run_all.sh
```

If your system's `make` has issues, use the `run_all.sh` fallback which runs the same scripts.

---

## 5) Data loading & preprocessing

* We load the 9 raw inertial signal files from `UCI HAR Dataset/{train,test}/Inertial Signals/` and stack them into arrays shaped `(n_samples, 128, 9)`.
* Labels are loaded from `y_train.txt` / `y_test.txt` and converted to 0..5.
* **Normalization**: z-score normalization is computed from the training set (`mu`, `sd`) and applied to train/val/test.
* **Splits**: Training uses a deterministic seed-based train/validation split. LOSO (leave-one-subject-out) evaluation is supported using `subject_train.txt`/`subject_test.txt` if needed.

---

## 6) Model architectures & training choices

**Final model (residual 1D-CNN)**

* Input: 128 time steps × 9 features.
* Residual blocks: 64 → 128 → 256 filters, global average pooling, dense head.
* Optimizer: Adam, LR scheduling by `ReduceLROnPlateau`, early-stopping, and `ModelCheckpoint`.
* Loss: categorical crossentropy with **label smoothing** (0.1).
* Training improvements: class weighting, deterministic seed, model saving in `.keras` format.

**Baselines available:** TCN (causal dilated conv stack), LSTM/GRU, and simple classical baselines (feature extraction + RandomForest) were implemented as scripts for quick comparison.

---

## 7) Augmentations & DataGenerator

On-the-fly augmentations implemented in `src/data.py` and used by the `DataGenerator`:

* **Jitter:** add Gaussian noise to each sample.
* **Scaling:** multiplicative factor per-channel.
* **Magnitude warp:** smooth, piecewise scaling across time.
* **Time shift:** circular shift of the window.
* **Random mask:** randomly zero-out small fractions of the signal.
* **Mixup:** optional mixup augmentation (controlled by `--mixup` and `--mixup_alpha`).

These augmentations are applied stochastically per-sample at training time and improved generalization significantly (observed gain vs baseline training without augmentation).

---

## 8) Evaluation, explainability & robustness

**Evaluation**

* Official test set performance printed and saved to `artifacts/classification_report.txt`.
* Confusion matrices (raw & normalized), PR/ROC CSVs, and calibration/ECE saved to `artifacts/`.

**Explainability**

* Saliency maps (gradient-based) and sliding-window perturbation importance tests are available in `src/explain.py` and write artifacts to `artifacts/explanations/`.

**Robustness**

* Scripts sweep noise amplitude, random missingness, and window shift to measure model degradation under perturbation. Results are saved under `artifacts/robustness/` as CSVs and plots.

---

## 9) Export & deployment

* Export to **TFLite** (fp32 and dynamic-range quantized int8) is available via `src/export.py`.
* Optional **ONNX** export is attempted when `tf2onnx` is installed; fallback writes an error to `onnx_error.txt` if conversion fails.
* Latency benchmark (batch=1 inference) using Keras is recorded and saved to `artifacts/export/inference_latency_ms.csv`.

---

## 10) Artifacts produced (example)

`artifacts/` contains (automatically written by scripts):

* `best_<model>.keras` — best checkpoint by validation loss
* `final_<model>.keras` — final saved model
* `curves_<model>.png` — training/validation loss & accuracy
* `classification_report.txt` — test classification report (precision/recall/f1)
* `cm_raw.png`, `cm_norm.png` — confusion matrices
* `curves/roc_class*.csv`, `curves/pr_class*.csv` — per-class PR/ROC data
* `explanations/` — saliency CSVs, perturbation importance CSVs
* `robustness/` — `noise_sweep.csv`, `missingness_sweep.csv`, `window_shift.csv`
* `export/` — `model_fp.tflite`, `model_int8.tflite`, `model.onnx` (if success), `inference_latency_ms.csv`

---

## 11) Reproducibility notes

* All runs are deterministic with a fixed seed (`--seed`, default 42) — results may vary slightly due to floating point non-determinism across platforms.
* Model & artifacts saved under `artifacts/` with clear filenames; use the CLI flags or Makefile to reproduce experiments exactly.
* `requirements.txt` (if present) pins TensorFlow and protobuf to avoid installation conflicts.

---

## 12) Troubleshooting

* If you see `ModuleNotFoundError: No module named 'src'` when running scripts, ensure you run from the project root and that `src/__init__.py` and `src/models/__init__.py` exist. Also ensure `PYTHONPATH` includes the project root (Makefile exports it by default).
* If ONNX export fails: check `artifacts/export/onnx_error.txt` for the converter error. Installing a compatible `tf2onnx` (>=1.17) usually fixes the mismatch.
* If you see protobuf version conflicts during `pip` installs, uninstall older `protobuf` and reinstall compatible range: `pip install "protobuf>=4.24,<5"`.
* If `make` shows tab/whitespace issues, use the provided `run_all.sh` instead.

---

## 13) Next steps (suggestions)

* Add a small hyperparameter sweep (grid search) and an experiment CSV logger to record runs.
* Add a lightweight web-demo (Streamlit or Flask) showing live inference and saliency overlays on sample recordings.
* Replace mixup with advanced contrastive pretraining (TS-TCC) for better representation transfer.
