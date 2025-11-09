#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
SEED=${SEED:-42}
MODEL=${MODEL:-cnn}
EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-64}
VAL_SPLIT=${VAL_SPLIT:-0.2}
ART_DIR=${ART_DIR:-artifacts}
SAMPLE_IDX=${SAMPLE_IDX:-0}
MODEL_PATH="${ART_DIR}/final_${MODEL}.keras"

# echo ">>> (run_all) Setup - installing pinned deps (skippable)"
# "$PYTHON" -m pip install --upgrade pip setuptools wheel
# optional: comment out below to skip package installation
# "$PYTHON" -m pip uninstall -y protobuf tf2onnx || true
# "$PYTHON" -m pip install "tensorflow==2.17.*" "protobuf>=4.24,<5" "numpy<2" pandas scikit-learn matplotlib
# "$PYTHON" -m pip install "tf2onnx>=1.17.0,<2" onnx onnxruntime || true

echo ">>> (run_all) Training model=${MODEL}"
TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 "$PYTHON" scripts/train.py --model "${MODEL}" --epochs "${EPOCHS}" --batch "${BATCH}" --seed "${SEED}" --val_split "${VAL_SPLIT}" --art_dir "${ART_DIR}"

echo ">>> (run_all) Evaluating model_path=${MODEL_PATH}"
TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 "$PYTHON" scripts/eval.py --model_path "${MODEL_PATH}" --seed "${SEED}" --art_dir "${ART_DIR}"

echo ">>> (run_all) Explainability sample_idx=${SAMPLE_IDX}"
TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 "$PYTHON" scripts/explain.py --model_path "${MODEL_PATH}" --seed "${SEED}" --art_dir "${ART_DIR}/explanations" --sample_idx "${SAMPLE_IDX}"

echo ">>> (run_all) Robustness sweeps"
TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 "$PYTHON" scripts/benchmark.py --model_path "${MODEL_PATH}" --seed "${SEED}" --art_dir "${ART_DIR}/robustness"

echo ">>> (run_all) Exporting"
TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 "$PYTHON" scripts/export.py --model_path "${MODEL_PATH}" --seed "${SEED}" --art_dir "${ART_DIR}/export"

echo ">>> Done. Artifacts in ${ART_DIR}"
