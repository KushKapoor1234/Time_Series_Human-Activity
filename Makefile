# Simple Makefile for HAR_Project (ensure you run `make` from project root)

PYTHON ?= python
SEED ?= 42
MODEL ?= cnn
EPOCHS ?= 80
BATCH ?= 64
VAL_SPLIT ?= 0.2
ART_DIR ?= artifacts
SAMPLE_IDX ?= 0

MODEL_CLEAN := $(strip $(MODEL))
ART_DIR_CLEAN := $(strip $(ART_DIR))
MODEL_PATH ?= $(ART_DIR_CLEAN)/final_$(MODEL_CLEAN).keras

# Ensure project root (cwd) is on PYTHONPATH when make runs commands
export PYTHONPATH := $(PWD)

.PHONY: all setup train eval loso explain robustness export clean deepclean check

all: train eval explain robustness export

setup:
	@echo ">>> Installing deps (optional)"
	@$(PYTHON) -m pip install --upgrade pip setuptools wheel
	@-$(PYTHON) -m pip uninstall -y protobuf tf2onnx || true
	@$(PYTHON) -m pip install "tensorflow==2.17.*" "protobuf>=4.24,<5" "numpy<2" pandas scikit-learn matplotlib
	@-$(PYTHON) -m pip install "tf2onnx>=1.17.0,<2" onnx onnxruntime || true

train:
	@echo ">>> Training model=$(MODEL_CLEAN)"
	@TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 $(PYTHON) scripts/train.py --model $(MODEL_CLEAN) --epochs $(EPOCHS) --batch $(BATCH) --seed $(SEED) --val_split $(VAL_SPLIT) --art_dir $(ART_DIR_CLEAN)

eval:
	@echo ">>> Evaluating MODEL_PATH=$(MODEL_PATH)"
	@TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 $(PYTHON) scripts/eval.py --model_path "$(MODEL_PATH)" --seed $(SEED) --art_dir $(ART_DIR_CLEAN)

loso:
	@echo ">>> LOSO evaluation MODEL_PATH=$(MODEL_PATH)"
	@TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 $(PYTHON) scripts/eval.py --model_path "$(MODEL_PATH)" --seed $(SEED) --art_dir $(ART_DIR_CLEAN) --loso

explain:
	@echo ">>> Explainability sample_idx=$(SAMPLE_IDX)"
	@TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 $(PYTHON) scripts/explain.py --model_path "$(MODEL_PATH)" --seed $(SEED) --art_dir $(ART_DIR_CLEAN)/explanations --sample_idx $(SAMPLE_IDX)

robustness:
	@echo ">>> Robustness sweeps"
	@TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 $(PYTHON) scripts/benchmark.py --model_path "$(MODEL_PATH)" --seed $(SEED) --art_dir $(ART_DIR_CLEAN)/robustness

export:
	@echo ">>> Exporting (ONNX/TFLite) + latency"
	@TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 $(PYTHON) scripts/export.py --model_path "$(MODEL_PATH)" --seed $(SEED) --art_dir $(ART_DIR_CLEAN)/export

check:
	@echo ">>> Running tests"
	@$(PYTHON) -m pytest -q

clean:
	@echo ">>> Removing intermediate plots/reports (keeping models)"
	@find "$(ART_DIR_CLEAN)" -type f \( -name "*.png" -o -name "*.txt" -o -name "*.csv" \) -print0 2>/dev/null | xargs -0 -r rm -f

deepclean:
	@echo ">>> Removing all artifacts"
	@rm -rf "$(ART_DIR_CLEAN)"
