PYTHON ?= python
CONFIG ?= config/default.yaml
DATA_DIR ?= ./data
OUTPUT_DIR ?= ./experiments
TARGET_DEVICE ?= pi4
LABEL_SCHEMA ?= 3class
CLIP_SECONDS ?= 3

.PHONY: setup download features train_teacher train_student evaluate_all quantize export profile report lint test smoke clean

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

lint:
	$(PYTHON) -m compileall src

clean:
	rm -rf artifacts outputs __pycache__ */__pycache__

smoke:
	$(PYTHON) -m pytest tests/test_smoke.py -k tiny

test:
	$(PYTHON) -m pytest

download:
	$(PYTHON) -m src.data.speechocean_downloader --config $(CONFIG) --data-dir $(DATA_DIR) --output-dir $(OUTPUT_DIR) --split "train+validation+test"
	$(PYTHON) -m src.data.manifest --config $(CONFIG) --data-dir $(DATA_DIR) --output-dir $(OUTPUT_DIR)

features:
	$(PYTHON) -m src.features.build_features --config $(CONFIG) --data-dir $(DATA_DIR) --output-dir $(OUTPUT_DIR)

train_teacher:
	$(PYTHON) -m src.train --config $(CONFIG) --model mlp_teacher --output-dir $(OUTPUT_DIR)

train_student:
	$(PYTHON) -m src.train --config $(CONFIG) --model mlp_small --teacher-checkpoint $(OUTPUT_DIR)/checkpoints/teacher.pt --output-dir $(OUTPUT_DIR)

evaluate_all:
	$(PYTHON) -m src.evaluate --config $(CONFIG) --output-dir $(OUTPUT_DIR) --snr-sweep "0,5,10,15,20,30"

quantize:
	$(PYTHON) -m src.quantize --config $(CONFIG) --output-dir $(OUTPUT_DIR)

export:
	$(PYTHON) -m src.export --config $(CONFIG) --output-dir $(OUTPUT_DIR)

profile:
	$(PYTHON) -m src.profile_device --config $(CONFIG) --output-dir $(OUTPUT_DIR) --target-device $(TARGET_DEVICE)

report:
	$(PYTHON) -m src.reporting.aggregate --config $(CONFIG) --output-dir $(OUTPUT_DIR)
