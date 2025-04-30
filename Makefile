# Makefile

# Variables
PYTHON=python3
TRAIN_SCRIPT=src/train.py

# Targets
train:
	$(PYTHON) $(TRAIN_SCRIPT)

infer:
	$(PYTHON) src/infer.py

clean:
	rm -f exported_models/model_*

help:
	@echo "Makefile options:"
	@echo "  make train    - Run the training script"
	@echo "  make clean    - Remove saved model"
	@echo "  make help     - Show this help message"

