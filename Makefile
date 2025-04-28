# Makefile

# Variables
PYTHON=python3
TRAIN_SCRIPT=src/train.py

# Targets
train:
	$(PYTHON) $(TRAIN_SCRIPT)

clean:
	rm -f best_model.pth

help:
	@echo "Makefile options:"
	@echo "  make train    - Run the training script"
	@echo "  make clean    - Remove saved model"
	@echo "  make help     - Show this help message"

