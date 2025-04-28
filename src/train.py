import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import os

from dataset import SkeletonizationDataset
from models import UNet

# 1. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Hyperparameters 
batch_size = 8
learning_rate = 1e-3
num_epochs = 10
loss_function_name = "BCEWithLogitsLoss"
architecture_name = "UNet (1 input channel, 1 output class)"

# 3. Dataset and DataLoader
train_dataset = SkeletonizationDataset(
    image_dir="dataset/image", label_dir="dataset/labels"
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 4. Model, Loss, Optimizer
model = UNet(n_channels=1, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create directory if not exists
os.makedirs("exported_models", exist_ok=True)

# Generate timestamp for file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"exported_models/model_{timestamp}.pth"
metadata_filename = f"exported_models/model_{timestamp}_metadata.txt"

# 5. Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        outputs = outputs.squeeze(1)  # Shape: (batch_size, H, W)
        labels = labels.squeeze(1)    # Shape: (batch_size, H, W)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

# 6. Evaluation after training
def evaluate_accuracy(loader, model):
    model.eval()
    total_pixels = 0
    correct_pixels = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            correct_pixels += (preds == labels).sum().item()
            total_pixels += torch.numel(preds)

    acc = correct_pixels / total_pixels
    return acc

accuracy = evaluate_accuracy(train_loader, model)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

# 7. Save model
torch.save(model.state_dict(), model_filename)
print(f"✅ Model saved to {model_filename}")

# 8. Save metadata
with open(metadata_filename, "w") as f:
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Architecture: {architecture_name}\n")
    f.write(f"Loss Function: {loss_function_name}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Learning Rate: {learning_rate}\n")
    f.write(f"Epochs: {num_epochs}\n")
    f.write(f"Final Training Accuracy: {accuracy * 100:.2f}%\n")

print(f"✅ Metadata saved to {metadata_filename}")