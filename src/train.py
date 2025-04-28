# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SkeletonizationDataset
from models import UNet

# 1. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Hyperparameters
batch_size = 8
learning_rate = 1e-3
num_epochs = 10

# 3. Dataset and DataLoader
train_dataset = SkeletonizationDataset(
    image_dir="dataset/image", label_dir="dataset/labels"
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 4. Model, Loss, Optimizer
model = UNet(n_channels=1, n_classes=1).to(
    device
)  # Assuming RGB input and 1 output class
criterion = nn.BCEWithLogitsLoss()  # For binary segmentation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        labels = labels.squeeze(1)  # Shape: (batch_size, H, W)

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
