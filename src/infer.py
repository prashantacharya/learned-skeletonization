import os
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from models import UNet
from utils.augmentations import apply_random_distortions
from utils.thinning import iterative_thinning

# --- Config ---
image_dir = "dataset/image"
label_dir = "dataset/labels"
model_version = "model_20250501_181537"
model_path = f"exported_models/{model_version}.pth"
output_dir = "exported_models/inference_results_" + model_version
os.makedirs(output_dir, exist_ok=True)

num_samples = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = UNet(n_channels=1, n_classes=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Get image filenames ---
all_images = sorted(os.listdir(image_dir))
sample_images = random.sample(all_images, num_samples)

# --- Define transform ---
transform = transforms.ToTensor()

for idx, image_name in enumerate(sample_images, 1):
    image_path = os.path.join(image_dir, image_name)

    file_id = image_name.split("_")[1]  # '00293.png'
    label_name = "target_" + file_id
    label_path = os.path.join(label_dir, label_name)

    # Load images
    original = Image.open(image_path).convert("L")
    distorted = apply_random_distortions(original.copy())
    ground_truth = Image.open(label_path).convert("L")

    # Preprocess distorted input
    input_tensor = transform(distorted).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        prediction = (output > 0.5).float()

    # Postprocess
    original_np = transform(original).squeeze().numpy()
    distorted_np = transform(distorted).squeeze().numpy()
    ground_truth_np = transform(ground_truth).squeeze().numpy()
    prediction_np = prediction.squeeze().cpu().numpy()

    # Apply iterative thinning to the prediction
    thinned_prediction_np = iterative_thinning(prediction_np)

    # --- 2x2 Plot ---
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    axs[0, 0].imshow(original_np, cmap="gray")
    axs[0, 0].set_title("Original")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(distorted_np, cmap="gray")
    axs[0, 1].set_title("Distorted Input")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(ground_truth_np, cmap="gray")
    axs[1, 0].set_title("Ground Truth")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(thinned_prediction_np, cmap="gray")
    axs[1, 1].set_title("Thinned Prediction")
    axs[1, 1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"prediction_{idx}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {save_path}")
