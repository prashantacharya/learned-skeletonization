import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import ImageFilter
import numpy as np
import torch

def apply_random_distortions(image):
    # image is a PIL Image

    # Random Gaussian Blur
    if random.random() < 0.3:
        radius = random.uniform(0.1, 2.0)
        image = image.filter(ImageFilter.GaussianBlur(radius))

    # Random Additive Gaussian Noise
    if random.random() < 0.3:
        image = add_gaussian_noise(image)

    # Random slight Brightness Change
    if random.random() < 0.3:
        factor = random.uniform(0.7, 1.3)
        image = TF.adjust_brightness(image, factor)

    # Random slight Contrast Change
    if random.random() < 0.3:
        factor = random.uniform(0.7, 1.3)
        image = TF.adjust_contrast(image, factor)

    return image

def add_gaussian_noise(image, mean=0.0, std=0.05):
    tensor_img = TF.to_tensor(image)
    noise = torch.randn_like(tensor_img) * std + mean
    noisy_img = tensor_img + noise
    noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
    return TF.to_pil_image(noisy_img)
