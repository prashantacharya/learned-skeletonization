import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.augmentations import apply_random_distortions


class SkeletonizationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.label_dir = label_dir
        self.target_transform = target_transform

        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.label_filenames = sorted(os.listdir(self.label_dir))

        assert len(self.image_filenames) == len(
            self.label_filenames
        ), "Mismatch between number of images and labels!"

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        image = Image.open(img_path).convert("L")
        label = Image.open(label_path).convert("L")  # assuming label is grayscale (1 channel)

        # Apply distortions only to the input image, NOT to label!
        image = apply_random_distortions(image)

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = transforms.ToTensor()(label)

        return image, label
