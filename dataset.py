
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


transform = A.Compose([
    A.Resize(256, 256),  # optional if already resized
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
], additional_targets={"image0": "image"})

# Dataset class (clean version, no saving or augmenting)
class PairedFaceDataset(Dataset):
    def __init__(self, before_dirs, after_dirs, filenames, transform=None):
        self.before_dirs = before_dirs  # [before_resized_path, augmented_before_path]
        self.after_dirs = after_dirs    # [after_resized_path, augmented_after_path]
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

    # Detect if it's an augmented image (starts with 10000 or more)
        is_aug = int(filename.split("_")[0]) >= 10000

    # Select folder based on image type
        before_dir = self.before_dirs[1] if is_aug else self.before_dirs[0]
        after_dir = self.after_dirs[1] if is_aug else self.after_dirs[0]

        before_path = os.path.join(before_dir, filename)
        after_path = os.path.join(after_dir, filename.replace("_b", "_a"))

        before = np.array(Image.open(before_path).convert("RGB"))
        after = np.array(Image.open(after_path).convert("RGB"))

        if self.transform:
            augmented = self.transform(image=before, image0=after)
            before = augmented["image"]
            after = augmented["image0"]

        return before, after
