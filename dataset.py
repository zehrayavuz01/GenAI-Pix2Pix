import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class PairedImageDataset(Dataset):
    def __init__(self, before_dir, after_dir,   both_transform=None,
        transform_input=None,
        transform_target=None,
        filenames=None,):
        self.before_dir = before_dir
        self.after_dir = after_dir
        self.both_transform = both_transform
        self.transform_input = transform_input
        self.transform_target = transform_target

        # Sort file names to match pairs correctly
        self.filenames = filenames if filenames is not None else os.listdir(before_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        before_filename = self.filenames[idx]  # e.g. "77_b.jpg"
        after_filename = before_filename.replace('_b', '_a')  # Adjust to match "77.jpg" or similar

        before_path = os.path.join(self.before_dir, before_filename)
        after_path = os.path.join(self.after_dir, after_filename)

        # Open as numpy arrays for Albumentations
        before_img = np.array(Image.open(before_path).convert("RGB"))
        after_img = np.array(Image.open(after_path).convert("RGB"))

        # Apply joint transform (resize)
        if self.both_transform:
            augmented = self.both_transform(image=before_img, image0=after_img)
            before_img = augmented["image"]
            after_img = augmented["image0"]

        # Apply individual transforms
        if self.transform_input:
            before_img = self.transform_input(image=before_img)["image"]

        if self.transform_target:
            after_img = self.transform_target(image=after_img)["image"]

        return before_img, after_img

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

"""dataset = PairedImageDataset("/Users/zehra/PycharmProjects/MedicalGAN/before_resized", "/Users/zehra/PycharmProjects/MedicalGAN/after_resized", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for before_img, after_img in dataloader:
    print(before_img.shape, after_img.shape)"""





