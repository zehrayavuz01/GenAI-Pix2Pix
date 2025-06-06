

from google.colab import drive
drive.mount('/content/drive')

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ORIGINAL_BEFORE_DIR = "/content/drive/MyDrive/reface/before_resized"
ORIGINAL_AFTER_DIR = "/content/drive/MyDrive/reface/after_resized"

AUGMENTED_BEFORE_DIR = "/content/drive/MyDrive/reface/augmented_before"
AUGMENTED_AFTER_DIR = "/content/drive/MyDrive/reface/augmented_after"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
IDENTITY_LAMBDA = 10
PERCEPTUAL_LAMBDA = 1

both_transform = A.Compose([
    A.Resize(256, 256),  # optional if already resized
    A.Normalize(mean=[0.5]*3, std=[0.5]*3, max_pixel_value=255.0),
    ToTensorV2()
], additional_targets={"image0": "image"})
