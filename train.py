
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
drive.mount('/content/drive')

import sys
sys.path.append("/content/drive/MyDrive/reface")

!pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

import torch
print(torch.__version__)
import numpy as np
print(np.__version__)

!pip uninstall -y torch torchvision torchaudio numpy
!pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
!pip install numpy==1.26.4

import os

aug_before_dir = "/content/drive/MyDrive/reface/augmented_before"
aug_after_dir = "/content/drive/MyDrive/reface/augmented_after"


for fname in os.listdir(aug_before_dir):
    if fname.endswith("_b.jpg"):
        old_path = os.path.join(aug_before_dir, fname)
        idx = fname.split("_")[0]
        new_name = f"{int(idx):06d}_b.jpg"
        new_path = os.path.join(aug_before_dir, new_name)
        os.rename(old_path, new_path)


for fname in os.listdir(aug_after_dir):
    if fname.endswith("_a.jpg"):
        old_path = os.path.join(aug_after_dir, fname)
        idx = fname.split("_")[0]
        new_name = f"{int(idx):06d}_a.jpg"
        new_path = os.path.join(aug_after_dir, new_name)
        os.rename(old_path, new_path)


# %cd /content/drive/MyDrive/reface/
import sys
sys.path.append("/content/drive/MyDrive/reface")



import numpy as np
import cv2
import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
from dataset import PairedFaceDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import os
from insightface.app import FaceAnalysis
from torchvision.models import vgg16
from torchvision import transforms


torch.backends.cudnn.benchmark = True






app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def compute_identity_loss(fake_img, real_img):
    fake_np = fake_img.permute(0, 2, 3, 1).detach().cpu().numpy()
    real_np = real_img.permute(0, 2, 3, 1).detach().cpu().numpy()
    embeddings_fake = []
    embeddings_real = []
    for f_img, r_img in zip(fake_np, real_np):
        f_img = (f_img * 255).clip(0,255).astype(np.uint8)
        r_img = (r_img * 255).clip(0,255).astype(np.uint8)
        f_img = cv2.cvtColor(f_img, cv2.COLOR_RGB2BGR)
        r_img = cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR)

        # Resize to ArcFace expected size
        f_img = cv2.resize(f_img, (112, 112))
        r_img = cv2.resize(r_img, (112, 112))

        try:
            f_embedding = app.get(f_img)[0]['embedding']
            r_embedding = app.get(r_img)[0]['embedding']
        except IndexError:
            continue  # skip if face not detected in either image
        embeddings_fake.append(f_embedding)
        embeddings_real.append(r_embedding)
        print("InsightFace result:", app.get(f_img))
    if not embeddings_fake:
        return torch.tensor(0.0, device=fake_img.device)

    embeddings_fake.append(f_embedding)
    embeddings_real.append(r_embedding)

    embeddings_fake = torch.tensor(embeddings_fake, device=fake_img.device)
    embeddings_real = torch.tensor(embeddings_real, device=real_img.device)

    loss = F.mse_loss(embeddings_fake, embeddings_real)
    return loss

class VGGPerceptualLoss(nn.Module):
  def __init__(self):
      super().__init__()
      vgg = vgg16(pretrained=True).features[:16].eval()
      for param in vgg.parameters():
          param.requires_grad = False
      self.vgg = vgg.to(config.DEVICE)
      self.transform=transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

  def forward(self, fake_img, real_img):
      fake = self._preprocess(fake_img)
      real = self._preprocess(real_img)
      return nn.functional.l1_loss(self.vgg(fake), self.vgg(real))

  def _preprocess(self, img):
      
      return torch.stack([self.transform(im) for im in img])



vgg_loss = VGGPerceptualLoss().to(config.DEVICE)

def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):

        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)


        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            ID = compute_identity_loss(y_fake, x) * 10

            PERC=vgg_loss(y_fake,y)*1
            G_loss = G_fake_loss + L1 + ID + PERC


        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            print(f"G: {G_loss.item():.4f}, L1: {L1.item():.4f}, ID: {ID.item():.4f}, PERC: {PERC.item():.4f}")
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    before_original = config.ORIGINAL_BEFORE_DIR
    before_augmented = config.AUGMENTED_BEFORE_DIR

    # Combine original + augmented filenames
    original_files = sorted([f for f in os.listdir(before_original) if f.endswith(".jpg")])
    augmented_files = sorted([f for f in os.listdir(before_augmented) if f.endswith(".jpg")])
    all_filenames = original_files + augmented_files

    train_filenames, val_filenames = train_test_split(all_filenames, test_size=0.1, random_state=42)

    before_dirs = [config.ORIGINAL_BEFORE_DIR, config.AUGMENTED_BEFORE_DIR]
    after_dirs = [config.ORIGINAL_AFTER_DIR, config.AUGMENTED_AFTER_DIR]

    # Initialize models and optimizers
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)

    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    # Datasets
    train_dataset = PairedFaceDataset(before_dirs, after_dirs, train_filenames, transform=config.both_transform)
    val_dataset = PairedFaceDataset(before_dirs, after_dirs, val_filenames, transform=config.both_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()

#evaluation


import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from math import log10
from glob import glob
from tqdm import tqdm

# --- CONFIG ---
eval_folder = "/content/drive/MyDrive/reface/evaluation"
final_epoch = 443  # <-- Change if needed
save_results_path = "/content/drive/MyDrive/reface/eval_results.txt"

# --- METRICS ---
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    # Ensure images are at least 7x7
    if min(img1.shape[0], img1.shape[1]) < 7:
        raise ValueError("Image too small for SSIM")

    return ssim(img1, img2, data_range=img2.max() - img2.min(), channel_axis=-1)

# --- EVALUATION LOOP ---
gen_paths = sorted(glob(os.path.join(eval_folder, f"y_gen_{final_epoch}_*.png")))

psnr_list = []
ssim_list = []

for gen_path in tqdm(gen_paths):
    gen_filename = os.path.basename(gen_path)
    index = gen_filename.split("_")[-1].split(".")[0]  # get index like '93'

    # Find corresponding label file
    label_pattern = os.path.join(eval_folder, f"label_*_{index}.png")
    label_candidates = sorted(glob(label_pattern))
    if not label_candidates:
        print(f"[!] No label found for {gen_filename}")
        continue
    label_path = label_candidates[-1]  # pick latest if multiple

    # Load images
    gen_img = cv2.imread(gen_path)
    label_img = cv2.imread(label_path)

    if gen_img.shape != label_img.shape:
        print(f"[!] Shape mismatch for index {index}")
        continue

    psnr_val = calculate_psnr(gen_img, label_img)
    ssim_val = calculate_ssim(gen_img, label_img)

    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

# --- RESULTS ---
avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)

print(f"\n Evaluation Complete — Final Epoch: {final_epoch}")
print(f"Avg PSNR: {avg_psnr:.2f}")
print(f"Avg SSIM: {avg_ssim:.4f}")

with open(save_results_path, "w") as f:
    f.write(f"Evaluation Results for Epoch {final_epoch}\n")
    f.write(f"Avg PSNR: {avg_psnr:.2f}\n")
    f.write(f"Avg SSIM: {avg_ssim:.4f}\n")
