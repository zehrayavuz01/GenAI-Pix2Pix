



import torch
import config
from torchvision.utils import save_image
import os

def save_some_examples(gen, val_loader, epoch, folder):
    gen.eval()
    for i, (x, y) in enumerate(val_loader):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        with torch.no_grad():
            y_fake = gen(x)

        for j in range(x.size(0)):
            save_image(x[j] * 0.5 + 0.5, f"{folder}/input_{epoch}_{i*len(x)+j}.png")
            save_image(y[j] * 0.5 + 0.5, f"{folder}/label_{epoch}_{i*len(x)+j}.png")
            save_image(y_fake[j] * 0.5 + 0.5, f"{folder}/y_gen_{epoch}_{i*len(x)+j}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
