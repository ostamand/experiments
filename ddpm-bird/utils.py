import sys
from pathlib import Path
from random import randint

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
from torchvision.transforms import v2


def generate_random_samples(
    model, noise_scheduler, n: int, image_size=32, seed: int = None, pg: bool = True
):
    seed = seed if seed is not None else randint(0, sys.maxsize)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = torch.randn(n, 3, image_size, image_size).to(device)
    for t in tqdm(noise_scheduler.timesteps, disable= not pg):
        with torch.no_grad():
            residual = model(sample, t).sample
        sample = noise_scheduler.step(residual, t, sample).prev_sample
    return sample


def generate_image_from_samples(x, resize=None):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    if resize:
        x = v2.Resize((resize, resize))(x)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im
