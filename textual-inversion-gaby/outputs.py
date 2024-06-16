import json
from pathlib import Path
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import os
import torch
from configs import TextualInversionConfig


def save(tokenizer, text_encoder, out_dir: Path):
    text_encoder.save_pretrained(out_dir / "text_encoder")
    tokenizer.save_pretrained(out_dir / "tokenizer")


def load(out_dir: Path):
    with open(out_dir / "config.json", "r") as f:
        config = json.load(f)
    tokenizer = CLIPTokenizer.from_pretrained(out_dir, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(out_dir, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config["model_name"], subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config["model_name"], subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(
        config["model_name"], subfolder="scheduler"
    )
    return tokenizer, text_encoder, vae, unet, scheduler


def save_all_images(images, out_dir: Path, prefix=""):
    if not out_dir.exists():
        os.makedirs(out_dir)
    for i, image in enumerate(images):
        pil_image = Image.fromarray(image)
        with open(out_dir / f"{prefix}_{i}.png", "wb") as f:
            pil_image.save(f)


def save_embedding(text_encoder, config: TextualInversionConfig, out_dir: Path, suffix: str = ""):
    weights = text_encoder.get_input_embeddings().weight.data
    embedding = {
        "string_to_token": {"*": weights.shape[0]},
        "string_to_param": {"*": weights[-1][None, :].detach().cpu()},
        "name": config.name,
        "step": config.train.train_steps,
        "sd_checkpoint": None,
        "sd_checkpoint_name": config.sd_checkpoint_name,
    }
    out_file_name = config.name
    if suffix:
        out_file_name += f"-{suffix}"
    torch.save(embedding, out_dir / f"{out_file_name}.pt")
