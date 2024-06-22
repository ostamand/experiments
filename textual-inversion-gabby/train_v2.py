from pathlib import Path
from PIL import Image
import numpy as np
import random
import torch
from transformers import CLIPTokenizer
from pydantic import BaseModel
import yaml
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionPipeline,
)
import shutil
from torch.utils.data import DataLoader, Dataset
import math
from tqdm.auto import tqdm
import torch.nn.functional as F
import os
from IPython.display import display
import glob
import yaml


class TextualInversionConfig(BaseModel):
    name: str = "gabby01"
    data_dir: str = "../input/gabby-2-512px/gabby-2-512px"
    out_dir: str = "./out"
    sd_checkpoint_name: str = "runwayml/stable-diffusion-v1-5"
    n_vectors: int = 1
    evaluation_prompt: str = "a photo of {}"
    lr: float = 0.0001
    train_steps: int = 500
    bs: int = 8
    log_each: int = 100
    repeats: int = 10
    description: str | None = None


def update_prompt_with_tokens(prompt, n_vectors):
    token_str = ""
    for i in range(n_vectors):
        token_str += f"<token{i}>"
    return prompt.replace("{}", token_str)


"""
prompt_templates = [
    "a photo of a {}.",
    "a rendering of a {}.",
    "a cropped photo of the {}.",
    "the photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a photo of my {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a photo of one {}.",
    "a close-up photo of the {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a good photo of a {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "a photo of the large {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
]
"""

prompt_templates = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
    "an illustration of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "an illustration of a clean {}",
    "an illustration of a dirty {}",
    "a dark photo of the {}",
    "an illustration of my {}",
    "an illustration of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "an illustration of the {}",
    "a good photo of the {}",
    "an illustration of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "an illustration of the clean {}",
    "a rendition of a {}",
    "an illustration of a nice {}",
    "a good photo of a {}",
    "an illustration of the nice {}",
    "an illustration of the small {}",
    "an illustration of the weird {}",
    "an illustration of the large {}",
    "an illustration of a cool {}",
    "an illustration of a small {}",
    "a depiction of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a depiction of a clean {}",
    "a depiction of a dirty {}",
    "a dark photo of the {}",
    "a depiction of my {}",
    "a depiction of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a depiction of the {}",
    "a good photo of the {}",
    "a depiction of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a depiction of the clean {}",
    "a rendition of a {}",
    "a depiction of a nice {}",
    "a good photo of a {}",
    "a depiction of the nice {}",
    "a depiction of the small {}",
    "a depiction of the weird {}",
    "a depiction of the large {}",
    "a depiction of a cool {}",
    "a depiction of a small {}",
]


class GabbyDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        tokenizer: CLIPTokenizer,
        n_vectors: int = 1,
        repeats: int = 4,
    ):
        self.data_dir = data_dir
        self.repeats = repeats
        self.tokenizer = tokenizer
        self.n_vectors = n_vectors

        self.images = [
            f for f in self.data_dir.glob("*") if f.suffix == ".jpg"
        ] * self.repeats

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # get image
        img_path = self.images[idx]
        img = np.asarray(Image.open(img_path).convert("RGB")) / 255 * 2 - 1
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        # extract conditional prompt
        text = update_prompt_with_tokens(
            random.choice(prompt_templates), self.n_vectors
        )
        ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )["input_ids"]

        return dict(text_ids=ids, imgs=img)


def load_pretrained(
    model_name: str,
) -> tuple[
    CLIPTokenizer, CLIPTextModel, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
]:
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.float16
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_name, subfolder="unet", torch_dtype=torch.float16
    )
    scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # freeze all model weights except `(token_embeddings)`
    tm = text_encoder.text_model
    for m in (
        vae,
        unet,
        tm.encoder,
        tm.final_layer_norm,
        tm.embeddings.position_embedding,
    ):
        for p in m.parameters():
            p.requires_grad = False

    return tokenizer, text_encoder, vae, unet, scheduler


def update_embeddings_for_new_token(tokenizer, text_encoder, n_tokens: int):
    # add new tokens to tokenizer
    for i in range(config.n_vectors):
        tokenizer.add_tokens(f"<token{i}>")
    # need to resize the input embeddings from the text encoder since we added a new token to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))


def update_models_dtype(models, dtype):
    for m in models:
        for p in m.parameters():
            p.data = p.to(dtype=dtype)


def train(
    tokenizer,
    text_encoder,
    vae,
    unet,
    scheduler,
    data_dir: Path,
    out_dir: Path,
    config: TextualInversionConfig,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup dataset
    dataset = GabbyDataset(data_dir, tokenizer, config.n_vectors, config.repeats)
    dataloader = DataLoader(dataset, batch_size=config.bs, shuffle=True)

    # optimizer for input embeddings
    opt = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(), lr=config.lr
    )

    # total number of epochs
    epochs = math.ceil(config.train_steps / len(dataloader))
    print(f"Number of epochs: {epochs}")

    # for evaluation
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )

    # progress bar
    pb = tqdm(range(config.train_steps))
    pb.set_description("Steps")

    scaler = torch.cuda.amp.GradScaler()

    losses = []
    global_step = 0
    text_encoder.train()
    for _ in range(epochs):
        for _, batch in enumerate(dataloader):
            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float16,
            ):
                text_ids = batch["text_ids"].to(device)
                imgs = batch["imgs"].to(device)
                bs = imgs.shape[0]

                encoder_hidden_states = text_encoder(text_ids)[0]

                latents = (
                    vae.encode(imgs).latent_dist.sample().detach()
                    * vae.config.scaling_factor
                )

                # add noise to latents
                timesteps = (
                    torch.randint(0, scheduler.config.num_train_timesteps, (bs,))
                    .long()
                    .to(device)
                )
                noise = torch.randn(latents.shape).to(device)
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                # predict noise from noisy latents
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                loss = F.mse_loss(noise_pred, noise)
                losses.append(loss.item())

            scaler.scale(loss).backward()

            # set grads to zero for everything but new token
            grads = text_encoder.get_input_embeddings().weight.grad
            grads.data[: -config.n_vectors, :] = grads.data[
                : -config.n_vectors, :
            ].fill_(0)

            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % config.log_each == 0:
                text_encoder.eval()

                # vae & unet to float32 for training
                update_models_dtype([vae, unet], torch.float32)

                seed = 0
                pipe(
                    config.evaluation_prompt,
                    generator=torch.Generator().manual_seed(seed),
                )[0][0].save(out_dir / f"step_{global_step}.png")
                save_embedding(text_encoder, config, out_dir, suffix=global_step)

                text_encoder.train()
                # vae & unet to float16 for training
                update_models_dtype([vae, unet], torch.float16)

            # update progress bar
            pb.update(1)
            pb.set_postfix(loss=losses[-1])
            if global_step >= config.train_steps:
                break


def save_embedding(
    text_encoder, config: TextualInversionConfig, out_dir: Path, suffix: str = ""
):
    weights = text_encoder.get_input_embeddings().weight.data
    embedding = {
        "string_to_token": {"*": weights.shape[0]},
        "string_to_param": {"*": weights[-config.n_vectors :].detach().cpu()},
        "name": config.name,
        "step": config.train_steps,
        "sd_checkpoint": None,
        "sd_checkpoint_name": config.sd_checkpoint_name,
    }
    out_file_name = config.name
    if suffix:
        out_file_name += f"-{suffix}"
    torch.save(embedding, out_dir / f"{out_file_name}.pt")


def save(tokenizer, text_encoder, out_dir: Path):
    text_encoder.save_pretrained(out_dir / "text_encoder")
    tokenizer.save_pretrained(out_dir / "tokenizer")


if __name__ == "__main__":
    config = TextualInversionConfig()
    config.n_vectors = 2
    config.train_steps = 5000
    config.log_each = 500
    config.lr = 0.005 * config.bs
    config.out_dir = "./out/vectors4"
    config.data_dir: str = "./gabby-2-512px"

    config.evaluation_prompt = update_prompt_with_tokens(
        config.evaluation_prompt, config.n_vectors
    )

    config.description = "same config but now vector size is two, steps=5000"

    out_dir = Path(config.out_dir)
    data_dir = Path(config.data_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # dump config to out folder
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config.dict(), f)

    tokenizer, text_encoder, vae, unet, scheduler = load_pretrained(
        config.sd_checkpoint_name
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = text_encoder.to(device).eval()
    vae = vae.to(device).eval()
    unet = unet.to(device).eval()

    # update embeddings
    update_embeddings_for_new_token(tokenizer, text_encoder, config.n_vectors)

    train(
        tokenizer,
        text_encoder,
        vae,
        unet,
        scheduler,
        data_dir,
        out_dir,
        config,
        device=device,
    )

    save_embedding(text_encoder, config, out_dir)
    save(tokenizer, text_encoder, out_dir)
