import os
import argparse
import subprocess
from pathlib import Path
import json
import math
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator

from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from diffusers import DDPMScheduler, UNet2DModel
import wandb

from utils import generate_random_samples, generate_image_from_samples

load_dotenv()
logger = logging.getLogger(__name__)


# from: https://github.com/huggingface/diffusers/blob/05be622b1c152bf11026f97f083bb1b989231aec/src/diffusers/optimization.py#L154-L185
def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class BirdDataset(Dataset):
    """Combines train, test & valid data together, ignores labels for now"""

    def __init__(self, data_dir: Path, image_size: int = 128, subset: int = -1):
        self.data_dir = data_dir
        self.df = pd.read_csv(data_dir / "birds.csv")
        self.subset = subset

        # cleanup: folder issues
        auklet = self.df["labels"].map(lambda x: x.replace(" ", "") == "PARAKETTAKULET")
        self.df.loc[(auklet) & (self.df["data set"] == "train"), "filepaths"] = self.df[
            auklet
        ]["filepaths"].map(lambda x: x.replace("PARAKETT  AKULET", "PARAKETT  AUKLET"))
        self.df.loc[(auklet) & (self.df["data set"] == "test"), "filepaths"] = self.df[
            auklet
        ]["filepaths"].map(lambda x: x.replace("PARAKETT  AKULET", "PARAKETT  AUKLET"))
        self.df.loc[(auklet) & (self.df["data set"] == "valid"), "filepaths"] = self.df[
            auklet
        ]["filepaths"].map(lambda x: x.replace("PARAKETT  AKULET", "PARAKETT AUKLET"))

        self.transforms = v2.Compose(
            [
                v2.Resize((image_size, image_size)),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.5], [0.5]),
            ]
        )

        self.idx = None
        if subset > 0:
            assert subset < len(self.df)  # make sure subset is actually smaller
            self.idx = torch.randint(0, len(self.df), (subset,))

    def __len__(self):
        if self.subset == -1:
            return len(self.df)
        return self.subset

    def __getitem__(self, idx):
        if self.subset > 0:
            idx = self.idx[idx].item()
        img_path = self.df.iloc[idx].filepaths
        image = read_image(self.data_dir / img_path)
        image = self.transforms(image)
        return image


def build_model(image_size: int = 64, dp: float = 0.0):
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        dropout=dp,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def fit(
    model: UNet2DModel,
    dataset: Dataset,
    noise_scheduler: DDPMScheduler,
    args: argparse.Namespace
):
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device
    model.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    steps_per_epoch = int(len(dataloader) / args.gradient_accumulation_steps) #!
    lr_scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=int(steps_per_epoch * args.epochs * 0.33),
        num_training_steps= steps_per_epoch * args.epochs,
    )

    lossf = torch.nn.MSELoss()
    losses = []

    model, opt, dataloader, lr_scheduler = accelerator.prepare(
        model, opt, dataloader, lr_scheduler
    )

    log_each_batch = max(len(dataloader) // 5, 1)
    for epoch in range(args.epochs):
        for batch_i, x in enumerate(dataloader):
            with accelerator.accumulate(model):
                bs = x.shape[0]

                noise = torch.randn(x.shape).to(device)
                timesteps = torch.randint(0, args.num_train_timesteps, (bs,)).long().to(device)

                noisy_images = noise_scheduler.add_noise(x, noise, timesteps)

                noise_preds = model(noisy_images, timesteps).sample

                with accelerator.autocast():
                    loss = lossf(noise_preds, noise)

                losses.append(loss.item())

                accelerator.backward(loss)
                opt.step()
                lr_scheduler.step()
                opt.zero_grad()

            if args.batch_pg and (batch_i + 1) % log_each_batch == 0:
                logger.info(
                    f"status: epoch {epoch+1}, batch {batch_i+1:.2f}/{len(dataloader)}"
                )

        # log epoch loss
        loss_avg = sum(losses[-len(dataloader) :]) / len(dataloader)

        # generate some images
        model.eval()
        samples = generate_random_samples(
            model, noise_scheduler, n=4, image_size=args.image_size, seed=args.seed, pg=False
        )
        img = generate_image_from_samples(samples)
        img.save(Path(args.out_dir) / f"sample_{epoch+1}.jpg")
        model.train()

        # current lr
        last_lr = lr_scheduler.get_last_lr()[0]

        wandb.log(
            {
                "loss": loss_avg,
                "sample": wandb.Image(img),
                "lr": last_lr,
            }
        )

        logger.info(f"epoch {epoch+1}, loss {loss_avg}, lr {last_lr}")

    return losses


def save_outs(
    out_dir: Path, config: dict[str, any], model: UNet2DModel, train_losses: list
):
    out = {"config": config, "train_losses": train_losses}
    if not out_dir.exists():
        os.makedirs(out_dir)
    torch.save(model.state_dict(), out_dir / "state_dict_model.pt")
    with open(out_dir / "out.json", "w") as f:
        json.dump(out, f)


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # make sure output folder exits
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        os.makedirs(out_dir)

    # check data
    data_dir: Path = Path("./data/birds")
    if not data_dir.exists():
        # get the data from kaggle
        subprocess.run(
            "kaggle datasets download -d gpiosenka/100-bird-species",
            shell=True,
            executable="/bin/bash",
        )
        subprocess.run(
            "mkdir -p data/birds; unzip -q -d ./data/birds 100-bird-species.zip",
            shell=True,
            executable="/bin/bash",
        )

    # check if we can go over subset of dataset
    dataloader = DataLoader(
        BirdDataset(data_dir, image_size=32, subset=1000), batch_size=32, shuffle=True
    )
    for _ in tqdm(dataloader):
        pass

    # get ready for training, create model, dataset & scheduler
    model = build_model(image_size=args.image_size, dp=args.dp)

    dataset = BirdDataset(
        data_dir,
        image_size=args.image_size,
        subset=args.subset,  # will go over subset of all images
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps, beta_schedule="squaredcos_cap_v2"
    )

    # populate configs
    config = vars(args)
    config["train_steps"] = math.ceil(len(dataset) / args.batch_size / args.gradient_accumulation_steps) * args.epochs

    # wandb init
    wandb.init(
        project=f"ddpm-bird-{args.image_size}px",
        config=config,
    )

    # train model
    train_losses = fit(model, dataset, noise_scheduler, args)

    # save outputs
    save_outs(out_dir, config, model, train_losses)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--subset", type=int, default=-1)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-pg", action="store_true")
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dp", type=float, default=0.0)
    parser.add_argument("--mixed-precision", type=str, default="no")  # fp16
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args = parser.parse_args()
    logger.info(f"running with: {vars(args)}")
    return args


""" 
References:

- BIRDS 525 SPECIES- IMAGE CLASSIFICATION: https://www.kaggle.com/datasets/gpiosenka/100-bird-species
    - images are 224x224
- https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb
- https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py
- Denoising Diffusion Probabilistic Models: https://arxiv.org/pdf/2006.11239
    - Our 32 × 32 models use four feature map resolutions (32 × 32 to 4 × 4), and our 256 × 256 models use six
    - All models have two convolutional residual blocks
        per resolution level and self-attention blocks at the 16 × 16 resolution between the convolutional
        blocks [6]
    - We set the dropout rate on CIFAR10 to 0.1 by sweeping over the values {0.1, 0.2, 0.3, 0.4}.
    - We trained on CelebA-HQ for 0.5M steps, LSUN Bedroom for 2.4M steps, LSUN Cat for 1.8M steps, and LSUN Church for 1.2M steps. The larger LSUN Bedroom model was trained for 1.15M steps.
    - We used random horizontal flips during training for CIFAR10;
    - We set the learning rate to 2 × 10−4 without any sweeping, and we lowered it to 2 × 10−5 for the 256 × 256 images
    - We set the batch size to 128 for CIFAR10 and 64 for larger images. We did not sweep over these values.
    - We used EMA on model parameters with a decay factor of 0.9999. We did not sweep over this value.

    0.5e6 / 30,000 / 64 = approx. 260 epochs?

Setup:

- wandb login

Examples:

run test training on subset of data, 32px
    python ddpm-bird/train_simple.py --epochs 5 --subset 1000 --out out/test

run 5 epochs, all data: 
    python ddpm-bird/train_simple.py --epochs 5 --out out/ddpm-bird/epochs-5

run 30 epochs, all data, 64px
    python ddpm-bird/train_simple.py --epochs 30 --image-size 64 --batch-size 64 --out out/ddpm-bird/64px-epochs-30 --lr 1e-4

run 30 epochs, all data, 128px
    python ddpm-bird/train_simple.py --epochs 30 --image-size 128 --batch-size 16 --gradient-accumulation-steps 2 --out out/ddpm-bird/128px-epochs-30 --lr 1e-4

run 30 epochs, all data, 224px
    python ddpm-bird/train_simple.py --epochs 1 --image-size 224 --batch-size 8 --gradient-accumulation-steps 4 --out out/ddpm-bird/224px-epochs-1 --lr 1e-4
"""
if __name__ == "__main__":
    args = parse_args()
    main(args)
