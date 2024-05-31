import os
import argparse
import subprocess
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2

from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from diffusers import DDPMScheduler, UNet2DModel
import wandb

from utils import generate_random_samples, generate_image_from_samples

load_dotenv()


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


def build_model(image_size: int = 64):
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def fit(
    model: UNet2DModel,
    dataset: Dataset,
    noise_scheduler: DDPMScheduler,
    epochs: int = 1,
    batch_size: int = 64,
    num_train_timesteps: int = 1000,
    image_size: int = 32,
    lr=4e-4,
    batch_pg=True,
    out_dir = "out",
    seed: int = 42,
    **kwargs
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    log_each_batch = max(len(dataloader) // 5, 1)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()

    losses = []

    model.train()
    for epoch in range(epochs):
        for batch_i, x in enumerate(dataloader):
            x = x.to(device)
            bs = x.shape[0]

            noise = torch.randn(x.shape).to(device)
            timesteps = torch.randint(0, num_train_timesteps, (bs,)).long().to(device)

            noisy_images = noise_scheduler.add_noise(x, noise, timesteps)

            noise_preds = model(noisy_images, timesteps).sample

            loss = lossf(noise_preds, noise)
            losses.append(loss.item())

            loss.backward()
            opt.step()
            opt.zero_grad()

            if batch_pg and (batch_i + 1) % log_each_batch == 0:
                print(f"status: epoch {epoch+1}, batch {batch_i+1}/{len(dataloader)}")

        # log epoch loss
        loss_avg = sum(losses[-len(dataloader) :]) / len(dataloader)
        print(f"done: epoch {epoch+1}, loss {loss_avg}")

        # generate some images
        model.eval()
        samples = generate_random_samples(model, noise_scheduler, n=4, image_size=image_size, seed=seed)
        img = generate_image_from_samples(samples, resize=image_size*2)
        img.save(Path(out_dir) / f"sample_{epoch+1}.jpg")
        model.train()

        wandb.log({
            "loss": loss_avg,
            "sample": wandb.Image(img)
            }
        )

    return losses


def save_outs(
    out_dir: Path, args: dict[str, any], model: UNet2DModel, train_losses: list
):
    out = {"args": args, "train_losses": train_losses}
    if not out_dir.exists():
        os.makedirs(out_dir)
    torch.save(model.state_dict(), out_dir / "state_dict_model.pt")
    with open(out_dir / "out.json", "w") as f:
        json.dump(out, f)


def main(args):
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

    # train
    model = build_model(image_size=args.image_size)

    dataset = BirdDataset(
        data_dir,
        image_size=args.image_size,
        subset=args.subset,  # will go over subset of all images
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps, beta_schedule="squaredcos_cap_v2"
    )

    # wandb
    wandb.init(
        project="ddpm-bird-32px",
        config=vars(args),
    )

    train_losses = fit(
        model,
        dataset,
        noise_scheduler,
        **vars(args)
    )

    # save outputs
    save_outs(out_dir, vars(args), model, train_losses)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--subset", type=int, default=-1)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-pg", action="store_true")
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"running with: {vars(args)}")
    return args


""" 
References:

- BIRDS 525 SPECIES- IMAGE CLASSIFICATION: https://www.kaggle.com/datasets/gpiosenka/100-bird-species
- https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb
- https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py

Setup:

- wandb login

Examples:

run test training sample: 
    python ddpm-bird/train_simple.py --epochs 10 --subset 1000 --out out/test

run 5 epochs all data: 
    python ddpm-bird/train_simple.py --epochs 5 --out out/ddpm-bird/epochs-5

"""
if __name__ == "__main__":
    args = parse_args()
    main(args)
