from dataset import GabyDataset
import models
import outputs
import generate
import configs
from configs import TextualInversionConfig
import shutil
import torch
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import argparse
import logging
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./embeddings/baseline.yaml")
    parser.add_argument("--out-dir", type=str, default="./out/test")
    args = parser.parse_args()
    return args


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
    dataset = GabyDataset(
        data_dir, tokenizer, config.placeholder_token, repeats=config.train.repeats
    )
    dataloader = DataLoader(dataset, batch_size=config.train.bs, shuffle=True)

    # optimizer for input embeddings
    opt = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(), lr=config.train.lr
    )

    # total number of epochs
    epochs = math.ceil(config.train.train_steps / len(dataloader))
    logging.info(f"Number of epochs: {epochs}")

    # progress bar
    pb = tqdm(range(config.train.train_steps))
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
            grads.data[:-1, :] = grads.data[:-1, :].fill_(0)

            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % config.train.log_each == 0:
                text_encoder.eval()
                image = generate.generate_images(
                    config.evaluation_prompt,
                    tokenizer,
                    scheduler,
                    text_encoder,
                    unet,
                    vae,
                    bs=1,
                    show_pg=False,
                )
                outputs.save_all_images(
                    image, out_dir / "training", prefix=f"step_{global_step}"
                )
                outputs.save_embedding(text_encoder, config, out_dir / "training", suffix=global_step)
                text_encoder.train()

            # update progress bar
            pb.update(1)
            pb.set_postfix(loss=losses[-1])
            if global_step >= config.train.train_steps:
                break


"""
python train.py --config ./embeddings/baseline.yaml --out-dir out/gabby-0.1
"""
if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    config = configs.load(args.config)
    out_dir = Path(args.out_dir)
    data_dir = Path(config.data_dir)

    # cleanup out dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # copy config file to output folder for reference
    shutil.copyfile(args.config, out_dir / "config.yaml")

    # get pretrained models
    tokenizer, text_encoder, vae, unet, scheduler = models.load_pretrained(
        config.sd_checkpoint_name
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = text_encoder.to(device).eval()
    vae = vae.to(device).eval()
    unet = unet.to(device).eval()

    # update embeddings
    models.update_embeddings_for_new_token(
        tokenizer, text_encoder, config.placeholder_token, config.initializer_token
    )

    # save evaluation image before we start the training
    image = generate.generate_images(
        config.evaluation_prompt,
        tokenizer,
        scheduler,
        text_encoder,
        unet,
        vae,
        bs=1,
    )

    outputs.save_all_images(image, out_dir / "training", prefix=f"step_0")

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

    # training completed, save embedding and diffuser model
    outputs.save_embedding(text_encoder, config, out_dir)
    outputs.save(tokenizer, text_encoder, out_dir)
