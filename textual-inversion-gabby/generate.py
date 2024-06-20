import torch
from tqdm import tqdm


def generate_images(
    prompt,
    tokenizer,
    scheduler,
    text_encoder,
    unet,
    vae,
    bs=2,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed: int | None = None,
    show_pg: bool = True,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed is not None:
        torch.manual_seed(seed)

    inference_steps = scheduler.config["num_train_timesteps"]
    scheduler.set_timesteps(num_inference_steps)

    text_input = tokenizer(
        [prompt] * bs,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = text_encoder(input_ids=text_input.input_ids.to(device))[0]
    uncond_input = tokenizer(
        [""] * bs,
        padding="max_length",
        max_length=text_input.input_ids.shape[-1],
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (
            bs,
            vae.config.latent_channels,
            unet.config.sample_size,
            unet.config.sample_size,
        )
    ).to(device)
    latents = latents * scheduler.init_noise_sigma

    for i, t in tqdm(
        enumerate(scheduler.timesteps),
        total=len(scheduler.timesteps),
        disable=not show_pg,
    ):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input.half(),
                t,
                encoder_hidden_states=text_embeddings.half(),
            ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        images = vae.decode(latents.half()).sample

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")

    scheduler.set_timesteps(inference_steps)

    return images
