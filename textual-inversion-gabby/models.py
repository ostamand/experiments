from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
import torch


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


def update_embeddings_for_new_token(
    tokenizer, text_encoder, placeholder_token: str, initializer_token: str
):
    # add new token to tokenizer
    tokenizer.add_tokens(placeholder_token)

    # get the token id of the initializer token we want to use to initialize the new embedding
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    initializer_token_id = token_ids[0]

    # need to resize the input embeddings from the text encoder since we added a new token to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # initialize new embedding with specified token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[-1] = token_embeds[initializer_token_id]
