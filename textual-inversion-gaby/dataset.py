from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import torch
from transformers import CLIPTokenizer

"""
prompt_templates = [
    "a photo of a <>.",
    "a rendering of a <>.",
    "a cropped photo of the <>.",
    "the photo of a <>.",
    "a photo of a clean <>.",
    "a photo of a dirty <>.",
    "a dark photo of the <>.",
    "a photo of my <>.",
    "a photo of the cool <>.",
    "a close-up photo of a <>.",
    "a bright photo of the <>.",
    "a cropped photo of a <>.",
    "a photo of the <>.",
    "a good photo of the <>.",
    "a photo of one <>.",
    "a close-up photo of the <>.",
    "a rendition of the <>.",
    "a photo of the clean <>.",
    "a rendition of a <>.",
    "a photo of a nice <>.",
    "a good photo of a <>.",
    "a photo of the nice <>.",
    "a photo of the small <>.",
    "a photo of the weird <>.",
    "a photo of the large <>.",
    "a photo of a cool <>.",
    "a photo of a small <>.",
]
"""

prompt_templates = [
    "a photo of a <>",
    "a rendering of a <>",
    "a cropped photo of the <>",
    "the photo of a <>",
    "a photo of a clean <>",
    "a photo of a dirty <>",
    "a dark photo of the <>",
    "a photo of my <>",
    "a photo of the cool <>",
    "a close-up photo of a <>",
    "a bright photo of the <>",
    "a cropped photo of a <>",
    "a photo of the <>",
    "a good photo of the <>",
    "a photo of one <>",
    "a close-up photo of the <>",
    "a rendition of the <>",
    "a photo of the clean <>",
    "a rendition of a <>",
    "a photo of a nice <>",
    "a good photo of a <>",
    "a photo of the nice <>",
    "a photo of the small <>",
    "a photo of the weird <>",
    "a photo of the large <>",
    "a photo of a cool <>",
    "a photo of a small <>",
    "an illustration of a <>",
    "a rendering of a <>",
    "a cropped photo of the <>",
    "the photo of a <>",
    "an illustration of a clean <>",
    "an illustration of a dirty <>",
    "a dark photo of the <>",
    "an illustration of my <>",
    "an illustration of the cool <>",
    "a close-up photo of a <>",
    "a bright photo of the <>",
    "a cropped photo of a <>",
    "an illustration of the <>",
    "a good photo of the <>",
    "an illustration of one <>",
    "a close-up photo of the <>",
    "a rendition of the <>",
    "an illustration of the clean <>",
    "a rendition of a <>",
    "an illustration of a nice <>",
    "a good photo of a <>",
    "an illustration of the nice <>",
    "an illustration of the small <>",
    "an illustration of the weird <>",
    "an illustration of the large <>",
    "an illustration of a cool <>",
    "an illustration of a small <>",
    "a depiction of a <>",
    "a rendering of a <>",
    "a cropped photo of the <>",
    "the photo of a <>",
    "a depiction of a clean <>",
    "a depiction of a dirty <>",
    "a dark photo of the <>",
    "a depiction of my <>",
    "a depiction of the cool <>",
    "a close-up photo of a <>",
    "a bright photo of the <>",
    "a cropped photo of a <>",
    "a depiction of the <>",
    "a good photo of the <>",
    "a depiction of one <>",
    "a close-up photo of the <>",
    "a rendition of the <>",
    "a depiction of the clean <>",
    "a rendition of a <>",
    "a depiction of a nice <>",
    "a good photo of a <>",
    "a depiction of the nice <>",
    "a depiction of the small <>",
    "a depiction of the weird <>",
    "a depiction of the large <>",
    "a depiction of a cool <>",
    "a depiction of a small <>",
]


class GabyDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        tokenizer: CLIPTokenizer,
        placeholder_token: str,
        repeats: int = 4,
    ):
        self.data_dir = data_dir
        self.repeats = repeats
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token

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
        text = random.choice(prompt_templates).replace("<>", self.placeholder_token)
        ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )["input_ids"]

        return dict(text_ids=ids, imgs=img)
