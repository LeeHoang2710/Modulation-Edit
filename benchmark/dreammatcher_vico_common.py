"""Shared ViCo subject/source-image helpers for DreamMatcher protocols."""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_image


CONCEPT_INPUT_NAMES = {
    "alarm_clock": "clock",
    "black_cat": "black_cat",
    "brown_cat": "brown_cat",
    "brown_dog": "brown_dog",
    "brown_dog2": "brown_dog2",
    "brown_teddybear": "teddybear",
    "cat_statue": "cat-toy",
    "duck_toy": "duck_toy",
    "elephant_statue": "elephant_statue",
    "fat_dog": "fat_dog",
    "monster_toy": "monster_toy",
    "pink_sunglasses": "pink_sunglasses",
    "red_teapot": "red_teapot",
    "red_vase": "vase",
    "tortoise_plushy": "tortoise_plushy",
    "wooden_barn": "barn",
}


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = read_image(str(path))[:3].unsqueeze(0).float() / 127.5 - 1.0
    return F.interpolate(image, (512, 512)).to(device)


def seed_everything(seed: int) -> int:
    """Seed the RNGs used by these runners without importing Lightning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
