import torch
import numpy as np
from PIL import Image


def read_hwc(path: str) -> torch.Tensor:
    image = Image.open(path)
    np_image = np.array(image.convert('RGB'))
    return torch.from_numpy(np_image)


def write_hwc(image: torch.Tensor, path: str):
    Image.fromarray(image.cpu().numpy()).save(path)



