import numpy as np
import torch

from PIL import Image

def save_pair_grid(sample_left, sample_right, out_path):
    # HWC uint8 가정. 크기 다르면 left 기준으로 right를 리사이즈
    h1, w1 = sample_left.shape[:2]
    img_left = Image.fromarray(sample_left)
    img_right = Image.fromarray(sample_right)
    if sample_right.shape[:2] != (h1, w1):
        img_right = img_right.resize((w1, h1), Image.BILINEAR)

    grid = Image.new("RGB", (w1 * 2, h1))
    grid.paste(img_left, (0, 0))
    grid.paste(img_right, (w1, 0))
    grid.save(out_path)