import numpy as np
import torch
from typing import List, Union, Optional, Tuple
from PIL import Image

ArrayLike = Union[np.ndarray, torch.Tensor, Image.Image]

def _to_pil(img: ArrayLike) -> Image.Image:
    """Convert numpy array, torch tensor, or PIL Image to PIL Image (RGB)."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    
    if isinstance(img, torch.Tensor):
        x = img.detach().cpu()
        # CHW -> HWC
        if x.ndim == 3 and x.shape[0] in (1, 3, 4):
            x = x.permute(1, 2, 0)
        x = x.numpy()
    else:
        x = np.asarray(img)

    # Normalize range
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        if x.min() >= -1.0 and x.max() <= 1.0:
            x = (x + 1.0) * 127.5
        elif x.min() >= 0.0 and x.max() <= 1.0:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)

    # Handle channels
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    if x.shape[-1] > 3:
        x = x[..., :3]
    return Image.fromarray(x, mode="RGB")


def save_image_grid(
    images: List[ArrayLike],
    out_path: str,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    *,
    padding: int = 2,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    resize_mode: str = "first"  # "first" | "max" | "min" | "none"
) -> None:
    """
    Save a grid of images.
    
    Args:
        images: List of images (numpy HWC, torch CHW/HWC, or PIL Images)
        out_path: Output file path
        nrows: Number of rows (optional, auto-computed if None)
        ncols: Number of columns (optional, auto-computed if None)
        padding: Padding between images in pixels
        bg_color: Background color RGB tuple
        resize_mode: How to handle different sizes:
            - "first": resize all to match first image
            - "max": resize all to max width/height
            - "min": resize all to min width/height
            - "none": keep original sizes
    """
    if not images:
        raise ValueError("images list cannot be empty")
    
    # Convert all to PIL
    pil_images = [_to_pil(img) for img in images]
    n_images = len(pil_images)
    
    # Determine grid layout
    if nrows is None and ncols is None:
        ncols = int(np.ceil(np.sqrt(n_images)))
        nrows = int(np.ceil(n_images / ncols))
    elif nrows is None:
        nrows = int(np.ceil(n_images / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n_images / nrows))
    
    # Determine target size
    if resize_mode == "first":
        target_w, target_h = pil_images[0].size
    elif resize_mode == "max":
        target_w = max(img.width for img in pil_images)
        target_h = max(img.height for img in pil_images)
    elif resize_mode == "min":
        target_w = min(img.width for img in pil_images)
        target_h = min(img.height for img in pil_images)
    else:  # "none"
        target_w = max(img.width for img in pil_images)
        target_h = max(img.height for img in pil_images)
    
    # Resize if needed (except for "none" mode)
    if resize_mode != "none":
        pil_images = [img.resize((target_w, target_h), Image.BILINEAR) 
                      if img.size != (target_w, target_h) else img
                      for img in pil_images]
    
    # Create canvas
    canvas_w = ncols * target_w + (ncols - 1) * padding
    canvas_h = nrows * target_h + (nrows - 1) * padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
    
    # Paste images
    for idx, img in enumerate(pil_images):
        row = idx // ncols
        col = idx % ncols
        x = col * (target_w + padding)
        y = row * (target_h + padding)
        canvas.paste(img, (x, y))
    
    canvas.save(out_path)


def save_pair_grid(sample_left, sample_right, out_path):
    """Convenience wrapper for 1x2 grid (backward compatibility)."""
    save_image_grid([sample_left, sample_right], out_path, nrows=1, ncols=2, padding=0)