"""
SinusSurgeryDataset
-------------------
PyTorch Dataset for UW-Sinus-Surgery-C/L.

Discovers image/label pairs from the data root automatically.
Parses video IDs directly from filenames (S01-S10 for cadaver, L01-L03 for live).
Supports filtering by video ID — enabling video-wise train/test splits.

Returns per sample:
    image         : FloatTensor (3, H, W)  SAM3-normalised
    mask          : FloatTensor (1, H, W)  binary {0., 1.}
    box_prompt    : LongTensor  (4,)       [x1, y1, x2, y2] from GT mask bbox
    point_prompt  : LongTensor  (2,)       [cx, cy] centre-of-mass of GT mask
    text_prompt   : str                    configurable, default "surgical instrument"
    has_instrument: bool                   True if mask contains any instrument pixel
    video_id      : str                    e.g. "S01" or "L02"
    stem          : str                    full filename stem e.g. "S01_10020"
"""

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset
from torchvision import transforms as T


# ── SAM3 normalisation constants ──────────────────────────────────────────────
# SAM3 training pre-processing: pixel_mean=[123.675, 116.28, 103.53],
# pixel_std=[58.395, 57.12, 57.375] on 0-255 images.
# Dividing by 255 → values below, applied after ToTensor() which maps to [0,1].
_SAM_MEAN = (0.485, 0.456, 0.406)
_SAM_STD = (0.229, 0.224, 0.225)

# Regex to parse video ID from filename: "S01_10020.jpg" → "S01", "L02_300.jpg" → "L02"
_VIDEO_ID_RE = re.compile(r"^([SL]\d+)_")


# ── Transform builders ────────────────────────────────────────────────────────


def _image_transform(augment: bool, image_size: int) -> T.Compose:
    base = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=_SAM_MEAN, std=_SAM_STD),
    ]
    if augment:
        aug = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
        return T.Compose(aug + base)
    return T.Compose(base)


def _mask_transform(image_size: int) -> T.Compose:
    # T.ToTensor() divides uint8 by 255, turning mask value 1 → 0.00392.
    # We resize via PIL then convert to tensor manually to preserve {0, 1}.
    resize = T.Resize(
        (image_size, image_size), interpolation=T.InterpolationMode.NEAREST
    )

    def transform(mask_pil: Image.Image) -> torch.Tensor:
        mask_pil = resize(mask_pil)
        return torch.from_numpy(np.array(mask_pil)).unsqueeze(0).float()

    return transform


# ── Prompt generators ─────────────────────────────────────────────────────────


def generate_box_prompt(mask: np.ndarray) -> torch.Tensor:
    """[x1, y1, x2, y2] bounding box of instrument region. Zeros if no instrument."""
    rows = np.any(mask == 1, axis=1)
    cols = np.any(mask == 1, axis=0)
    if not rows.any():
        return torch.zeros(4, dtype=torch.long)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.long)


def generate_point_prompt(mask: np.ndarray) -> torch.Tensor:
    """[cx, cy] centre-of-mass of instrument region. Zeros if no instrument."""
    if mask.max() == 0:
        return torch.zeros(2, dtype=torch.long)
    cy, cx = ndimage.center_of_mass(mask)
    return torch.tensor([int(cx), int(cy)], dtype=torch.long)


# ── Dataset ───────────────────────────────────────────────────────────────────


class SinusSurgeryDataset(Dataset):
    """
    Args:
        data_root   : str | Path
            Root of the UW-Sinus-Surgery-C/L dataset, e.g.
            "data/uw-sinus-surgery-CL"

        split       : "cadaver" | "live"
            Which sub-dataset to load. Maps to the "cadaver/" or "live/"
            subdirectory inside data_root.

        video_ids   : list[str] | None
            Which videos to include, parsed from filenames.
            Cadaver IDs: "S01" … "S10"
            Live IDs:    "L01", "L02", "L03"
            None → include all videos in the split.

        text_prompt : str
            Text prompt string returned with every sample.
            Default: "surgical instrument"

        augment     : bool
            Enable random flips + colour jitter. Use for train split only.

        image_size  : int
            Resize both image and mask to (image_size × image_size).

    Example — cadaver 80/20 split by video:
        train_ds = SinusSurgeryDataset(data_root, "cadaver",
                       video_ids=["S01","S02","S03","S04","S05","S06","S07","S08"],
                       augment=True)
        test_ds  = SinusSurgeryDataset(data_root, "cadaver",
                       video_ids=["S09","S10"])

    Example — live 3-fold CV, fold 1:
        train_ds = SinusSurgeryDataset(data_root, "live",
                       video_ids=["L02","L03"], augment=True)
        test_ds  = SinusSurgeryDataset(data_root, "live",
                       video_ids=["L01"])
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,  # "cadaver" or "live"
        video_ids: list[str] | None = None,
        text_prompt: str = "surgical instrument",
        augment: bool = False,
        image_size: int = 336,
    ):
        assert split in (
            "cadaver",
            "live",
        ), f"split must be 'cadaver' or 'live', got '{split}'"

        split_dir = Path(data_root) / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        assert images_dir.exists(), f"Images dir not found: {images_dir}"
        assert labels_dir.exists(), f"Labels dir not found: {labels_dir}"

        # Discover and filter image/label pairs
        all_images = sorted(images_dir.glob("*.jpg"))
        if video_ids is not None:
            video_ids_set = set(video_ids)
            all_images = [
                p for p in all_images if self._parse_video_id(p.stem) in video_ids_set
            ]

        # Build paired lists — label has same stem but .png extension
        self.samples = []
        for img_path in all_images:
            label_path = labels_dir / (img_path.stem + ".png")
            assert label_path.exists(), f"Missing label for {img_path.name}"
            self.samples.append((img_path, label_path))

        assert (
            len(self.samples) > 0
        ), f"No samples found in {images_dir} for video_ids={video_ids}"

        self.text_prompt = text_prompt
        self.image_tf = _image_transform(augment, image_size)
        self.mask_tf = _mask_transform(image_size)
        self.image_size = image_size

        # Expose metadata
        self.split = split
        self.video_ids = sorted({self._parse_video_id(p.stem) for p, _ in self.samples})

    @staticmethod
    def _parse_video_id(stem: str) -> str:
        """Extract video ID from filename stem: 'S01_10020' → 'S01'."""
        m = _VIDEO_ID_RE.match(stem)
        assert m, f"Cannot parse video ID from filename: '{stem}'"
        return m.group(1)

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return (
            f"SinusSurgeryDataset(split={self.split!r}, "
            f"videos={self.video_ids}, n={len(self)})"
        )

    def __getitem__(self, idx: int) -> dict:
        img_path, label_path = self.samples[idx]

        # ── load ──────────────────────────────────────────────────────────────
        image = Image.open(img_path).convert("RGB")
        mask_raw = np.array(Image.open(label_path))  # H×W uint8 {0,1}

        # ── prompts — all derived from ground-truth mask ───────────────────────
        has_instrument = bool(mask_raw.max() == 1)
        box_prompt = generate_box_prompt(mask_raw)
        point_prompt = generate_point_prompt(mask_raw)

        # ── scale box/point to resized image space ────────────────────────────
        orig_h, orig_w = mask_raw.shape
        sx = self.image_size / orig_w
        sy = self.image_size / orig_h
        box_prompt = (box_prompt.float() * torch.tensor([sx, sy, sx, sy])).long()
        point_prompt = (point_prompt.float() * torch.tensor([sx, sy])).long()

        # ── transforms ────────────────────────────────────────────────────────
        image_tensor = self.image_tf(image)
        mask_tensor = self.mask_tf(Image.fromarray(mask_raw))

        return {
            "image": image_tensor,  # (3, H, W) float32
            "mask": mask_tensor,  # (1, H, W) float32 {0.,1.}
            "box_prompt": box_prompt,  # (4,) long [x1,y1,x2,y2]
            "point_prompt": point_prompt,  # (2,) long [cx,cy]
            "text_prompt": self.text_prompt,  # str
            "has_instrument": has_instrument,  # bool
            "video_id": self._parse_video_id(img_path.stem),  # "S01"/"L02"
            "stem": img_path.stem,  # "S01_10020"
            "image_path": str(img_path),  # full absolute path for back-tracing
        }
