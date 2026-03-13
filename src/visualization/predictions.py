"""
visualization/predictions.py — Save prediction vs ground-truth comparison grids.

Produces a 3-column image grid:
    [Input image | Ground-truth mask | Predicted mask]

Saved as PNG to the experiment output directory. Designed to be called from
LightningModule.on_test_epoch_end so each experiment leaves visual evidence.

Usage
-----
    save_prediction_grid(
        images=batch["image"],      # (B, 3, H, W) float, SAM-normalized
        masks=batch["mask"],        # (B, 1, H, W) float {0, 1}
        logits=pred,                # (B, 1, H, W) float logits
        save_path=Path("outputs/C_C_fold1/version_0/predictions_epoch05.png"),
        max_samples=8,
    )
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils


# SAM normalization constants — used to invert the image for display
_SAM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_SAM_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denormalize(images: torch.Tensor) -> torch.Tensor:
    """
    Invert SAM image normalization so pixels are in [0, 1] for display.

    Args:
        images: (B, 3, H, W) float tensor, SAM-normalized.

    Returns:
        (B, 3, H, W) float tensor in [0, 1].
    """
    mean = _SAM_MEAN.to(images.device)
    std = _SAM_STD.to(images.device)
    return (images * std + mean).clamp(0.0, 1.0)


def save_prediction_grid(
    images: torch.Tensor,
    masks: torch.Tensor,
    logits: torch.Tensor,
    save_path: Path,
    max_samples: int = 8,
    threshold: float = 0.5,
) -> None:
    """
    Save a 3-column comparison grid: image | GT mask | predicted mask.

    Args:
        images:      (B, 3, H, W) SAM-normalized input images.
        masks:       (B, 1, H, W) binary ground-truth masks {0.0, 1.0}.
        logits:      (B, 1, H, W) raw model logits (pre-sigmoid).
        save_path:   File path to save the PNG grid.
        max_samples: Maximum number of samples to include in the grid.
        threshold:   Sigmoid threshold for binarising predictions.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(images.size(0), max_samples)
    imgs = _denormalize(images[:n].cpu())           # (n, 3, H, W)
    gts = masks[:n].cpu().expand(-1, 3, -1, -1)    # (n, 3, H, W) grayscale→RGB
    preds = (logits[:n].sigmoid() > threshold).float().cpu().expand(-1, 3, -1, -1)

    # Interleave: [img_0, gt_0, pred_0, img_1, gt_1, pred_1, ...]
    rows: list[torch.Tensor] = []
    for i in range(n):
        rows.extend([imgs[i], gts[i], preds[i]])
    grid = vutils.make_grid(torch.stack(rows), nrow=3, padding=2, normalize=False)

    # Save as EPS via matplotlib (torchvision save_image only supports PNG)
    arr = grid.permute(1, 2, 0).numpy()  # (H, W, 3)
    fig, ax = plt.subplots(1, 1, figsize=(arr.shape[1] / 100, arr.shape[0] / 100), dpi=100)
    ax.imshow(arr)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(save_path, format="eps", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
