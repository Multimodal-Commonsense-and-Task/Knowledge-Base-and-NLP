"""Grad-CAM attention maps.

The paper generates attention maps from the last convolutional layer of each trained
EfficientNetB0 to check that the model attends to features related to BRVO -- in their
results, the arteriovenous crossing regions of the retinal vascular arcade.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .models import MultimodalNet, UnimodalNet


def _last_conv(module: torch.nn.Module) -> torch.nn.Module:
    """Deepest Conv2d in the module -- the layer Grad-CAM hooks."""
    convs = [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)]
    if not convs:
        raise ValueError("no Conv2d layer found to attach Grad-CAM to")
    return convs[-1]


def grad_cam(model: torch.nn.Module, fundus: torch.Tensor, vessel: torch.Tensor,
             kind: str, class_idx: int = 1, device: str = "cpu") -> torch.Tensor:
    """Return an (H, W) map in [0, 1] for one hemisection.

    For the multimodal model the map is taken over the fundus branch, which is the
    branch that carries the image the map is overlaid on.
    """
    model.eval().to(device)
    if kind == "multimodal":
        target = _last_conv(model.fundus_backbone)
    elif isinstance(model, UnimodalNet):
        target = _last_conv(model.backbone)
    else:
        target = _last_conv(model)

    acts, grads = {}, {}
    h1 = target.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = target.register_full_backward_hook(
        lambda m, gi, go: grads.__setitem__("v", go[0]))

    f = fundus.unsqueeze(0).to(device).requires_grad_(True)
    v = vessel.unsqueeze(0).to(device)
    try:
        if kind == "fundus":
            logits = model(f)
        elif kind == "vessel":
            logits = model(v)
        else:
            logits = model(f, v)
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()

        a, g = acts["v"], grads["v"]                      # (1, C, h, w)
        weights = g.mean(dim=(2, 3), keepdim=True)        # global-average-pooled grads
        cam = F.relu((weights * a).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=fundus.shape[-2:], mode="bilinear",
                            align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / cam.max()).detach().cpu() if float(cam.max()) > 0 else cam.detach().cpu()
    finally:
        h1.remove()
        h2.remove()


def save_overlay(fundus: torch.Tensor, cam: torch.Tensor, out_path: Path | str,
                 alpha: float = 0.45) -> Path:
    """Write the fundus image with the attention map overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    img = fundus.permute(1, 2, 0).clamp(0, 1).numpy()
    ax[0].imshow(img); ax[0].set_title("hemisection"); ax[0].axis("off")
    ax[1].imshow(img)
    ax[1].imshow(cam.numpy(), cmap="jet", alpha=alpha)
    ax[1].set_title("Grad-CAM"); ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradcam] saved -> {out_path}")
    return out_path
