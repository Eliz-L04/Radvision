# grad_cam.py
"""
Grad-CAM / Attention Rollout Visualization for ViT Knee MRI
============================================================
Uses Attention Rollout (Abnar & Zuidema 2020) across all 12 ViT encoder layers
to produce meaningful spatial heatmaps — far more reliable than single-layer
CLS attention which tends to be near-uniform.

The overlay blends the jet heatmap ONTO the original MRI slice (semi-transparent)
so anatomical detail remains visible, matching diagnostic quality expectations.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import ViTModel, ViTImageProcessor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# =============================================================================
# Device selection
# =============================================================================

def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# Attention Rollout helper
# =============================================================================

def _attention_rollout(attentions, discard_ratio: float = 0.9) -> np.ndarray:
    """
    Compute Attention Rollout (Abnar & Zuidema 2020).

    Args:
        attentions: tuple of (1, heads, tokens, tokens) per layer — from model.attentions
        discard_ratio: fraction of low-attention weights to zero out per layer

    Returns:
        1-D float32 array of length num_patches with rolled-out CLS attention,
        values in [0, 1].
    """
    result = torch.eye(attentions[0].size(-1))  # identity

    with torch.no_grad():
        for attn in attentions:
            # attn: (1, heads, tokens, tokens)
            attn_avg = attn[0].mean(dim=0)  # (tokens, tokens) — avg over heads

            # Discard low-attention weights (noise reduction)
            flat = attn_avg.view(-1)
            thresh = flat.kthvalue(int(discard_ratio * flat.numel()))[0]
            attn_avg = torch.where(attn_avg < thresh,
                                   torch.zeros_like(attn_avg), attn_avg)

            # Add residual connection (ensures gradients flow through identity)
            attn_avg = attn_avg + torch.eye(attn_avg.size(-1))
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            result = torch.matmul(attn_avg, result)

    # CLS row (index 0) → patch tokens (index 1 …)
    cls_rollout = result[0, 1:].cpu().numpy()  # (num_patches,)

    # Normalize to [0, 1]
    vmin, vmax = cls_rollout.min(), cls_rollout.max()
    if vmax > vmin:
        cls_rollout = (cls_rollout - vmin) / (vmax - vmin)
    else:
        cls_rollout = np.zeros_like(cls_rollout)

    return cls_rollout.astype(np.float32)


# =============================================================================
# ViTGradCAM class  (public API)
# =============================================================================

class ViTGradCAM:
    """Attention-Rollout based CAM for ViT. No backward pass needed."""

    def __init__(self, model_name: str = "google/vit-base-patch16-224",
                 device=None, discard_ratio: float = 0.9):
        self.device = device or _get_device()
        self.discard_ratio = discard_ratio
        print(f"[INFO] ViTGradCAM (Attention Rollout) on {self.device}")

        import logging
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

        from transformers import ViTConfig
        config = ViTConfig.from_pretrained(model_name)
        config.output_attentions = True  # enable at model level

        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        ).to(self.device)
        self.model.eval()

        # Hook-based fallback: capture attn weights from EVERY encoder layer
        self._hooked_attns = [None] * len(self.model.encoder.layer)
        for i, layer in enumerate(self.model.encoder.layer):
            idx = i
            def make_hook(idx):
                def hook(module, input, output):
                    # ViT SelfAttention returns (context_layer,) or (context_layer, weights)
                    if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                        self._hooked_attns[idx] = output[1].detach().cpu()
                return hook
            layer.attention.attention.register_forward_hook(make_hook(idx))

    def generate_cam(self, pil_image: Image.Image) -> np.ndarray:
        """
        Returns a (14, 14) float32 attention rollout map in [0, 1].
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        self._hooked_attns = [None] * len(self.model.encoder.layer)  # reset

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # --- Prefer outputs.attentions (native); fallback to hooks
        if outputs.attentions and outputs.attentions[0] is not None:
            attentions_cpu = [a.detach().cpu() for a in outputs.attentions]
            source = "model.attentions"
        elif any(a is not None for a in self._hooked_attns):
            attentions_cpu = [a for a in self._hooked_attns if a is not None]
            source = "hooks"
        else:
            print("[WARN] No attention weights captured — returning blank CAM.")
            return np.zeros((14, 14), dtype=np.float32)

        cam_flat = _attention_rollout(attentions_cpu, discard_ratio=self.discard_ratio)
        grid = int(round(cam_flat.shape[0] ** 0.5))
        cam = cam_flat.reshape(grid, grid)
        return cam.astype(np.float32)

    # Keep old method name for backward compat
    def generate_heatmap(self, pil_image: Image.Image) -> np.ndarray:
        return self.generate_cam(pil_image)


# =============================================================================
# Overlay utilities
# =============================================================================

def overlay_heatmap(image: Image.Image,
                    cam: np.ndarray,
                    alpha: float = 0.55) -> np.ndarray:
    """
    Blend jet-colourmap heatmap ONTO the original image (alpha-composite).
    Anatomical detail remains visible beneath the colour overlay.

    Returns: uint8 RGB array (224×224×3).
    """
    img_rgb = np.array(image.convert("RGB").resize((224, 224))).astype(np.float32)

    cam_up = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
    cam_up = np.clip(cam_up, 0, 1)

    # Apply jet colourmap
    jet = plt.get_cmap("jet")
    heatmap_rgba = jet(cam_up)                          # (H, W, 4) float [0,1]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.float32)  # (H, W, 3)

    # Alpha-blend: heatmap over original
    result = alpha * heatmap_rgb + (1.0 - alpha) * img_rgb
    return np.clip(result, 0, 255).astype(np.uint8)


def overlay_red_highlight(image: Image.Image,
                           cam: np.ndarray,
                           alpha: float = 0.6,
                           threshold: float = 0.3) -> np.ndarray:
    """
    Highlight high-attention regions in RED over the grayscale MRI.
    Pixels with cam >= threshold get a red tint.

    Returns: uint8 RGB array (224×224×3).
    """
    img_gray = np.array(image.resize((224, 224)).convert("L")).astype(np.float32)
    img_gray_norm = img_gray / 255.0

    cam_up = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
    cam_up = np.clip(cam_up, 0, 1)

    mask = (cam_up >= threshold).astype(np.float32)
    intensity = cam_up * mask

    r = np.clip(img_gray_norm * (1 - alpha * mask) + alpha * intensity, 0, 1)
    g = np.clip(img_gray_norm * (1 - alpha * mask), 0, 1)
    b = np.clip(img_gray_norm * (1 - alpha * mask), 0, 1)

    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


# =============================================================================
# Multi-plane visualiser  (standalone / CLI)
# =============================================================================

def visualize_grad_cam(volume: np.ndarray,
                       slice_idx: int = None,
                       save_path: str = None,
                       threshold: float = 0.3,
                       plane_name: str = ""):
    """
    Show 3 panels: Original | Jet Overlay | Red Highlight
    """
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2

    sd = volume[slice_idx].copy().astype(np.float32)
    sd -= sd.min()
    if sd.max() > 0:
        sd /= sd.max()
    sd = (sd * 255).astype(np.uint8)
    img = Image.fromarray(sd)

    gcam = ViTGradCAM()
    cam = gcam.generate_cam(img)

    jet_overlay = overlay_heatmap(img, cam)
    red_overlay = overlay_red_highlight(img, cam, threshold=threshold)

    label = f"{plane_name.upper()} — slice {slice_idx}" if plane_name else f"slice {slice_idx}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="black")
    fig.suptitle(f"Grad-CAM | Knee MRI | {label}", color="white",
                 fontsize=13, fontweight="bold")

    panels = [
        (np.array(img.convert("RGB")), "Original"),
        (jet_overlay, "Grad-CAM Heatmap"),
        (red_overlay, "Overlay"),
    ]
    for ax, (arr, title) in zip(axes, panels):
        ax.imshow(arr)
        ax.set_title(title, color="white", fontsize=11)
        ax.axis("off")
        ax.set_facecolor("black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"[INFO] Saved → {save_path}")
    plt.show()


def visualize_three_planes(volumes: dict,
                            save_path: str = None,
                            threshold: float = 0.3):
    """
    Show all 3 planes (axial/coronal/sagittal) in a 3-row × 3-col layout —
    matching the reference image format.

    Args:
        volumes: dict of {"axial": np.ndarray, "coronal": ..., "sagittal": ...}
    """
    planes = [p for p in ["axial", "coronal", "sagittal"] if p in volumes]
    gcam = ViTGradCAM()

    fig, axes = plt.subplots(len(planes), 3,
                             figsize=(15, 5 * len(planes)),
                             facecolor="black")
    fig.suptitle("Grad-CAM  |  Knee MRI  —  3 Views", color="white",
                 fontsize=14, fontweight="bold")

    for row, plane in enumerate(planes):
        vol = volumes[plane]
        sl = vol.shape[0] // 2
        sd = vol[sl].copy().astype(np.float32)
        sd -= sd.min()
        if sd.max() > 0:
            sd /= sd.max()
        sd = (sd * 255).astype(np.uint8)
        img = Image.fromarray(sd)

        cam = gcam.generate_cam(img)
        jet_ov = overlay_heatmap(img, cam)
        red_ov = overlay_red_highlight(img, cam, threshold=threshold)

        row_axes = axes[row] if len(planes) > 1 else axes

        row_axes[0].imshow(np.array(img.convert("RGB")))
        row_axes[1].imshow(jet_ov)
        row_axes[2].imshow(red_ov)

        for ax in row_axes:
            ax.axis("off")
            ax.set_facecolor("black")

        row_axes[0].set_ylabel(plane.upper(), color="white",
                               fontsize=12, fontweight="bold", rotation=0,
                               labelpad=40, va="center")

    # Column headers (top row only)
    top = axes[0] if len(planes) > 1 else axes
    for ax, t in zip(top, ["Original", "Grad-CAM Heatmap", "Overlay"]):
        ax.set_title(t, color="white", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"[INFO] 3-plane figure saved → {save_path}")
    plt.show()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="Grad-CAM for Knee MRI volumes")
    parser.add_argument("--input",  help="Path to preprocessed .npy volume (single plane)")
    parser.add_argument("--axial",    help="Path to axial .npy volume (3-plane mode)")
    parser.add_argument("--coronal",  help="Path to coronal .npy volume")
    parser.add_argument("--sagittal", help="Path to sagittal .npy volume")
    parser.add_argument("--output",   help="Path to save figure PNG")
    parser.add_argument("--slice",    type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--plane",    default="", help="Plane label for single-plane mode")
    args = parser.parse_args()

    if args.axial or args.coronal or args.sagittal:
        vols = {}
        for key in ["axial", "coronal", "sagittal"]:
            path = getattr(args, key)
            if path and os.path.exists(path):
                vols[key] = np.load(path)
        visualize_three_planes(vols, save_path=args.output, threshold=args.threshold)

    elif args.input:
        vol = np.load(args.input)
        visualize_grad_cam(vol, slice_idx=args.slice, save_path=args.output,
                           threshold=args.threshold, plane_name=args.plane)
    else:
        print("Usage: python grad_cam.py --input volume.npy [--output out.png]")
        print("       python grad_cam.py --axial a.npy --coronal c.npy --sagittal s.npy")
