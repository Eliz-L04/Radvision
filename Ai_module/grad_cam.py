# grad_cam.py
"""
XAI Heatmap Visualization for ViT Knee MRI
===========================================
Uses **Patch-Token Norm CAM** (EigenCAM-style):

  - The ViT last hidden state yields one 768-d embedding per 16×16 patch.
  - The L2 norm of each patch token reflects how much "information" the
    model encoded from that spatial region.
  - Patches covering real tissue (ligaments, cartilage, bone) have much
    higher norm than patches over featureless black background.
  - This produces anatomically-focused heatmaps without any classifier head.

A **foreground mask** (Otsu + morphological closing) clips activations to
the actual MRI scan region so hot colours never bleed into the border.
Gaussian smoothing + histogram equalisation give a clean, vivid result.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from transformers import ViTModel, ViTImageProcessor
import matplotlib.pyplot as plt


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
# Foreground (scan region) mask
# =============================================================================

def _make_scan_mask(img_gray: np.ndarray,
                    morph_iters: int = 4) -> np.ndarray:
    """
    Returns a binary float32 mask (H, W) where 1 = inside MRI scan,
    0 = black background outside the scan.

    Uses Otsu thresholding + morphological closing + largest connected
    component to isolate the scan body.
    """
    gray_u8 = np.clip(img_gray, 0, 255).astype(np.uint8)

    # Otsu global threshold
    _, binary = cv2.threshold(gray_u8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing: fill internal holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                              iterations=morph_iters)

    # Keep only the largest foreground blob
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        closed = (labels == largest).astype(np.uint8) * 255

    return (closed > 0).astype(np.float32)


# =============================================================================
# Patch-Token Norm CAM helper
# =============================================================================

def _patch_norm_cam(last_hidden_state: torch.Tensor,
                    grid_h: int = 14, grid_w: int = 14) -> np.ndarray:
    """
    Compute a spatial heatmap from the L2 norm of each patch token.

    Args:
        last_hidden_state: (1, N_tokens, hidden_dim) from ViT forward pass.
            Token 0 = CLS, tokens 1 … grid_h*grid_w = patch tokens.
        grid_h, grid_w: spatial grid size (14×14 for vit-base-patch16-224).

    Returns:
        float32 array (grid_h, grid_w) normalised to [0, 1].
    """
    # Patch tokens only: (num_patches, hidden_dim)
    patch_tokens = last_hidden_state[0, 1:, :]          # (196, 768)
    norms = torch.norm(patch_tokens, dim=-1)             # (196,)

    cam_flat = norms.cpu().float().numpy()               # (196,)

    # Normalize to [0, 1]
    lo, hi = cam_flat.min(), cam_flat.max()
    if hi > lo:
        cam_flat = (cam_flat - lo) / (hi - lo)
    else:
        cam_flat = np.zeros_like(cam_flat)

    return cam_flat.reshape(grid_h, grid_w).astype(np.float32)


def _eigen_cam(last_hidden_state: torch.Tensor,
               n_layers_hidden_states=None,
               grid_h: int = 14, grid_w: int = 14) -> np.ndarray:
    """
    EigenCAM: project patch tokens onto the first principal component.
    More discriminative than plain norm for complex textures.

    Returns float32 (grid_h, grid_w) in [0, 1].
    """
    patch_tokens = last_hidden_state[0, 1:, :].cpu().float().numpy()  # (196, 768)

    # SVD: first right-singular vector captures the dominant activation direction
    try:
        U, S, Vt = np.linalg.svd(patch_tokens, full_matrices=False)
        cam_flat = U[:, 0]                  # (196,) — first principal component
        # Flip so positive = active
        if cam_flat.mean() < 0:
            cam_flat = -cam_flat
    except np.linalg.LinAlgError:
        # Fallback to norm-based
        cam_flat = np.linalg.norm(patch_tokens, axis=-1)

    lo, hi = cam_flat.min(), cam_flat.max()
    if hi > lo:
        cam_flat = (cam_flat - lo) / (hi - lo)
    else:
        cam_flat = np.zeros_like(cam_flat)

    return cam_flat.reshape(grid_h, grid_w).astype(np.float32)


# =============================================================================
# ViTGradCAM class  (public API)
# =============================================================================

class ViTGradCAM:
    """
    Patch-Token Norm + EigenCAM for ViT.

    Combines norm-based and PCA-based maps with a geometric mean to
    produce stable, anatomy-focused heatmaps even on a ViT that was
    pre-trained on natural images (not fine-tuned on knee MRI).
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224",
                 device=None,
                 sigma: float = 2.0):
        """
        Args:
            sigma: Gaussian smoothing sigma applied to the raw CAM grid.
                   Higher = smoother blob, lower = finer detail.
        """
        self.device = device or _get_device()
        self.sigma = sigma
        print(f"[INFO] ViTGradCAM (Patch-Norm + EigenCAM) on {self.device}")

        import logging
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(
            model_name, ignore_mismatched_sizes=True
        ).to(self.device)
        self.model.eval()

    def generate_cam(self, pil_image: Image.Image) -> np.ndarray:
        """
        Returns a (14, 14) float32 CAM in [0, 1].

        Strategy:
          1. Run ViT forward pass with output_attentions=True
          2. Extract CLS→patch attention from the last attention layer,
             averaged over all heads  (196,)
          3. Compute patch-token L2 norm map  (196,)
          4. Geometric mean of attention × norm — blends semantic focus
             (what the model looks at) with information density (how much
             is encoded per patch)
          5. Smooth with Gaussian (reduces patch-grid artefacts)
          6. Renormalize to [0, 1]
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # ── CLS attention map ────────────────────────────────────────────
        # Last attention layer: (1, num_heads, seq_len, seq_len)
        last_attn = outputs.attentions[-1]        # (1, 12, 197, 197)
        # CLS token (idx 0) → patch tokens (idx 1..196), avg over heads
        cls_attn  = last_attn[0, :, 0, 1:]        # (12, 196)
        attn_flat = cls_attn.mean(dim=0).cpu().float().numpy()  # (196,)

        # ── Patch token norm map ─────────────────────────────────────────
        lhs = outputs.last_hidden_state           # (1, 197, 768)
        patch_tokens = lhs[0, 1:, :]             # (196, 768)
        norm_flat = torch.norm(patch_tokens, dim=-1).cpu().float().numpy()  # (196,)

        # ── Normalize each signal to [0, 1] ─────────────────────────────
        def _norm01(x: np.ndarray) -> np.ndarray:
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

        attn_map = _norm01(attn_flat)
        norm_map = _norm01(norm_flat)

        # ── Geometric mean: focus where BOTH signals are strong ──────────
        cam = np.sqrt(attn_map * norm_map + 1e-8)

        # ── Smooth to remove coarse patch-grid artefacts ─────────────────
        cam = gaussian_filter(cam.reshape(14, 14), sigma=self.sigma)

        # ── Renormalize ──────────────────────────────────────────────────
        lo, hi = cam.min(), cam.max()
        if hi > lo:
            cam = (cam - lo) / (hi - lo)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    # Keep old method name for backward compat
    def generate_heatmap(self, pil_image: Image.Image) -> np.ndarray:
        return self.generate_cam(pil_image)


# =============================================================================
# Overlay utilities
# =============================================================================

def _equalize_cam(cam: np.ndarray) -> np.ndarray:
    """
    Histogram-stretch the CAM so the top 5 % of values fill the red end of
    jet and the bottom 20 % are pushed to blue, giving a vivid, spread map.
    """
    lo = np.percentile(cam, 20)
    hi = np.percentile(cam, 95)
    if hi > lo:
        cam = np.clip((cam - lo) / (hi - lo), 0, 1)
    return cam


def overlay_heatmap(image: Image.Image,
                    cam: np.ndarray,
                    alpha: float = 0.55) -> np.ndarray:
    """
    Blend jet-colormap heatmap ONTO the original image.
    The heatmap is:
      - upsampled to 224×224 with bicubic interpolation
      - masked to the MRI scan foreground (no heatmap on black background)
      - histogram-equalised for vivid colour spread

    Returns: uint8 RGB array (224×224×3).
    """
    img_rgb  = np.array(image.convert("RGB").resize((224, 224))).astype(np.float32)
    img_gray = np.array(image.convert("L").resize((224, 224))).astype(np.float32)

    cam_up = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
    cam_up = np.clip(cam_up, 0, 1)

    # Foreground mask → suppress activations outside scan
    scan_mask = _make_scan_mask(img_gray)
    cam_up = cam_up * scan_mask

    # Histogram equalisation inside the mask
    cam_up = _equalize_cam(cam_up)

    # Jet colormap
    jet = plt.get_cmap("jet")
    heatmap_rgb = (jet(cam_up)[:, :, :3] * 255).astype(np.float32)

    # Alpha-blend only inside scan; outside stays original
    mask_3d = scan_mask[:, :, np.newaxis]
    result = (alpha * heatmap_rgb + (1.0 - alpha) * img_rgb) * mask_3d + \
             img_rgb * (1.0 - mask_3d)

    return np.clip(result, 0, 255).astype(np.uint8)


def overlay_red_highlight(image: Image.Image,
                           cam: np.ndarray,
                           alpha: float = 0.6,
                           threshold: float = 0.3) -> np.ndarray:
    """
    Highlight high-attention regions in RED over the grayscale MRI.
    Activated pixels (cam >= threshold AND inside the scan) get a red tint.

    Returns: uint8 RGB array (224×224×3).
    """
    img_gray = np.array(image.resize((224, 224)).convert("L")).astype(np.float32)
    img_gray_norm = img_gray / 255.0

    cam_up = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
    cam_up = np.clip(cam_up, 0, 1)

    # Foreground mask
    scan_mask = _make_scan_mask(img_gray)
    cam_up = cam_up * scan_mask
    cam_up = _equalize_cam(cam_up)

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
    Show 2×2 grid: 2 representative slices × [Original Image Slice | Heatmap Overlay Slice].
    If slice_idx is given, only that slice is shown (1×2 grid).
    """
    n_vol = volume.shape[0]
    if slice_idx is not None:
        slice_indices = [slice_idx]
    else:
        slice_indices = [max(0, n_vol // 3), min(n_vol - 1, 2 * n_vol // 3)]

    gcam   = ViTGradCAM()
    n_rows = len(slice_indices)

    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(10, 5 * n_rows),
                             facecolor="black")
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # ensure 2-D indexing

    label = plane_name.upper() if plane_name else "Knee MRI"
    fig.suptitle(f"Grad-CAM  |  {label}", color="white",
                 fontsize=13, fontweight="bold")

    axes[0, 0].set_title("Original Image Slice",  color="white", fontsize=11)
    axes[0, 1].set_title("Heatmap Overlay Slice", color="white", fontsize=11)

    for row, sl in enumerate(slice_indices):
        sd = volume[sl].copy().astype(np.float32)
        sd -= sd.min()
        if sd.max() > 0:
            sd /= sd.max()
        sd = (sd * 255).astype(np.uint8)
        img = Image.fromarray(sd)

        cam     = gcam.generate_cam(img)
        overlay = overlay_heatmap(img, cam, alpha=0.5)

        img_gray = np.array(img.convert("L").resize((224, 224)))

        axes[row, 0].imshow(img_gray, cmap="gray")
        axes[row, 0].axis("off")
        axes[row, 0].set_facecolor("black")

        axes[row, 1].imshow(overlay)
        axes[row, 1].axis("off")
        axes[row, 1].set_facecolor("black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"[INFO] Saved → {save_path}")
    plt.show()


def visualize_three_planes(volumes: dict,
                            save_path: str = None,
                            threshold: float = 0.3):
    """
    Show all 3 planes (axial/coronal/sagittal) in a 3-row × 3-col layout.
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

    # Column headers
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
    parser.add_argument("--input",    help="Path to preprocessed .npy volume (single plane)")
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
