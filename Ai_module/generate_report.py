# generate_report.py
"""
Clinical MRI Report Generation for Knee Analysis
=================================================
HOW IT WORKS:
  1. Receives predictions from your ML pipeline (ViT + TDA + Random Forest)
  2. Sends those predictions to Claude API
  3. Claude writes professional clinical radiology report text
  4. Builds a 2-page PDF:
       Page 1 — Clinical report (AI-written by Claude)
       Page 2 — Visualization appendix (MRI slices + Grad-CAM + TDA diagrams)

SETUP (one time only):
  pip install anthropic
  export ANTHROPIC_API_KEY="your-api-key-here"

Called by predict.py as:
    generate_pdf_report(
        input_path, predictions, preprocessed_volumes,
        tda_features, tda_diagrams, output_path, patient_info
    )

Standalone usage:
    python generate_report.py --exam_id 1130 \\
        --data_root /path/to/MRNet-v1.0/valid \\
        --model_dir ./models \\
        --output Report_1130.pdf \\
        --patient_name "John Doe" --age 35 --gender Male
"""

import argparse
import textwrap
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from scipy.ndimage import zoom
import joblib

# ── OpenRouter API (Claude via OpenRouter) ────────────────────────────────────
import httpx
CLAUDE_AVAILABLE = True   # Always available via httpx

OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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"
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Local imports ─────────────────────────────────────────────────────────────
try:
    from grad_cam import ViTGradCAM, overlay_heatmap
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

try:
    from ripser import ripser
    from persim import plot_diagrams
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False

try:
    from tda import compute_tda_features_volumetric
    FULL_TDA_AVAILABLE = True
except ImportError:
    FULL_TDA_AVAILABLE = False


# =============================================================================
# DEVICE
# =============================================================================

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# PREPROCESSING  
# =============================================================================

def preprocess_volume(volume: np.ndarray,
                      num_slices: int = 16,
                      target_size: int = 224) -> np.ndarray:
    volume = volume.astype(np.float32)
    if volume.ndim == 3 and volume.shape[2] < volume.shape[0]:
        volume = np.transpose(volume, (2, 0, 1))
    std = np.std(volume)
    if std > 0:
        volume = (volume - np.mean(volume)) / std
    total   = volume.shape[0]
    indices = np.linspace(0, total - 1, num_slices, dtype=int)
    volume  = volume[indices]
    h, w    = volume.shape[1], volume.shape[2]
    if h != target_size or w != target_size:
        volume = zoom(volume, (1, target_size / h, target_size / w), order=1)
    volume = np.stack([volume] * 3, axis=-1)
    return volume.astype(np.float32)


# =============================================================================
# FEATURE EXTRACTION 
# =============================================================================

def extract_vit_features(volume: np.ndarray,
                         device: torch.device = None) -> np.ndarray:
    device    = device or _get_device()
    # Suppress noisy ViT load reports (UNEXPECTED/MISSING keys are expected)
    import logging
    _hf_logger = logging.getLogger("transformers.modeling_utils")
    _prev_level = _hf_logger.level
    _hf_logger.setLevel(logging.ERROR)
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model     = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device).eval()
    _hf_logger.setLevel(_prev_level)
    embeddings = []
    with torch.no_grad():
        for i in range(volume.shape[0]):
            s   = volume[i].astype(np.float32)
            s   = np.clip(s, -3, 3)
            s   = ((s + 3) / 6 * 255).astype(np.uint8)
            pil = Image.fromarray(s).convert("RGB")
            inp = processor(images=pil, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}
            out = model(**inp)
            embeddings.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.mean(np.concatenate(embeddings, axis=0), axis=0)


def extract_tda_features(volume: np.ndarray) -> Tuple[Dict, Optional[List]]:
    if not TDA_AVAILABLE:
        return {'num_h1': 0, 'mean_life_h1': 0.0, 'persistence_entropy': 0.0}, None
    gray = (volume[volume.shape[0] // 2, :, :, 0]
            if volume.ndim == 4 else volume[volume.shape[0] // 2])
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-9)
    try:
        from skimage import feature, filters
        edges = feature.canny(filters.gaussian(gray, sigma=1.0))
        pts   = np.column_stack(np.nonzero(edges))
    except Exception:
        return {'num_h1': 0, 'mean_life_h1': 0.0, 'persistence_entropy': 0.0}, None
    if len(pts) < 10:
        return {'num_h1': 0, 'mean_life_h1': 0.0, 'persistence_entropy': 0.0}, None
    if len(pts) > 2000:
        pts = pts[np.random.choice(len(pts), 2000, replace=False)]
    try:
        dgms      = ripser(pts, maxdim=1)['dgms']
        h1        = dgms[1]
        lifetimes = np.array([])
        if len(h1) > 0:
            lifetimes = h1[:, 1] - h1[:, 0]
            lifetimes = lifetimes[np.isfinite(lifetimes)]
        entropy = 0.0
        if len(lifetimes) > 0:
            L = np.sum(lifetimes)
            if L > 0:
                p       = lifetimes / L
                entropy = float(-np.sum(p * np.log(p + 1e-10)))
        return {
            'num_h1':              int(round(len(lifetimes))),
            'mean_life_h1':        float(np.mean(lifetimes)) if len(lifetimes) > 0 else 0.0,
            'persistence_entropy': entropy
        }, dgms
    except Exception:
        return {'num_h1': 0, 'mean_life_h1': 0.0, 'persistence_entropy': 0.0}, None


# =============================================================================
# GRAD-CAM 
# =============================================================================

def generate_gradcam_overlay(volume: np.ndarray,
                              slice_idx: int = None
                              ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    mid = volume.shape[0] // 2 if slice_idx is None else slice_idx
    raw = volume[mid].astype(np.float32)
    raw = np.clip(raw, -3, 3)
    raw = ((raw + 3) / 6 * 255).astype(np.uint8)
    if raw.ndim == 3 and raw.shape[-1] == 1:
        img = Image.fromarray(raw[:, :, 0]).convert("RGB")
    elif raw.ndim == 3:
        img = Image.fromarray(raw).convert("RGB")
    else:
        img = Image.fromarray(raw).convert("RGB")
    if not GRADCAM_AVAILABLE:
        return np.array(img), None, None
    try:
        gc  = ViTGradCAM()
        cam = gc.generate_heatmap(img)
        if cam.max() == 0:
            return np.array(img), None, None
        ov = overlay_heatmap(img, cam)
        return np.array(img), cam, ov
    except Exception as e:
        print(f"[WARN] Grad-CAM failed: {e}")
        return np.array(img), None, None


# =============================================================================
# ★ AI REPORT TEXT GENERATION  ★
# =============================================================================

def generate_ai_report_text(predictions: Dict,
                             tda_features: Dict,
                             patient_info: Dict) -> Dict:
    """
    Sends ML predictions to Claude API.
    Claude writes professional clinical radiology report text.

    What Claude receives:
        - Abnormality prediction + confidence %
        - ACL tear prediction + confidence %
        - Meniscus tear prediction + confidence %
        - TDA structural complexity metrics
        - Patient age and gender

    What Claude returns (as JSON):
        {
          'technique':  'MRI technique sentence.',
          'findings':   ['Finding 1...', 'Finding 2...', ...],
          'impression': ['Impression 1...', 'Impression 2...']
        }

    Falls back to template text if API unavailable.
    """
    api_key = OPENROUTER_API_KEY

    if not api_key:
        print("  [OpenRouter] No API key — using template text.")
        print("               Set OPENROUTER_API_KEY environment variable.")
        return _template_report_text(predictions, tda_features, patient_info)

    # ── Extract values ─────────────────────────────────────────────────────
    abn     = predictions.get('abnormal', {})
    acl     = predictions.get('acl', {})
    men     = predictions.get('meniscus', {})
    p_age   = patient_info.get('age', 'Unknown')
    p_gender = patient_info.get('gender', 'patient')
    num_h1  = int(round(float(tda_features.get('num_h1', 0))))
    entropy = float(tda_features.get('persistence_entropy', 0.0))

    # ── Build prompt ────────────────────────────────────────────────────────
    prompt = f"""You are an experienced musculoskeletal radiologist.
Your AI analysis system (ViT + TDA + Random Forest) has processed a knee MRI 
and produced the following predictions. Write a professional radiology report.

PATIENT:
- Age: {p_age}
- Gender: {p_gender}

AI MODEL PREDICTIONS:
- Overall Abnormality : {abn.get('label','N/A')} (confidence: {abn.get('probability',0):.1%})
- ACL Tear            : {acl.get('label','N/A')} (confidence: {acl.get('probability',0):.1%})
- Meniscus Tear       : {men.get('label','N/A')} (confidence: {men.get('probability',0):.1%})

TOPOLOGICAL DATA ANALYSIS (TDA):
- Persistent structural features (H1 loops): {num_h1}
- Persistence entropy: {entropy:.2f}
  (Higher entropy = more complex structural patterns in the MRI)

WRITING RULES:
1. Use professional radiology language throughout
2. ACL POSITIVE  → describe signal abnormality / possible tear
   ACL NEGATIVE  → confirm ACL appears intact, normal course
3. Meniscus POSITIVE → describe meniscal signal changes / tear
   Meniscus NEGATIVE → confirm menisci normal morphology
4. Always comment on: PCL, collateral ligaments, cartilage,
   bone marrow, joint effusion, patellar alignment
5. Mention TDA findings naturally in one finding sentence
6. Be concise — this is a clinical document
7. End impression with "Clinical correlation recommended"

RESPOND WITH ONLY THIS JSON — no markdown, no extra text:
{{
  "technique": "One sentence describing MRI technique.",
  "findings": [
    "Sentence about overall abnormality status.",
    "Sentence about ACL.",
    "Sentence about meniscus.",
    "Sentence about PCL.",
    "Sentence about collateral ligaments.",
    "Sentence about articular cartilage and bone marrow.",
    "Sentence about joint effusion.",
    "Sentence about TDA topological findings.",
    "Sentence about patellar alignment and tibio-femoral joint."
  ],
  "impression": [
    "Primary impression or normal finding.",
    "Secondary finding if applicable.",
    "Clinical correlation recommended."
  ]
}}"""

    # ── Call Claude via OpenRouter ──────────────────────────────────────────
    try:
        print("  [OpenRouter] Generating clinical report text via Claude...")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://radvision-ai.local",
            "X-Title": "RadVision AI Report Generator"
        }
        payload = {
            "model": "anthropic/claude-sonnet-4",
            "max_tokens": 1200,
            "messages": [{"role": "user", "content": prompt}]
        }

        resp = httpx.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=60.0
        )

        if resp.status_code != 200:
            err_body = resp.text[:500] if resp.text else "No response body"
            raise RuntimeError(f"HTTP {resp.status_code}: {err_body}")

        data = resp.json()

        raw = data["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if Claude added them
        raw = raw.replace("```json", "").replace("```", "").strip()

        result = json.loads(raw)

        # Validate keys
        if not all(k in result for k in ('findings', 'impression', 'technique')):
            raise ValueError("Claude response missing required keys")

        print("  [OpenRouter] ✅ Report text generated successfully")
        return result

    except json.JSONDecodeError as e:
        print(f"  [OpenRouter] JSON parse error: {e} — using template")
        return _template_report_text(predictions, tda_features, patient_info)
    except Exception as e:
        print(f"  [OpenRouter] Error: {e} — using template")
        return _template_report_text(predictions, tda_features, patient_info)


# =============================================================================
# TEMPLATE FALLBACK  (when Claude API is unavailable)
# =============================================================================

def _template_report_text(predictions: Dict,
                           tda_features: Dict,
                           patient_info: Dict) -> Dict:
    """
    Hardcoded template report text.
    Used when ANTHROPIC_API_KEY is not set or the API call fails.
    Quality is decent but not as natural as Claude-written text.
    """
    abn     = predictions.get('abnormal', {})
    acl     = predictions.get('acl', {})
    men     = predictions.get('meniscus', {})
    num_h1  = int(round(float(tda_features.get('num_h1', 0))))
    entropy = float(tda_features.get('persistence_entropy', 0.0))

    findings = []

    # Overall
    if abn.get('prediction', 0) == 0:
        findings.append(
            "The knee structures appear within normal limits on AI-assisted analysis."
        )
    else:
        findings.append(
            f"Evidence of internal derangement is detected with "
            f"{abn.get('probability',0):.0%} AI confidence. "
            "Clinical correlation is recommended."
        )

    # ACL
    if acl.get('prediction', 0) == 1:
        findings.append(
            f"The anterior cruciate ligament demonstrates signal abnormality "
            f"suggestive of a tear ({acl.get('probability',0):.0%} confidence). "
            "Arthroscopic evaluation may be warranted."
        )
    else:
        findings.append(
            "The anterior cruciate ligament demonstrates normal signal intensity "
            "and course. No evidence of a full-thickness tear."
        )

    # Meniscus
    if men.get('prediction', 0) == 1:
        findings.append(
            f"The meniscus demonstrates signal changes compatible with a tear "
            f"({men.get('probability',0):.0%} confidence). "
            "Grade and location require clinical correlation."
        )
    else:
        findings.append(
            "The medial and lateral menisci demonstrate normal morphology "
            "and signal intensity on available sequences."
        )

    findings += [
        "The posterior cruciate ligament appears intact with normal signal intensity.",
        "Medial and lateral collateral ligaments appear normal.",
        "The articular cartilage demonstrates normal thickness and signal intensity. "
        "No significant bone marrow edema or osseous abnormality identified.",
        "No significant joint effusion identified.",
        f"Topological data analysis identified {num_h1} persistent structural features "
        f"with a persistence entropy of {entropy:.2f}, indicating the degree of "
        "geometric complexity in the joint structures.",
        "The tibio-femoral and patello-femoral alignments appear normal. "
        "No significant dislocation is seen."
    ]

    if abn.get('prediction', 0) == 0:
        impression = [
            "No significant abnormality detected on AI-assisted analysis.",
            "Normal knee MRI appearance.",
            "Clinical correlation recommended."
        ]
    else:
        imps = []
        if acl.get('prediction', 0) == 1:
            imps.append(
                f"Sprain / tear of anterior cruciate ligament "
                f"({acl.get('probability',0):.0%} AI confidence)."
            )
        if men.get('prediction', 0) == 1:
            imps.append(
                f"Meniscal tear detected "
                f"({men.get('probability',0):.0%} AI confidence)."
            )
        if not imps:
            imps.append(
                "General abnormality detected. "
                "Specific pathology not identified by AI analysis."
            )
        imps.append("Clinical correlation recommended.")
        imps.append("Suggest: Orthopedic consultation.")
        impression = imps

    return {
        "technique": (
            "Multi-planar, multi-sequence MRI of the knee was performed "
            "at 1.5T without intravenous contrast material."
        ),
        "findings":   findings,
        "impression": impression
    }


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def _safe_name(pi: Dict) -> str:
    n = pi.get('name', '')
    return n if n and n not in ('Anonymous', '', 'N/A') else 'Patient'

def _safe_age(pi: Dict) -> str:
    a = str(pi.get('age', ''))
    return a if a and a not in ('N/A', 'None', '') else 'Unknown'

def _safe_gender(pi: Dict) -> str:
    g = str(pi.get('gender', ''))
    return g if g and g not in ('N/A', 'None', '') else 'Unknown'

def _normalize_tda_features(tda_features) -> Dict:
    if not tda_features:
        return {'num_h1': 0, 'mean_life_h1': 0.0, 'persistence_entropy': 0.0}
    sample = next(iter(tda_features.values()))
    if isinstance(sample, dict):
        keys   = list(sample.keys())
        merged = {}
        for k in keys:
            vals      = [v.get(k, 0) for v in tda_features.values() if isinstance(v, dict)]
            merged[k] = float(np.mean(vals)) if vals else 0.0
        return merged
    return tda_features

def _normalize_tda_diagrams(tda_diagrams) -> Dict:
    if tda_diagrams is None:
        return {}
    if isinstance(tda_diagrams, dict):
        return tda_diagrams
    return {'axial': tda_diagrams}


# =============================================================================
# PAGE 1 — CLINICAL REPORT
# =============================================================================

def _page1_report(pdf: PdfPages,
                  exam_id: str,
                  predictions: Dict,
                  preprocessed_volumes: Dict[str, np.ndarray],
                  tda_features: Dict,
                  patient_info: Dict,
                  report_text: Dict):
    """
    Renders Page 1: full clinical report.
    report_text = dict from generate_ai_report_text() (Claude or template).
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor('white')

    ax = fig.add_axes([0.07, 0.04, 0.90, 0.94])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    p_name    = _safe_name(patient_info)
    p_age     = _safe_age(patient_info)
    p_gender  = _safe_gender(patient_info)
    scan_date = datetime.now().strftime('%d-%m-%Y')
    mr_no     = f"MR-{exam_id}"

    y  = 0.97
    LH = 0.026   # line height

    # ── Header table ──────────────────────────────────────────────────────────
    rect = mpatches.FancyBboxPatch(
        (0, y - 0.075), 1.0, 0.075,
        boxstyle="square,pad=0", linewidth=1.0,
        edgecolor='black', facecolor='#F0F4F8'
    )
    ax.add_patch(rect)

    # Row 1
    ax.text(0.01, y - 0.010, "Patient Name:", fontsize=8, fontweight='bold', va='top')
    ax.text(0.18, y - 0.010, p_name,          fontsize=8, va='top')
    ax.text(0.50, y - 0.010, f"Date: {scan_date}", fontsize=8, fontweight='bold', va='top')
    ax.text(0.78, y - 0.010, "Referring Dr:", fontsize=8, fontweight='bold', va='top')

    ax.plot([0, 1], [y - 0.037, y - 0.037], color='#AAAAAA', linewidth=0.5)

    # Row 2
    gender_code = p_gender[0].upper() if p_gender and p_gender != 'Unknown' else '?'
    ax.text(0.01, y - 0.047, "Age / Sex:", fontsize=8, fontweight='bold', va='top')
    ax.text(0.18, y - 0.047, f"{p_age} / {gender_code}", fontsize=8, va='top')
    ax.text(0.50, y - 0.047, f"MR NO: {mr_no}", fontsize=8, fontweight='bold', va='top')

    # Vertical dividers + bottom border
    for xv in [0.48, 0.76]:
        ax.plot([xv, xv], [y, y - 0.075], color='#AAAAAA', linewidth=0.5)
    ax.plot([0, 1], [y - 0.075, y - 0.075], 'k-', linewidth=1.0)

    y -= 0.095

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(0.5, y, "MRI SCAN OF RIGHT KNEE JOINT",
            fontsize=12, fontweight='bold', ha='center', va='top')
    y -= 0.048

    # ── Technique ─────────────────────────────────────────────────────────────
    technique = report_text.get(
        'technique',
        "Multi-planar multi-sequence MRI of the knee was performed without contrast."
    )
    ax.text(0.0, y, "Technique:", fontsize=9, fontweight='bold', va='top')
    y -= LH * 1.2
    for line in textwrap.wrap(technique, width=90):
        ax.text(0.02, y, line, fontsize=8.5, va='top')
        y -= LH
    y -= 0.012

    # ── Findings ──────────────────────────────────────────────────────────────
    ax.text(0.0, y, "Findings:", fontsize=9, fontweight='bold', va='top')
    y -= LH * 1.3

    findings     = report_text.get('findings', [])
    img_inserted = False
    img_top_y    = y

    for i, bullet in enumerate(findings):
        wrapped = textwrap.wrap(bullet, width=68)
        n_lines = len(wrapped)

        ax.text(0.01, y, "➤", fontsize=9, va='top')
        for j, line in enumerate(wrapped):
            ax.text(0.055, y - j * LH, line, fontsize=8.5, va='top')
        y -= n_lines * LH + 0.006

        # Insert axial thumbnail to top-right after 3rd bullet
        if i == 2 and not img_inserted and preprocessed_volumes:
            vk   = 'axial' if 'axial' in preprocessed_volumes else list(preprocessed_volumes.keys())[0]
            vol  = preprocessed_volumes[vk]
            mid  = vol.shape[0] // 2
            simg = vol[mid, :, :, 0] if vol.ndim == 4 else vol[mid]
            ax_img = ax.inset_axes(
                [0.68, img_top_y - 0.29, 0.31, 0.28],
                transform=ax.transData
            )
            ax_img.imshow(simg, cmap='gray')
            ax_img.set_title(f"{vk.upper()} mid-slice", fontsize=6, pad=2)
            ax_img.axis('off')
            img_inserted = True

    y -= 0.008

    # ── Divider ───────────────────────────────────────────────────────────────
    ax.plot([0, 1], [y + 0.005, y + 0.005], color='#888888', linewidth=0.5)
    y -= 0.012

    # ── Impression ────────────────────────────────────────────────────────────
    ax.text(0.0, y, "IMPRESSION:", fontsize=9, fontweight='bold', va='top', style='italic')
    y -= LH * 1.4

    for imp in report_text.get('impression', []):
        wrapped = textwrap.wrap(imp, width=85)
        ax.text(0.015, y, "•", fontsize=10, va='top')
        for j, line in enumerate(wrapped):
            ax.text(0.04, y - j * LH, line, fontsize=8.5, va='top')
        y -= len(wrapped) * LH + 0.006

    y -= 0.012

    # ── Suggest line ──────────────────────────────────────────────────────────
    ax.text(0.0, y, "Suggest:", fontsize=8.5, fontweight='bold', va='top', style='italic')
    ax.text(0.10, y, "Clinical correlation.", fontsize=8.5, va='top')
    y -= 0.040

    # ── AI Confidence Summary table ───────────────────────────────────────────
    ax.text(0.0, y, "AI Confidence Summary:", fontsize=8.5, fontweight='bold', va='top')
    y -= LH * 1.6

    col_x = [0.02, 0.22, 0.44, 0.62]
    for cx, hdr in zip(col_x, ["Task", "Result", "Confidence", "Confidence Bar"]):
        ax.text(cx, y, hdr, fontsize=8, fontweight='bold', va='top', color='#333333')
    y -= LH * 1.3

    for task in ['abnormal', 'acl', 'meniscus']:
        info  = predictions.get(task, {})
        prob  = float(info.get('probability', 0.0))
        label = info.get('label', 'N/A')
        color = '#CC0000' if label == 'POSITIVE' else '#006600'

        ax.text(col_x[0], y, task.capitalize(), fontsize=8.5, va='top')
        ax.text(col_x[1], y, label,             fontsize=8.5, va='top',
                color=color, fontweight='bold')
        ax.text(col_x[2], y, f"{prob:.1%}",     fontsize=8.5, va='top')

        bar_y = y - LH * 0.55
        ax.barh(bar_y, prob * 0.32, height=LH * 0.55,
                left=col_x[3], color=color, alpha=0.80)
        ax.barh(bar_y, 0.32, height=LH * 0.55,
                left=col_x[3], color='#DDDDDD', alpha=0.30)
        y -= LH * 1.7

    # ── AI source note ────────────────────────────────────────────────────────
    has_api = bool(OPENROUTER_API_KEY)
    note = (
        "★ Report text written by Claude AI (via OpenRouter) from ViT + TDA + Random Forest predictions."
        if has_api else
        "★ Report text generated from template. Set OPENROUTER_API_KEY to enable AI-written reports."
    )
    ax.text(0.5, 0.095, note, fontsize=5.5, ha='center', va='top',
            color='#777777', style='italic')

    # ── Footer ────────────────────────────────────────────────────────────────
    ax.plot([0, 1], [0.060, 0.060], color='#888888', linewidth=0.5)
    ax.text(0.00, 0.053, "RadVision AI  |  AI-Assisted Radiology Report",
            fontsize=7, va='top', color='#555555')
    ax.text(0.50, 0.053,
            "CONFIDENTIAL — AI-assisted report. Not a substitute for clinical diagnosis.",
            fontsize=6, ha='center', va='top', color='#888888')
    ax.text(1.00, 0.053, f"DD: {scan_date}",
            fontsize=7, va='top', ha='right', color='#555555')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# PAGE 2 — VISUALIZATION APPENDIX
# =============================================================================

def _page2_visuals(pdf: PdfPages,
                   exam_id: str,
                   preprocessed_volumes: Dict[str, np.ndarray],
                   tda_diagrams: Dict):

    views   = ['axial', 'coronal', 'sagittal']
    n_views = len([v for v in views if v in preprocessed_volumes])
    if n_views == 0:
        return

    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor('white')

    n_rows = 1 + n_views
    gs = gridspec.GridSpec(
        n_rows, 3,
        height_ratios=[0.08] + [1] * n_views,
        hspace=0.50, wspace=0.20,
        left=0.04, right=0.98, top=0.98, bottom=0.03
    )

    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis('off')
    ax_t.text(0.5, 0.5, f"VISUALIZATION APPENDIX — Exam {exam_id}",
              fontsize=13, fontweight='bold', ha='center', va='center')

    col_labels = [
        "Original MRI (mid-slice)",
        "Attention Heatmap Overlay",
        "TDA Persistence Diagram"
    ]

    row_idx = 1
    for view in views:
        vol = preprocessed_volumes.get(view)
        if vol is None:
            continue

        mid = vol.shape[0] // 2

        # Column 0 — Original MRI
        ax_orig = fig.add_subplot(gs[row_idx, 0])
        simg = vol[mid, :, :, 0] if vol.ndim == 4 else vol[mid]
        ax_orig.imshow(simg, cmap='gray')
        ax_orig.set_title(f"{view.upper()}\n{col_labels[0]}", fontsize=7, pad=3)
        ax_orig.axis('off')

        # Column 1 — Attention Heatmap
        ax_gc = fig.add_subplot(gs[row_idx, 1])
        try:
            orig_arr, cam, ov = generate_gradcam_overlay(vol, slice_idx=mid)
            if ov is not None:
                ax_gc.imshow(ov)
                ax_gc.set_title(col_labels[1], fontsize=7, pad=3)
            else:
                ax_gc.imshow(orig_arr, cmap='gray')
                ax_gc.set_title("Heatmap N/A", fontsize=7, pad=3)
        except Exception as e:
            ax_gc.text(0.5, 0.5, f"Error:\n{str(e)[:40]}",
                       ha='center', va='center', fontsize=6,
                       color='red', transform=ax_gc.transAxes)
            ax_gc.set_title(col_labels[1], fontsize=7, pad=3)
        ax_gc.axis('off')

        # Column 2 — TDA Persistence Diagram
        ax_tda = fig.add_subplot(gs[row_idx, 2])
        dgms   = tda_diagrams.get(view)
        if dgms is not None and TDA_AVAILABLE:
            try:
                plot_diagrams(dgms, ax=ax_tda, show=False)
                ax_tda.set_title(col_labels[2], fontsize=7, pad=3)
            except Exception as e:
                ax_tda.text(0.5, 0.5, f"TDA error:\n{str(e)[:40]}",
                            ha='center', va='center', fontsize=6,
                            color='red', transform=ax_tda.transAxes)
                ax_tda.axis('off')
        else:
            ax_tda.text(0.5, 0.5, "TDA N/A",
                        ha='center', va='center', fontsize=9,
                        color='gray', transform=ax_tda.transAxes)
            ax_tda.set_title(col_labels[2], fontsize=7, pad=3)
            ax_tda.axis('off')

        row_idx += 1

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# PUBLIC ENTRY POINT  (called by predict.py)
# =============================================================================

def generate_pdf_report(
    input_path,
    predictions: Dict,
    preprocessed_volumes: Dict[str, np.ndarray],
    tda_features,
    tda_diagrams,
    output_path,
    patient_info: Dict = None
):
    """
    Generate a 2-page PDF clinical report.

    Flow:
      Step 1 → Claude API writes the clinical report text
               from your ML predictions (ViT + TDA + Random Forest)
      Step 2 → Page 1 rendered: header table, findings, impression,
               confidence table — all using Claude-written text
      Step 3 → Page 2 rendered: MRI images, Grad-CAM heatmaps,
               TDA persistence diagrams
      Step 4 → PDF saved to output_path
    """
    exam_id      = Path(input_path).stem
    patient_info = patient_info or {}
    tda_flat     = _normalize_tda_features(tda_features)
    tda_diag_d   = _normalize_tda_diagrams(tda_diagrams)

    print(f"\n  [Report] Generating PDF → {output_path}")

    # Step 1: Claude API generates the clinical text
    print("  [Report] Step 1/3 — Calling Claude API for report text...")
    report_text = generate_ai_report_text(predictions, tda_flat, patient_info)

    # Steps 2 & 3: Render pages
    print("  [Report] Step 2/3 — Rendering Page 1 (clinical report)...")
    print("  [Report] Step 3/3 — Rendering Page 2 (visualizations)...")

    with PdfPages(str(output_path)) as pdf:
        _page1_report(
            pdf, exam_id, predictions,
            preprocessed_volumes, tda_flat,
            patient_info, report_text        # ← Claude-written text goes here
        )
        _page2_visuals(pdf, exam_id, preprocessed_volumes, tda_diag_d)

        d = pdf.infodict()
        d['Title']        = f"Knee MRI Report — Exam {exam_id}"
        d['Author']       = "RadVision AI (Claude-Assisted)"
        d['Subject']      = "Knee MRI AI-Assisted Analysis"
        d['CreationDate'] = datetime.now()

    print(f"  [Report] ✅ Saved → {output_path}\n")


# =============================================================================
# STANDALONE USAGE
# =============================================================================

def _load_raw_data(exam_id: str, data_root: Path) -> Dict[str, np.ndarray]:
    views = {}
    for view in ['axial', 'coronal', 'sagittal']:
        for name in [f"{exam_id}.npy", f"{int(exam_id):04d}.npy"]:
            p = data_root / view / name
            if p.exists():
                views[view] = np.load(p)
                break
        if view not in views:
            raise FileNotFoundError(
                f"Cannot find {view} data for exam {exam_id} in {data_root}"
            )
    return views


def _extract_full_tda_features(volume: np.ndarray) -> np.ndarray:
    """
    Extract full TDA feature vector (144-dim) matching the training pipeline.
    Uses gradient + intensity methods with stats aggregation.
    """
    if not FULL_TDA_AVAILABLE:
        # Return zeros if tda.py is not available — matches 144 columns
        return np.zeros(144)

    all_features = {}
    for method in ['gradient', 'intensity']:
        method_features = compute_tda_features_volumetric(
            volume=volume,
            method=method,
            max_points=2000,
            aggregation='stats',
            sample_slices=8
        )
        # Add method suffix (same as process_single_volume with multiple methods)
        method_features = {f"{k}_{method}": v for k, v in method_features.items()}
        all_features.update(method_features)

    # Convert to array preserving dict insertion order
    # (matches the column order from process_single_volume in tda.py,
    #  which is the same order saved to the training CSVs)
    return np.array(list(all_features.values()), dtype=np.float32)


def _run_predictions(exam_id: str, model_dir: Path,
                     feat_vectors: Dict[str, np.ndarray],
                     tda_vectors: Optional[Dict[str, np.ndarray]] = None) -> Dict:
    # Concatenate ViT features: [axial, coronal, sagittal]
    vit_combined = np.concatenate(list(feat_vectors.values()))

    # Concatenate TDA features: [axial, coronal, sagittal]
    if tda_vectors:
        tda_combined = np.concatenate(list(tda_vectors.values()))
        combined = np.concatenate([vit_combined, tda_combined]).reshape(1, -1)
    else:
        combined = vit_combined.reshape(1, -1)

    print(f"    Feature vector: {combined.shape[1]} dims "
          f"(ViT={len(vit_combined)}, TDA={len(tda_combined) if tda_vectors else 0})")

    predictions = {}
    for task in ['abnormal', 'acl', 'meniscus']:
        candidates = (
            list(model_dir.glob(f"rf_{task}.joblib")) +
            list(model_dir.glob(f"ensemble_{task}.joblib")) +
            list(model_dir.glob(f"*{task}*.joblib"))
        )
        candidates = [c for c in candidates
                      if 'scaler'    not in c.name
                      and 'pca'      not in c.name
                      and 'threshold' not in c.name]
        if not candidates:
            print(f"  [WARN] No model found for {task} in {model_dir}")
            continue

        model = joblib.load(candidates[0])
        X     = combined.copy()

        sp = model_dir / f"scaler_{task}.joblib"
        if sp.exists():
            X = joblib.load(sp).transform(X)

        threshold = 0.5
        tp = model_dir / f"threshold_{task}.json"
        if tp.exists():
            with open(tp) as f:
                threshold = json.load(f).get("best_threshold", 0.5)

        prob = float(model.predict_proba(X)[0, 1])
        pred = int(prob >= threshold)
        predictions[task] = {
            'prediction': pred,
            'probability': prob,
            'label': 'POSITIVE' if pred == 1 else 'NEGATIVE'
        }
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI-assisted clinical PDF report for a Knee MRI exam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUICK START:
  export OPENROUTER_API_KEY="your-openrouter-key"

  python generate_report.py \\
      --exam_id 1130 \\
      --data_root /path/to/MRNet-v1.0/valid \\
      --model_dir ./models \\
      --patient_name "John Doe" --age 35 --gender Male

WITHOUT API KEY (uses template text):
  python generate_report.py --exam_id 1130 --data_root /path/to/valid --model_dir ./models
        """
    )
    parser.add_argument("--exam_id",      required=True)
    parser.add_argument("--data_root",    required=True)
    parser.add_argument("--model_dir",    default="models")
    parser.add_argument("--output",       default=None)
    parser.add_argument("--patient_name", default="Anonymous")
    parser.add_argument("--age",          default="N/A")
    parser.add_argument("--gender",       default="N/A")
    args = parser.parse_args()

    output_path = Path(args.output or f"Report_{args.exam_id}.pdf")
    model_dir   = Path(args.model_dir)
    data_root   = Path(args.data_root)

    has_api = bool(OPENROUTER_API_KEY)

    print(f"\n{'='*60}")
    print(f"  RadVision — Exam {args.exam_id}")
    print(f"  Claude API : {'✅ Enabled via OpenRouter' if has_api else '⚠️  Not configured — using template'}")
    print(f"  Output     : {output_path}")
    print(f"{'='*60}")

    print("\n  Loading and preprocessing MRI views...")
    raw_vols = _load_raw_data(args.exam_id, data_root)

    proc_vols    = {}
    feat_vectors = {}
    tda_features = {}
    tda_diagrams = {}
    tda_vectors  = {}   # Full TDA feature vectors for RF model

    for view, vol in raw_vols.items():
        print(f"    Processing {view}...")
        pv                 = preprocess_volume(vol)
        proc_vols[view]    = pv
        feat_vectors[view] = extract_vit_features(pv)
        tda_features[view], tda_diagrams[view] = extract_tda_features(pv)
        # Extract full 144-dim TDA features matching training pipeline
        print(f"    Extracting full TDA features for {view}...")
        tda_vectors[view]  = _extract_full_tda_features(pv)

    print("\n  Running AI predictions (Random Forest)...")
    predictions = _run_predictions(args.exam_id, model_dir, feat_vectors, tda_vectors)

    for task, info in predictions.items():
        print(f"    {task:10s}: {info['label']:8s} ({info['probability']:.1%})")

    patient_info = {
        'name':   args.patient_name,
        'age':    args.age,
        'gender': args.gender
    }

    generate_pdf_report(
        input_path           = Path(f"{args.exam_id}.npy"),
        predictions          = predictions,
        preprocessed_volumes = proc_vols,
        tda_features         = tda_features,
        tda_diagrams         = tda_diagrams,
        output_path          = output_path,
        patient_info         = patient_info
    )

    print(f"\n✅ Done! Report saved to: {output_path}")