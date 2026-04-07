from flask import Flask, request, jsonify, send_file
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from dotenv import load_dotenv
import numpy as np
import gridfs
import io
import sys
import json
import traceback
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os
from bson import ObjectId
from datetime import datetime

# ── Ai_module path for lazy report generation imports ──
AI_MODULE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Ai_module")

# AI pipeline is imported lazily inside the /generate-report endpoint
# to avoid crashing the server if torch/transformers aren't in the backend venv
AI_PIPELINE_AVAILABLE = None  # None = not yet checked

# TDA imports
try:
    from ripser import ripser
    from persim import plot_diagrams
    from skimage import feature, filters
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False
    print("[WARN] TDA dependencies not installed. Install with:")
    print("       pip install ripser persim scikit-image")


# APP SETUP
load_dotenv()

app = Flask(__name__)
CORS(app)

app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)

users_collection = mongo.db.users
patient_collection = mongo.db.patient_scans
reports_collection = mongo.db.reports

# GridFS
fs = gridfs.GridFS(mongo.db)


# ROOT
@app.route("/")
def home():
    return jsonify({"message": "✅ RadVision Backend Running"})


# AUTH ROUTES (LOGIN / REGISTER) ✅ FIXED
@app.route("/register", methods=["POST"])
def register():
    data = request.json

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    dob = data.get("dob")

    if not all([username, email, password, dob]):
        return jsonify({"message": "All fields are required"}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"message": "User already exists"}), 400

    hashed_password = generate_password_hash(password)

    users_collection.insert_one({
        "username": username,
        "email": email,
        "password": hashed_password,
        "dob": dob
    })

    return jsonify({"message": "Registration successful"}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.json

    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"message": "Email and password required"}), 400

    user = users_collection.find_one({"email": email})

    if not user:
        return jsonify({"message": "User not found"}), 404

    if not check_password_hash(user["password"], password):
        return jsonify({"message": "Incorrect password"}), 401

    return jsonify({
        "message": "Login successful",
        "email": user["email"],
        "username": user["username"]
    }), 200


# UPLOAD MRI (.npy) + PATIENT DETAILS
@app.route("/upload-scan", methods=["POST"])
def upload_scan():
    patient_id = request.form.get("patientId")
    name = request.form.get("name")
    age = request.form.get("age")
    gender = request.form.get("gender")

    axial = request.files.get("axial")
    sagittal = request.files.get("sagittal")
    coronal = request.files.get("coronal")

    if not patient_id or not name or not age or not gender:
        return jsonify({"error": "Patient details missing"}), 400

    if not axial and not sagittal and not coronal:
        return jsonify({"error": "At least one MRI scan required"}), 400

    mri_files = {}
    mri_slices = {}   # cache total_slices per plane

    for plane_name, file_obj in [("axial", axial), ("sagittal", sagittal), ("coronal", coronal)]:
        if file_obj:
            raw = file_obj.read()
            mri_files[plane_name] = fs.put(raw, filename=f"{patient_id}_{plane_name}.npy")
            # Pre-compute total_slices so /mri-info never needs to reload the whole volume
            try:
                vol = np.load(io.BytesIO(raw), allow_pickle=True)
                mri_slices[plane_name] = int(vol.shape[0])
            except Exception:
                mri_slices[plane_name] = 0

    patient_collection.insert_one({
        "patient_id": patient_id,
        "name": name,
        "age": age,
        "gender": gender,
        "created_at": datetime.utcnow(),
        "mri": mri_files,
        "mri_slices": mri_slices,
    })

    return jsonify({
        "message": "MRI uploaded successfully",
        "patientId": patient_id,
        "files": list(mri_files.keys())
    }), 201

# MRI SLIDER INFO
@app.route("/mri-info/<patient_id>/<plane>")
def mri_info(patient_id, plane):
    record = patient_collection.find_one({"patient_id": patient_id})

    if not record:
        return jsonify({"error": "Patient not found"}), 404

    if plane not in record.get("mri", {}):
        return jsonify({"error": f"{plane} MRI not found"}), 404

    # Use cached slice count if available (fast path)
    cached = record.get("mri_slices", {})
    if plane in cached and cached[plane] > 0:
        return jsonify({"total_slices": cached[plane]})

    # Fallback: load volume from GridFS (slow path, for old records)
    file_data = fs.get(record["mri"][plane]).read()
    volume = np.load(io.BytesIO(file_data), allow_pickle=True)

    # Cache for next time
    patient_collection.update_one(
        {"patient_id": patient_id},
        {"$set": {f"mri_slices.{plane}": int(volume.shape[0])}}
    )

    return jsonify({"total_slices": int(volume.shape[0])})


# MRI SLICE IMAGE
@app.route("/mri-slice/<patient_id>/<plane>/<int:slice_idx>")
def mri_slice(patient_id, plane, slice_idx):
    record = patient_collection.find_one({"patient_id": patient_id})

    if not record or plane not in record["mri"]:
        return jsonify({"error": "MRI not found"}), 404

    file_data = fs.get(record["mri"][plane]).read()
    volume = np.load(io.BytesIO(file_data), allow_pickle=True)

    if slice_idx < 0 or slice_idx >= volume.shape[0]:
        return jsonify({"error": "Slice out of range"}), 400

    slice_img = volume[slice_idx].astype(float)

    # Normalize for browser display
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
    slice_img = (slice_img * 255).astype("uint8")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(slice_img, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


# TDA HELPER
def _run_tda_on_patient(patient_id, requested_plane=None):
    """Load MRI from GridFS, run TDA on mid-slice, return (features_dict, diagrams)."""
    if not TDA_AVAILABLE:
        return None, None, "TDA dependencies not installed (ripser, persim, scikit-image)"

    record = patient_collection.find_one({"patient_id": patient_id})
    if not record:
        return None, None, "Patient not found"

    mri = record.get("mri", {})
    if not mri:
        return None, None, "No MRI data found for this patient"

    # Use requested plane, or fall back to first available
    if requested_plane and requested_plane in mri:
        plane = requested_plane
    else:
        plane = None
        for p in ["axial", "coronal", "sagittal"]:
            if p in mri:
                plane = p
                break

    if plane is None:
        return None, None, "No MRI planes available"

    # Load volume from GridFS
    file_data = fs.get(mri[plane]).read()
    volume = np.load(io.BytesIO(file_data), allow_pickle=True)

    # Get middle slice as grayscale
    mid_idx = volume.shape[0] // 2
    slice_2d = volume[mid_idx].astype(np.float32)

    # Handle multi-channel (H, W, C) -> take first channel
    if slice_2d.ndim == 3:
        slice_2d = slice_2d[:, :, 0]

    # Normalize to [0, 1]
    vmin, vmax = slice_2d.min(), slice_2d.max()
    if vmax - vmin > 1e-6:
        slice_2d = (slice_2d - vmin) / (vmax - vmin)
    else:
        return None, None, "Slice has no contrast"

    # Edge detection -> point cloud
    try:
        edges = feature.canny(filters.gaussian(slice_2d, sigma=1.5))
        pts = np.column_stack(np.nonzero(edges))
    except Exception as e:
        return None, None, f"Edge detection failed: {str(e)}"

    if len(pts) < 10:
        return None, None, "Too few edge points for TDA"

    # Subsample if too many points (keep it fast)
    max_points = 2000
    if len(pts) > max_points:
        pts = pts[np.random.choice(len(pts), max_points, replace=False)]

    # Run ripser
    try:
        result = ripser(pts, maxdim=1)
        dgms = result["dgms"]
    except Exception as e:
        return None, None, f"Ripser failed: {str(e)}"

    # Compute features
    h0 = dgms[0] if len(dgms) > 0 else np.array([])
    h1 = dgms[1] if len(dgms) > 1 else np.array([])

    h0_lifetimes = np.array([])
    if len(h0) > 0:
        h0_lifetimes = h0[:, 1] - h0[:, 0]
        h0_lifetimes = h0_lifetimes[np.isfinite(h0_lifetimes)]

    h1_lifetimes = np.array([])
    if len(h1) > 0:
        h1_lifetimes = h1[:, 1] - h1[:, 0]
        h1_lifetimes = h1_lifetimes[np.isfinite(h1_lifetimes)]

    # Persistence entropy
    entropy = 0.0
    if len(h1_lifetimes) > 0:
        L = np.sum(h1_lifetimes)
        if L > 0:
            p = h1_lifetimes / L
            entropy = float(-np.sum(p * np.log(p + 1e-10)))

    # Which planes are available (for frontend dropdown)
    available_planes = [p for p in ["axial", "coronal", "sagittal"] if p in mri]

    features = {
        "plane": plane,
        "available_planes": available_planes,
        "num_points": int(len(pts)),
        "h0_count": int(len(h0_lifetimes)),
        "h1_count": int(len(h1_lifetimes)),
        "mean_life_h0": float(np.mean(h0_lifetimes)) if len(h0_lifetimes) > 0 else 0.0,
        "mean_life_h1": float(np.mean(h1_lifetimes)) if len(h1_lifetimes) > 0 else 0.0,
        "max_life_h1": float(np.max(h1_lifetimes)) if len(h1_lifetimes) > 0 else 0.0,
        "persistence_entropy": float(f"{entropy:.4f}"),
    }

    return features, dgms, None


# GRAD-CAM VISUALIZATION

# Lazy-loaded Grad-CAM module (shares the same AI_MODULE_DIR)
_GRADCAM_LOADED = None
_GRADCAM_INSTANCE = None   # singleton ViTGradCAM — avoid reloading ViT model per request

def _load_gradcam_module():
    global _GRADCAM_LOADED
    if _GRADCAM_LOADED is not None:   # only skip if previously SUCCEEDED
        return _GRADCAM_LOADED
    try:
        if AI_MODULE_DIR not in sys.path:
            sys.path.insert(0, AI_MODULE_DIR)
        from grad_cam import ViTGradCAM, overlay_heatmap, overlay_red_highlight
        import cv2
        _GRADCAM_LOADED = (ViTGradCAM, overlay_heatmap, overlay_red_highlight, cv2)
        print("[INFO] Grad-CAM module loaded successfully")
        return _GRADCAM_LOADED
    except ImportError as e:
        print(f"[WARN] grad_cam not available: {e}")
        return None   # NOT cached — will retry next request


def _get_gradcam_instance():
    """Returns a cached ViTGradCAM singleton so the ViT model is only loaded once."""
    global _GRADCAM_INSTANCE
    if _GRADCAM_INSTANCE is not None:
        return _GRADCAM_INSTANCE
    gcam_module = _load_gradcam_module()
    if gcam_module is None:
        return None
    ViTGradCAM = gcam_module[0]
    _GRADCAM_INSTANCE = ViTGradCAM()
    print("[INFO] ViTGradCAM singleton created (model loaded once)")
    return _GRADCAM_INSTANCE


@app.route("/grad-cam/<patient_id>")
def grad_cam_image(patient_id):
    """Returns a 3-panel PNG: Original | Jet Heatmap | Red Overlay for one MRI plane."""
    plane = request.args.get("plane", "axial")
    overlay_type = request.args.get("overlay", "both")   # "jet" | "red" | "both"
    threshold = float(request.args.get("threshold", "0.3"))

    # Check patient exists
    record = patient_collection.find_one({"patient_id": patient_id})
    if not record:
        return jsonify({"error": "Patient not found"}), 404

    mri = record.get("mri", {})
    available = [p for p in ["axial", "coronal", "sagittal"] if p in mri]
    if not available:
        return jsonify({"error": "No MRI data for this patient"}), 404

    # Use requested plane, or fallback to first available
    use_plane = plane if plane in mri else available[0]

    # Load AI module lazily
    gcam_module = _load_gradcam_module()
    if gcam_module is None:
        return jsonify({"error": "Grad-CAM dependencies not installed (torch, transformers, cv2)"}), 500

    ViTGradCAM, overlay_heatmap, overlay_red_highlight, cv2 = gcam_module

    try:
        # Load MRI volume from GridFS
        file_data = fs.get(mri[use_plane]).read()
        volume = np.load(io.BytesIO(file_data), allow_pickle=True)

        from PIL import Image as PILImage

        # Select 2 representative slices (33 % and 66 % through volume)
        n_vol = volume.shape[0]
        slice_indices = [max(0, n_vol // 3), min(n_vol - 1, 2 * n_vol // 3)]

        # Get Grad-CAM singleton (ViT loaded once)
        gc = _get_gradcam_instance()
        if gc is None:
            return jsonify({"error": "Grad-CAM model failed to load"}), 500

        # Build 2×2 figure: rows = slices, cols = [Original Image Slice | Heatmap Overlay Slice]
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor="black")
        fig.suptitle(
            f"Grad-CAM  |  {use_plane.capitalize()} Plane",
            color="white", fontsize=13, fontweight="bold"
        )

        axes[0, 0].set_title("Original Image Slice",  color="white", fontsize=11)
        axes[0, 1].set_title("Heatmap Overlay Slice", color="white", fontsize=11)

        for row_i, sl_idx in enumerate(slice_indices):
            sd = volume[sl_idx].copy().astype(np.float32)
            sd -= sd.min()
            if sd.max() > 0:
                sd /= sd.max()
            sd = (sd * 255).astype(np.uint8)

            img_slice = PILImage.fromarray(sd)
            cam_slice = gc.generate_cam(img_slice)
            jet_ov    = overlay_heatmap(img_slice, cam_slice, alpha=0.5)

            img_gray = np.array(img_slice.convert("L").resize((224, 224)))

            axes[row_i, 0].imshow(img_gray, cmap="gray")
            axes[row_i, 0].axis("off")
            axes[row_i, 0].set_facecolor("black")

            axes[row_i, 1].imshow(jet_ov)
            axes[row_i, 1].axis("off")
            axes[row_i, 1].set_facecolor("black")

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150,
                    facecolor="black", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Grad-CAM failed: {str(e)}"}), 500


@app.route("/grad-cam-info/<patient_id>")
def grad_cam_info(patient_id):
    """Returns JSON with available planes and CAM attention stats for the patient."""
    record = patient_collection.find_one({"patient_id": patient_id})
    if not record:
        return jsonify({"error": "Patient not found"}), 404

    mri = record.get("mri", {})
    available = [p for p in ["axial", "coronal", "sagittal"] if p in mri]

    gcam_module = _load_gradcam_module()
    if gcam_module is None:
        return jsonify({
            "available_planes": available,
            "error": "Grad-CAM dependencies not installed"
        }), 200

    ViTGradCAM, overlay_heatmap, overlay_red_highlight, cv2 = gcam_module
    plane = request.args.get("plane", available[0] if available else "axial")
    use_plane = plane if plane in mri else (available[0] if available else None)
    if not use_plane:
        return jsonify({"available_planes": available, "error": "No MRI uploaded"}), 200

    try:
        file_data = fs.get(mri[use_plane]).read()
        volume = np.load(io.BytesIO(file_data), allow_pickle=True)

        slice_idx = volume.shape[0] // 2
        sd = volume[slice_idx].copy().astype(np.float32)
        sd -= sd.min()
        if sd.max() > 0:
            sd /= sd.max()
        sd = (sd * 255).astype(np.uint8)

        from PIL import Image as PILImage
        img = PILImage.fromarray(sd)
        gc = _get_gradcam_instance()
        if gc is None:
            return jsonify({"available_planes": available, "error": "Grad-CAM model failed to load"}), 200
        cam = gc.generate_cam(img)

        threshold = 0.3
        cam_flat = cam.flatten()
        active_pct = float(np.sum(cam_flat >= threshold) / len(cam_flat) * 100)

        import cv2 as _cv2
        cam_up = _cv2.resize(cam, (224, 224), interpolation=_cv2.INTER_CUBIC)

        return jsonify({
            "available_planes": available,
            "plane": use_plane,
            "slice_index": slice_idx,
            "cam_mean": round(float(cam.mean()), 4),
            "cam_max": round(float(cam.max()), 4),
            "cam_min": round(float(cam.min()), 4),
            "active_region_pct": round(active_pct, 1),
            "threshold": threshold,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"available_planes": available, "error": str(e)}), 200


# TDA ANALYSIS (JSON metrics)
@app.route("/tda-analysis/<patient_id>")
def tda_analysis(patient_id):
    plane = request.args.get("plane", None)
    features, _, error = _run_tda_on_patient(patient_id, requested_plane=plane)
    if error:
        return jsonify({"error": error}), 400 if "not found" not in error.lower() else 404
    return jsonify(features)



# TDA PERSISTENCE DIAGRAM (PNG image)
@app.route("/tda-diagram/<patient_id>")
def tda_diagram(patient_id):
    plane = request.args.get("plane", None)
    _, dgms, error = _run_tda_on_patient(patient_id, requested_plane=plane)
    if error:
        return jsonify({"error": error}), 400

    # Larger, styled diagram
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f1715")
    ax.set_facecolor("#111917")

    plot_diagrams(dgms, ax=ax, show=False)

    ax.set_title("Persistence Diagram", fontsize=16, fontweight="bold", color="#d1fae5", pad=12)
    ax.set_xlabel("Birth", fontsize=12, color="#7dd3c0")
    ax.set_ylabel("Death", fontsize=12, color="#7dd3c0")
    ax.tick_params(colors="#7dd3c0", labelsize=10)

    for spine in ax.spines.values():
        spine.set_color("#1f3d36")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150,
                facecolor="#0f1715", edgecolor="none")
    plt.close(fig)
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


# GENERATE AI REPORT (Full Pipeline)
@app.route("/generate-report/<patient_id>", methods=["POST"])
def generate_report(patient_id):
    """
    Full AI pipeline: Load MRI -> Preprocess -> ViT Features -> TDA -> RF Predictions -> Claude Report.
    Returns structured JSON with findings, impressions, technique, predictions.
    """
    # ── Lazy-load AI pipeline on first call ──
    global AI_PIPELINE_AVAILABLE
    if AI_PIPELINE_AVAILABLE is None:
        try:
            if AI_MODULE_DIR not in sys.path:
                sys.path.insert(0, AI_MODULE_DIR)
            global preprocess_volume, extract_vit_features, extract_tda_features
            global generate_ai_report_text, _template_report_text, _get_device
            global joblib, torch
            from generate_report import (
                preprocess_volume,
                extract_vit_features,
                extract_tda_features,
                generate_ai_report_text,
                _template_report_text,
                _get_device,
            )
            import joblib
            import torch
            AI_PIPELINE_AVAILABLE = True
            print("[OK] AI pipeline loaded successfully")
        except ImportError as e:
            AI_PIPELINE_AVAILABLE = False
            print(f"[WARN] AI pipeline not available: {e}")
            print("       Install dependencies: pip install torch transformers joblib scipy httpx")

    if not AI_PIPELINE_AVAILABLE:
        return jsonify({"error": "AI pipeline not available. Check server logs."}), 500

    # Get patient record from MongoDB
    record = patient_collection.find_one({"patient_id": patient_id})
    if not record:
        return jsonify({"error": "Patient not found"}), 404

    mri = record.get("mri", {})
    if not mri:
        return jsonify({"error": "No MRI data found for this patient"}), 404

    # Get patient info from request body or DB record
    body = request.json or {}
    patient_info = {
        "name": body.get("name", record.get("name", "Patient")),
        "age": body.get("age", record.get("age", "Unknown")),
        "gender": body.get("gender", record.get("gender", "Unknown")),
    }

    try:
        device = _get_device()
        print(f"\n{'='*60}")
        print(f"  RadVision - Report Generation for Patient {patient_id}")
        print(f"  Device: {device}")
        print(f"{'='*60}")

        # -- Step 1: Load & preprocess all available MRI planes --
        print("\n  [Step 1/4] Loading and preprocessing MRI volumes...")
        preprocessed_volumes = {}
        views = ["axial", "coronal", "sagittal"]

        for view in views:
            if view not in mri:
                print(f"    ! {view} plane not uploaded - skipping")
                continue
            try:
                file_data = fs.get(mri[view]).read()
                volume = np.load(io.BytesIO(file_data), allow_pickle=True)
                print(f"    Loading {view}: shape={volume.shape}")
                pv = preprocess_volume(volume)
                preprocessed_volumes[view] = pv
                print(f"    OK {view} preprocessed: shape={pv.shape}")
            except Exception as e:
                print(f"    FAIL {view}: {e}")
                continue

        if not preprocessed_volumes:
            return jsonify({"error": "Failed to preprocess any MRI planes"}), 500

        # -- Step 2: Extract ViT features per plane --
        print("\n  [Step 2/4] Extracting ViT features...")
        feat_vectors = {}
        for view, pv in preprocessed_volumes.items():
            print(f"    Extracting ViT features for {view}...")
            feat_vectors[view] = extract_vit_features(pv, device)
            print(f"    OK {view}: {feat_vectors[view].shape[0]} dims")

        # -- Step 3: Extract TDA features per plane --
        print("\n  [Step 3/4] Running TDA analysis...")
        tda_features = {}
        for view, pv in preprocessed_volumes.items():
            print(f"    Computing TDA for {view}...")
            tda_feat, _ = extract_tda_features(pv)
            tda_features[view] = tda_feat
            print(f"    OK {view}: H1={tda_feat.get('num_h1', 0)}, entropy={tda_feat.get('persistence_entropy', 0):.4f}")

        # -- Step 4: Run Random Forest predictions --
        print("\n  [Step 4/4] Running RF predictions...")
        MODEL_DIR = os.path.join(AI_MODULE_DIR, "models")

        # Concatenate ViT features: [axial, coronal, sagittal]
        vit_combined = np.concatenate(list(feat_vectors.values()))
        combined = vit_combined.reshape(1, -1)
        print(f"    Feature vector: {combined.shape[1]} dims")

        predictions = {}
        for task in ["abnormal", "acl", "meniscus"]:
            model_path = os.path.join(MODEL_DIR, f"rf_{task}.joblib")
            if not os.path.exists(model_path):
                print(f"    WARN: No model found for {task}")
                continue

            model = joblib.load(model_path)
            X = combined.copy()

            # Apply scaler if exists
            scaler_path = os.path.join(MODEL_DIR, f"scaler_{task}.joblib")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                X = scaler.transform(X)

            # Load threshold if exists
            threshold = 0.5
            threshold_path = os.path.join(MODEL_DIR, f"threshold_{task}.json")
            if os.path.exists(threshold_path):
                with open(threshold_path) as f:
                    threshold = json.load(f).get("best_threshold", 0.5)

            # Handle feature dimension mismatch
            n_model_features = model.n_features_in_
            n_input_features = X.shape[1]
            if n_input_features != n_model_features:
                print(f"    WARN {task}: model expects {n_model_features} features, got {n_input_features}")
                if n_input_features < n_model_features:
                    X = np.pad(X, ((0, 0), (0, n_model_features - n_input_features)))
                else:
                    X = X[:, :n_model_features]

            prob = float(model.predict_proba(X)[0, 1])
            pred = int(prob >= threshold)
            predictions[task] = {
                "prediction": pred,
                "probability": round(prob, 4),
                "label": "POSITIVE" if pred == 1 else "NEGATIVE",
            }
            print(f"    OK {task}: {predictions[task]['label']} ({prob:.1%})")

        # -- Generate AI report text via Claude API --
        print("\n  Generating clinical report text via AI...")

        # Flatten TDA features for report generator
        tda_flat = {}
        for view, feat in tda_features.items():
            for k, v in feat.items():
                if k not in tda_flat:
                    tda_flat[k] = []
                tda_flat[k].append(float(v))
        tda_summary = {k: round(float(np.mean(v)), 4) for k, v in tda_flat.items()}

        report_text = generate_ai_report_text(predictions, tda_summary, patient_info)

        print(f"\n  Report generated successfully for patient {patient_id}")

        # -- Build response --
        tda_per_plane = {}
        for view, feat in tda_features.items():
            tda_per_plane[view] = {
                "h1_count": feat.get("num_h1", 0),
                "mean_life_h1": round(float(feat.get("mean_life_h1", 0)), 4),
                "persistence_entropy": round(float(feat.get("persistence_entropy", 0)), 4),
            }

        return jsonify({
            "technique": report_text.get("technique", ""),
            "findings": report_text.get("findings", []),
            "impression": report_text.get("impression", []),
            "predictions": predictions,
            "tda_summary": tda_summary,
            "tda_per_plane": tda_per_plane,
            "patient": patient_info,
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Report generation failed: {str(e)}"}), 500

# FINALIZE REPORT
@app.route("/finalize-report", methods=["POST"])
def finalize_report():
    data = request.json
    patient_id = data.get("patientId")
    if not patient_id:
        return jsonify({"error": "Patient ID required"}), 400

    radiologist_name = data.get("radiologistName", "")

    existing = reports_collection.find_one({"patient_id": patient_id})
    update_doc = {
        "status": "Completed",
        "radiologist_name": radiologist_name,
        "finalized_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    if existing:
        reports_collection.update_one(
            {"patient_id": patient_id},
            {"$set": update_doc}
        )
    else:
        update_doc["patient_id"] = patient_id
        update_doc["name"] = data.get("name", "")
        update_doc["age"] = data.get("age", "")
        update_doc["gender"] = data.get("gender", "")
        update_doc["scan_type"] = "Knee MRI"
        update_doc["created_at"] = datetime.utcnow()
        reports_collection.insert_one(update_doc)

    return jsonify({"message": "Report finalized", "status": "Completed"}), 200


# DASHBOARD STATS
@app.route("/dashboard-stats")
def dashboard_stats():
    total = reports_collection.count_documents({})
    completed = reports_collection.count_documents({"status": "Completed"})
    in_progress = reports_collection.count_documents({"status": {"$ne": "Completed"}})
    return jsonify({
        "total": total,
        "completed": completed,
        "pending": in_progress,
    })


# SAVE PROGRESS
@app.route("/save-progress", methods=["POST"])
def save_progress():
    data = request.json

    patient_id = data.get("patientId")
    if not patient_id:
        return jsonify({"error": "Patient ID required"}), 400

    # Check if report already exists for this patient
    existing = reports_collection.find_one({"patient_id": patient_id})

    report_doc = {
        "patient_id": patient_id,
        "name": data.get("name", ""),
        "age": data.get("age", ""),
        "gender": data.get("gender", ""),
        "scan_type": data.get("scanType", "Knee MRI"),
        "status": data.get("status", "In Progress"),
        "tda_metrics": data.get("tdaMetrics", None),
        "updated_at": datetime.utcnow(),
    }

    if existing:
        reports_collection.update_one(
            {"patient_id": patient_id},
            {"$set": report_doc}
        )
        return jsonify({"message": "Progress updated", "patientId": patient_id}), 200
    else:
        report_doc["created_at"] = datetime.utcnow()
        reports_collection.insert_one(report_doc)
        return jsonify({"message": "Progress saved", "patientId": patient_id}), 201


# GET SAVED REPORTS
@app.route("/saved-reports")
def saved_reports():
    reports = list(reports_collection.find({}, {"_id": 0}).sort("updated_at", -1))
    # Convert datetime to string for JSON serialization
    for r in reports:
        if "created_at" in r:
            r["created_at"] = r["created_at"].strftime("%Y-%m-%d %H:%M")
        if "updated_at" in r:
            r["updated_at"] = r["updated_at"].strftime("%Y-%m-%d %H:%M")
    return jsonify(reports)


# LIST PATIENTS
@app.route("/patients")
def patients():
    return jsonify(list(patient_collection.find({}, {"_id": 0})))

# RUN SERVER
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)

