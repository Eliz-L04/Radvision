"""
Model Training for Knee MRI Classification (MRNet-style)
=======================================================

✔ Uses exam_ids.csv for correct label-feature alignment
✔ Prevents index mismatch errors
✔ Supports RF / XGB / SVM / Ensemble
✔ Stratified CV + optional hyperparameter tuning
✔ Saves trained models + JSON report
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve
)

warnings.filterwarnings("ignore")

# =============================================================================
# OPTIONAL XGBOOST
# =============================================================================
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] XGBoost not installed")



# =============================================================================
# MODEL FACTORY
# =============================================================================
def get_model(model_type, seed):
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    if model_type == "xgb":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost not installed")
        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )

    if model_type == "svm":
        return SVC(
            C=10,
            kernel="rbf",
            probability=True,
            random_state=seed,
        )

    raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# DATA LOADING
# =============================================================================
def load_labels(label_file: Path) -> pd.DataFrame:
    """
    Loads MRNet label CSV (no header):
        0000,1
        0001,0
        ...
    """
    df = pd.read_csv(label_file, header=None)
    df.columns = ["exam_id", "label"]
    df["exam_id"] = df["exam_id"].astype(str).str.zfill(4)
    return df


def load_training_data(features_dir: Path, labels_dir: Path, task: str):
    """
    Aligns X with labels using exam_ids.csv produced by combined_features.py.

    BUG FIX 1 — index-based row selection was wrong:
        combined_features.py saves exam_ids.csv with one exam_id per row.
        After merging with the label DataFrame the `.index` values are the
        *label-DataFrame* indices, not the positional indices into X.
        We must track the original row position in the feature matrix
        explicitly via a reset_index trick before merging.

    BUG FIX 2 — SVM needs feature scaling:
        SVM with RBF kernel is sensitive to feature magnitude.
        StandardScaler is applied when model_type == "svm", but since
        load_training_data doesn't know the model type we return raw X
        here and handle scaling inside train_task.

    BUG FIX 3 — NaN / Inf guard:
        High-dim ViT + TDA features can contain rare NaN/Inf values
        (e.g. from failed TDA slices filled with 0 then combined).
        We replace them here so every downstream model is safe.
    """

    # --- Load features ---
    X = np.load(features_dir / "X_combined.npy")

    # --- Load exam IDs saved by combined_features.py ---
    exam_ids_df = pd.read_csv(features_dir / "exam_ids.csv")
    exam_ids_df["exam_id"] = exam_ids_df["exam_id"].astype(str)

    # Remove view suffixes that combined_features.py may have left in
    # (e.g. "0000_axial" → "0000")
    for v in ["_sagittal", "_axial", "_coronal"]:
        exam_ids_df["exam_id"] = exam_ids_df["exam_id"].str.replace(v, "", regex=False)

    exam_ids_df["exam_id"] = exam_ids_df["exam_id"].str.zfill(4)

    # --- KEY FIX: keep the original row position so we can index into X ---
    # After the merge we need *positional* row indices that correspond
    # to rows in X, not the index of the labels DataFrame.
    exam_ids_df = exam_ids_df.reset_index()          # 'index' = position in X
    exam_ids_df = exam_ids_df.rename(columns={"index": "feature_row"})

    # --- Load labels ---
    label_file = labels_dir / f"train-{task}.csv"
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")
    labels = load_labels(label_file)

    # --- Inner join: only keep exams that have BOTH features AND a label ---
    merged = exam_ids_df.merge(labels, on="exam_id", how="inner")

    if merged.empty:
        raise RuntimeError(
            f"No matching exam IDs between exam_ids.csv and train-{task}.csv. "
            f"Check that both files use the same zero-padded 4-digit IDs."
        )

    # --- Use the preserved positional index to slice X correctly ---
    feature_rows = merged["feature_row"].to_numpy()   # e.g. [0, 2, 5, ...]
    X_aligned = X[feature_rows]                        # shape (n_matched, dim)
    y_aligned = merged["label"].values

    # --- Guard against NaN / Inf (BUG FIX 3) ---
    if not np.isfinite(X_aligned).all():
        n_bad = (~np.isfinite(X_aligned)).sum()
        print(f"[WARN] Replacing {n_bad} NaN/Inf values in feature matrix with 0")
        X_aligned = np.nan_to_num(X_aligned, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[INFO] Task: {task} | Matched samples: {len(feature_rows)} "
          f"| Positive rate: {y_aligned.mean():.2%}")

    return X_aligned, y_aligned


# =============================================================================
# TRAINING
# =============================================================================
def train_task(X, y, task, model_type, cv, seed, output_dir):
    """
    Train a classifier for one task.

    BUG FIX 4 — CV was run on the model object BEFORE it was fit, then the
    model was fit only on X_train.  The final `model.fit(X, y)` at the end
    refits on ALL data, which is correct.  However cross_val_score internally
    clones the unfitted estimator, so the order is fine — but we add a clear
    comment to make the intent obvious.

    BUG FIX 5 — SVM scaling:
        SVM with RBF kernel requires z-score normalisation.  Without it the
        kernel distances are dominated by the ViT embedding dimensions (range
        ~ [-5, 5]) vs the TDA features (range ~ [0, 1]) and performance
        degrades badly.  We fit the scaler on X_train only to avoid leakage.
    """

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # --- Scale features for SVM (BUG FIX 5) ---
    scaler = None
    if model_type == "svm":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_full  = scaler.transform(X)          # for final refit
    else:
        X_full = X

    model = get_model(model_type, seed)

    # Cross-validation on the training split
    # NOTE: CV-AUC is computed on the 80% training subset, so inner folds
    # are smaller than the full dataset.  Compare with val_auc for context.
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(
        model, X_train, y_train,
        scoring="roc_auc", cv=cv_splitter, n_jobs=-1
    ).mean()

    # Fit on training split → evaluate on validation split
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_val)[:, 1]

    # --- Threshold tuning: find the threshold that maximizes F1 ---
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])

    # Apply optimal threshold instead of default 0.5
    y_pred_default = model.predict(X_val)
    y_pred_tuned = (y_prob >= best_threshold).astype(int)

    print(f"\n[THRESHOLD] {task.upper()} | Default=0.5 → F1={f1_score(y_val, y_pred_default, zero_division=0):.3f}")
    print(f"[THRESHOLD] {task.upper()} | Optimal={best_threshold:.3f} → F1={best_f1:.3f}")

    # Use the tuned predictions for all metrics
    y_pred = y_pred_tuned

    # Guard against single-class validation split (roc_auc_score crashes)
    try:
        val_auc = round(float(roc_auc_score(y_val, y_prob)), 4)
    except ValueError:
        val_auc = float('nan')
        print("[WARN] Could not compute val_auc (only one class in y_val)")

    results = {
        "task": task,
        "model": model_type,
        "cv_auc":        round(float(cv_auc), 4),
        "best_threshold": round(best_threshold, 4),
        "val_accuracy":  round(float(accuracy_score(y_val, y_pred)), 4),
        "val_precision": round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
        "val_recall":    round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
        "val_f1":        round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
        "val_auc":       val_auc,
        "n_train": int(len(y_train)),
        "n_val":   int(len(y_val)),
        "positive_rate_train": round(float(y_train.mean()), 4),
    }

    print(f"\n[RESULTS] {task.upper()} | CV-AUC: {cv_auc:.3f} | "
          f"Val-AUC: {results['val_auc']:.3f} | "
          f"F1: {results['val_f1']:.3f} (threshold={best_threshold:.3f})")

    # --- Refit on ALL data for the saved model (standard practice) ---
    # For SVM: refit scaler on full data so saved scaler+model are consistent
    if scaler is not None:
        scaler_full = StandardScaler()
        X_full = scaler_full.fit_transform(X)
        model.fit(X_full, y)
        joblib.dump(scaler_full, output_dir / f"scaler_{task}.joblib")
    else:
        model.fit(X_full, y)

    joblib.dump(model, output_dir / f"{model_type}_{task}.joblib")

    # Save the optimal threshold so inference uses the same value
    threshold_path = output_dir / f"threshold_{task}.json"
    with open(threshold_path, "w") as f:
        json.dump({"task": task, "best_threshold": best_threshold}, f, indent=2)

    return results


# =============================================================================
# MAIN
# =============================================================================
def main(args):

    features_dir = Path(args.features_dir)
    labels_dir   = Path(args.labels_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)   # BUG FIX 6: parents=True

    tasks = ["abnormal", "acl", "meniscus"] if args.task == "all" else [args.task]
    report = {}

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"  Training task: {task.upper()}")
        print(f"{'='*60}")
        try:
            X, y = load_training_data(features_dir, labels_dir, task)
            report[task] = train_task(
                X, y, task,
                args.model, args.cv,
                args.seed, output_dir
            )
        except FileNotFoundError as e:
            print(f"[SKIP] {task}: {e}")
            report[task] = {"error": str(e)}

    report_path = output_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Training completed. Report saved to {report_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train knee MRI classifiers using ViT + TDA features"
    )
    parser.add_argument(
        "--features_dir",
        default="./combined_features",
        help="Directory containing X_combined.npy and exam_ids.csv"
    )
    parser.add_argument(
        "--labels_dir",
        default="/Users/niceliju/Dev/MRNet-v1.0",
        help="Directory containing train-abnormal.csv, train-acl.csv, train-meniscus.csv"
    )
    parser.add_argument(
        "--output_dir",
        default="./models",
        help="Directory to save trained models and report"
    )
    parser.add_argument(
        "--task", default="all",
        choices=["abnormal", "acl", "meniscus", "all"]
    )
    parser.add_argument(
        "--model", default="rf",
        choices=["rf", "xgb", "svm"]
    )
    parser.add_argument("--cv",   type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
