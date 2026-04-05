# evaluate.py
"""
Model Evaluation for Knee MRI Classification
=============================================
Comprehensive evaluation of trained models with visualization.

Metrics:
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC with curve plot
- Confusion Matrix (heatmap)
- Precision-Recall curve
- Per-class metrics
- Cross-validation scores

Usage:
    python evaluate.py --task all --save_plots
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING
# =============================================================================

def load_labels(label_file: Path) -> pd.DataFrame:
    """Load labels from CSV or Excel file."""
    if label_file.suffix == '.csv':
        df = pd.read_csv(label_file, header=None)
    else:
        df = pd.read_excel(label_file, header=None, engine='openpyxl')
    
    df.columns = ['exam_id', 'label']
    df['exam_id'] = df['exam_id'].astype(int)
    return df

def load_data(features_dir: Path,
              labels_dir: Path,
              task: str,
              split: str = 'valid',
              exam_id_offset: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load features and labels for evaluation.

    The most common failure mode (and the one that caused the original error)
    is that combined_features.py was run separately for each split, so the
    exam_ids.csv for the VALID split contains IDs like 0,1,2,... that
    actually correspond to MRNet valid-set exams (1130, 1131, ...).

    Fix strategy (applied in order):
      1. Try a direct integer match between exam_ids.csv and the label file.
      2. If that yields 0 matches, check if the label IDs form a contiguous
         block and the feature IDs are 0-based sequential indices.  In that
         case apply an automatic offset (label_min - feature_min) so that
         feature row i maps to exam_id (i + offset).
      3. If exam_id_offset is supplied explicitly, use it unconditionally.
    """
    # Check if features are stored in a split-specific subdirectory
    target_features_dir = features_dir / split if (features_dir / split).exists() else features_dir

    # Load features
    X = np.load(target_features_dir / "X_combined.npy")
    exam_ids_df = pd.read_csv(target_features_dir / "exam_ids.csv")

    # Clean exam IDs: strip view suffixes that combined_features.py may have left
    exam_ids = exam_ids_df['exam_id'].astype(str)
    for suffix in ['_sagittal', '_axial', '_coronal', '.npy']:
        exam_ids = exam_ids.str.replace(suffix, '', regex=False)
    exam_ids = exam_ids.astype(int)

    # Load labels
    label_file = labels_dir / f"{split}-{task}.csv"
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")
    labels_df = load_labels(label_file)

    def _merge(ids: pd.Series) -> pd.DataFrame:
        return (
            pd.DataFrame({
                'exam_id': ids,
                'feature_idx': np.arange(len(ids))
            })
            .merge(labels_df, on='exam_id', how='inner')
        )

    merged = _merge(exam_ids)

    # --- Auto-offset fallback ---
    if len(merged) == 0:
        label_ids = labels_df['exam_id'].values
        feat_ids  = exam_ids.values

        # Determine offset to try
        if exam_id_offset is not None:
            offset = exam_id_offset
            print(f"  [INFO] Using explicit exam_id_offset={offset}")
        else:
            # Heuristic: if feature IDs look sequential from 0 and label IDs
            # are a shifted block of the same size, infer the offset.
            feat_min, feat_max = int(feat_ids.min()), int(feat_ids.max())
            label_min = int(label_ids.min())
            offset = label_min - feat_min
            print(f"  [INFO] Direct ID match failed. "
                  f"Feature IDs: [{feat_min}..{feat_max}], "
                  f"Label IDs start at {label_min}. "
                  f"Auto-applying offset={offset}.")

        shifted_ids = exam_ids + offset
        merged = _merge(shifted_ids)

        if len(merged) > 0:
            print(f"  [INFO] Offset fix succeeded: {len(merged)} samples matched.")
        else:
            # Give a detailed diagnostic before raising
            raise ValueError(
                f"No matching samples for {split}-{task} even after offset correction.\n"
                f"  -> Features loaded from : {target_features_dir}\n"
                f"  -> Feature IDs (first 5): {exam_ids.tolist()[:5]}\n"
                f"  -> Shifted IDs (first 5): {shifted_ids.tolist()[:5]}\n"
                f"  -> Label  IDs (first 5) : {labels_df['exam_id'].tolist()[:5]}\n"
                f"\n"
                f"  ROOT CAUSE: Your combined_features directory likely contains\n"
                f"  features extracted from the TRAINING split, not '{split}'.\n"
                f"  Re-run combined_features.py (and extract_vit.py / tda.py) on\n"
                f"  the '{split}' images and point --features_dir at that output."
            )

    X_matched = X[merged['feature_idx'].values]
    y = merged['label'].values
    matched_exam_ids = merged['exam_id'].values

    return X_matched, y, matched_exam_ids

# =============================================================================
# EVALUATION & PLOTTING FUNCTIONS
# =============================================================================

def compute_metrics(y_true: np.ndarray, 
                    y_pred: np.ndarray,
                    y_prob: np.ndarray) -> Dict:
    """Compute comprehensive evaluation metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_true, y_prob)),
        'average_precision': float(average_precision_score(y_true, y_prob)),
        'specificity': float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'n_samples': len(y_true),
        'n_positive': int(sum(y_true)),
        'n_negative': int(len(y_true) - sum(y_true))
    }
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    return metrics

def plot_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, 
                            task: str, output_dir: Path):
    """Generates and saves ROC, PR, and Confusion Matrix plots."""
    plt.figure(figsize=(18, 5))
    
    # 1. Confusion Matrix
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'{task.capitalize()} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 2. ROC Curve
    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{task.capitalize()} - ROC Curve')
    plt.legend(loc="lower right")
    
    # 3. Precision-Recall Curve
    plt.subplot(1, 3, 3)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {ap:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{task.capitalize()} - Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plot_path = output_dir / f"{task}_evaluation_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved evaluation plots to: {plot_path}")

# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

def evaluate_task(model_path: Path,
                  X: np.ndarray,
                  y: np.ndarray,
                  task: str,
                  output_dir: Path,
                  save_plots: bool = True) -> Dict:
    """Evaluate a single task."""
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {task.upper()}")
    print(f"{'='*60}")
    
    if not model_path.exists():
        print(f"  [ERROR] Model not found: {model_path}")
        return {}
    
    model = joblib.load(model_path)
    print(f"  Model: {model_path.name}")
    print(f"  Samples: {len(y)} (Pos: {sum(y)}, Neg: {len(y)-sum(y)})")
    
    scaler_path = model_path.parent / f"scaler_{task}.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
        print(f"  Applied scaler: {scaler_path.name}")
    
    y_prob = model.predict_proba(X)[:, 1]
    
    threshold_path = model_path.parent / f"threshold_{task}.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold = json.load(f)["best_threshold"]
        print(f"  Using saved optimal threshold: {threshold:.4f}")
    else:
        threshold = 0.5
        print(f"  Using default threshold: {threshold}")
    
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = compute_metrics(y, y_pred, y_prob)
    metrics['threshold'] = threshold
    
    print(f"\n  Results:")
    print(f"    Threshold:   {threshold:.4f}")
    print(f"    Accuracy:    {metrics['accuracy']:.4f}")
    print(f"    Precision:   {metrics['precision']:.4f}")
    print(f"    Recall:      {metrics['recall']:.4f}")
    print(f"    F1-Score:    {metrics['f1']:.4f}")
    print(f"    AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y, y_pred, target_names=['Negative', 'Positive'], zero_division=0))
    
    if save_plots:
        plot_evaluation_metrics(y, y_pred, y_prob, task, output_dir)
        
    # Save metrics to JSON
    results = {
        'metrics': metrics,
        'y_true': y.tolist(),
        'y_pred': y_pred.tolist(),
        'y_prob': y_prob.tolist()
    }
    
    with open(output_dir / f"{task}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return results

def main(args):
    """Main entry point for evaluation."""
    model_dir = Path(args.model_dir)
    features_dir = Path(args.features_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = ['abnormal', 'acl', 'meniscus'] if args.task == 'all' else [args.task]
    
    print("\n" + "=" * 60)
    print("  RADVISION MODEL EVALUATION")
    print("=" * 60)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Tasks: {', '.join(tasks)}")
    print(f"  Split: {args.split}")
    print("=" * 60)
    
    for task in tasks:
        try:
            model_files = list(model_dir.glob(f"*_{task}.joblib"))
            if not model_files:
                print(f"\n  [WARN] No model found for {task}")
                continue
            
            model_path = model_files[0]
            X, y, _ = load_data(features_dir, labels_dir, task, args.split,
                                 exam_id_offset=args.exam_id_offset)
            evaluate_task(model_path, X, y, task, output_dir, args.save_plots)
            
        except Exception as e:
            print(f"\n  [ERROR] Failed to evaluate {task}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained knee MRI classification models"
    )
    _script_dir = Path(__file__).parent.absolute()
    parser.add_argument("--model_dir", default=str(_script_dir / "models"))
    parser.add_argument("--features_dir", default=str(_script_dir / "combined_features"))
    parser.add_argument("--labels_dir", default="/Users/niceliju/Dev/MRNet-v1.0")
    parser.add_argument("--output_dir", default=str(_script_dir / "evaluation"))
    parser.add_argument("--task", default="all", choices=["abnormal", "acl", "meniscus", "all"])
    parser.add_argument("--split", default="valid", choices=["train", "valid"])
    parser.add_argument("--save_plots", action="store_true", help="Generate and save evaluation plots")
    parser.add_argument(
        "--exam_id_offset", type=int, default=None,
        help=(
            "Manual offset to add to feature exam IDs when they don't match label IDs. "
            "E.g. if features are indexed 0-N but valid labels start at 1130, pass --exam_id_offset 1130. "
            "If omitted, the offset is inferred automatically."
        )
    )
    
    args = parser.parse_args()
    main(args)