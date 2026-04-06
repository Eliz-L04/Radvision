# combined_features.py
"""
Combined Features for Multi-View Knee MRI
=========================================
Combines ViT embeddings and TDA features, supporting both single-view 
and multi-view (axial + coronal + sagittal) training.

Multi-view late fusion:
- ViT_axial (768) + ViT_coronal (768) + ViT_sagittal (768) = 2304-dim
- TDA_axial (7) + TDA_coronal (7) + TDA_sagittal (7) = 21-dim
- Total: 2325-dim feature vector per sample

Usage:
    python combined_features.py --all_views  # Multi-view training
    python combined_features.py --view axial  # Single-view training
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import joblib


# =============================================================================
# FEATURE LOADING
# =============================================================================

def load_vit_features(vit_dir: Path, view: str = None) -> pd.DataFrame:
    """
    Load ViT embeddings into a DataFrame.
    
    Args:
        vit_dir: Directory containing ViT embeddings
        view: If specified, look in vit_dir/view subdirectory
        
    Returns:
        DataFrame with exam_id and vit_features columns
    """
    if view:
        search_dir = vit_dir / view
    else:
        search_dir = vit_dir
    
    if not search_dir.exists():
        print(f"[WARN] ViT directory not found: {search_dir}")
        return pd.DataFrame()
    
    files = sorted(search_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No ViT embeddings found in {search_dir}")

    records = []
    for f in files:
        arr = np.load(f)
        
        # Extract exam ID from filename like "0000_axial_vit.npy" → "0000"
        # extract_vit.py saves as: {exam_id}_{plane}_vit.npy
        exam_id = f.stem  # e.g. "0000_axial_vit"
        # Strip "_vit" suffix first, then strip plane suffix if present
        for suffix in ['_vit', '_axial', '_coronal', '_sagittal']:
            if exam_id.endswith(suffix):
                exam_id = exam_id[: -len(suffix)]
        exam_id = exam_id.zfill(4)
        
        records.append({
            'exam_id': exam_id,
            'vit_features': arr
        })
    
    df = pd.DataFrame(records)
    print(f"  Loaded {len(df)} ViT embeddings" + (f" for {view}" if view else ""))
    return df


def load_tda_features(tda_csv: Path) -> pd.DataFrame:
    """Load TDA feature CSV."""
    if not tda_csv.exists():
        print(f"[WARN] TDA file not found: {tda_csv}")
        return pd.DataFrame()
    
    df = pd.read_csv(tda_csv)
    if "exam_id" not in df.columns:
        raise ValueError("TDA CSV must contain 'exam_id' column")
    
    df['exam_id'] = df['exam_id'].astype(str).str.zfill(4)
    print(f"  Loaded TDA features: {df.shape}")
    return df


# =============================================================================
# SINGLE-VIEW COMBINATION
# =============================================================================

def combine_single_view(vit_dir: Path, tda_csv: Path, view: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine ViT and TDA features for a single view.
    
    Returns:
        exam_ids, X_combined
    """
    vit_df = load_vit_features(vit_dir, view)
    tda_df = load_tda_features(tda_csv)
    
    if tda_df.empty:
        # No TDA features - use only ViT
        exam_ids = vit_df["exam_id"].values
        X_combined = np.stack(vit_df["vit_features"].values)
        return exam_ids, X_combined
    
    merged = pd.merge(vit_df, tda_df, on="exam_id", how="inner")
    
    # ViT features
    vit_features = np.stack(merged["vit_features"].values)
    
    # TDA features - use all available numeric columns (tda.py produces dynamic
    # column names depending on aggregation mode, e.g. h0_count_mean, h1_entropy_std…)
    available_tda_cols = [
        c for c in merged.columns
        if c not in ['exam_id', 'view', 'vit_features']
    ]
    
    tda_features = merged[available_tda_cols].values.astype(np.float32)
    print(f"[INFO] Using {len(available_tda_cols)} TDA feature columns")

    X_combined = np.concatenate([vit_features, tda_features], axis=1)
    print(f"[INFO] Combined features shape: {X_combined.shape}")
    
    return merged["exam_id"].values, X_combined


# =============================================================================
# MULTI-VIEW COMBINATION (Late Fusion)
# =============================================================================

def combine_multi_view(vit_dir: Path, tda_dir: Path, 
                       views: List[str] = ["axial", "coronal", "sagittal"]
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine features from all views using late fusion.
    
    Late Fusion Strategy:
    - Concatenate ViT embeddings: [ViT_axial | ViT_coronal | ViT_sagittal]
    - Concatenate TDA features from all views
    - Result: (768*3 + 7*3) = 2325-dim feature vector
    
    Returns:
        exam_ids, X_combined
    """
    print("\n" + "=" * 60)
    print("  MULTI-VIEW FEATURE COMBINATION")
    print("=" * 60)
    print(f"  Views: {', '.join(views)}")
    
    # Load ViT features for each view
    vit_dfs = {}
    for view in views:
        df = load_vit_features(vit_dir, view)
        if not df.empty:
            df = df.rename(columns={'vit_features': f'vit_{view}'})
            vit_dfs[view] = df
    
    if not vit_dfs:
        raise ValueError("No ViT features found for any view")
    
    # Merge all ViT features on exam_id
    merged = None
    for view, df in vit_dfs.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on='exam_id', how='inner')
    
    print(f"  After ViT merge: {len(merged)} samples with all views")
    
    # Load and merge TDA features
    tda_file = tda_dir / "tda_features_combined.csv"
    if not tda_file.exists():
        # Try individual view files
        for view in views:
            view_tda = tda_dir / f"tda_features_{view}.csv"
            if view_tda.exists():
                tda_df = pd.read_csv(view_tda)
                tda_df['exam_id'] = tda_df['exam_id'].astype(str).str.zfill(4)
                # Rename columns with view suffix
                rename_map = {c: f"{c}_{view}" for c in tda_df.columns if c not in ['exam_id', 'view']}
                tda_df = tda_df.rename(columns=rename_map)
                merged = merged.merge(tda_df.drop(columns=['view'], errors='ignore'), 
                                       on='exam_id', how='left')
    else:
        tda_df = pd.read_csv(tda_file)
        tda_df['exam_id'] = tda_df['exam_id'].astype(str).str.zfill(4)
        merged = merged.merge(tda_df, on='exam_id', how='left')
    
    # Fill NaN TDA values with 0
    merged = merged.fillna(0)
    
    print(f"  Final merged samples: {len(merged)}")
    
    # Build feature matrix
    feature_list = []
    
    # Add ViT features for each view
    for view in views:
        col = f'vit_{view}'
        if col in merged.columns:
            vit_features = np.stack(merged[col].values)
            feature_list.append(vit_features)
            print(f"  ViT {view}: {vit_features.shape}")
    
    # Add TDA features
    tda_cols = [c for c in merged.columns if c not in ['exam_id'] + [f'vit_{v}' for v in views]]
    if tda_cols:
        tda_features = merged[tda_cols].values.astype(np.float32)
        feature_list.append(tda_features)
        print(f"  TDA features: {tda_features.shape}")
    
    # Concatenate all features
    X_combined = np.concatenate(feature_list, axis=1)
    print(f"\n  Final combined shape: {X_combined.shape}")
    print(f"  Breakdown: {len(views)} views × 768 ViT + {len(tda_cols)} TDA = {X_combined.shape[1]} features")
    
    return merged["exam_id"].values, X_combined


# =============================================================================
# PCA DIMENSIONALITY REDUCTION
# =============================================================================

def apply_pca(X: np.ndarray, n_components: int = 256, 
              output_dir: Optional[Path] = None) -> np.ndarray:
    """Apply PCA for dimensionality reduction."""
    print(f"[INFO] Applying PCA: {X.shape[1]} → {n_components} dimensions")
    
    # Standardize first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    
    print(f"[INFO] Explained variance: {sum(pca.explained_variance_ratio_)*100:.1f}%")
    print(f"[INFO] Reduced shape: {X_reduced.shape}")
    
    # Save models
    if output_dir:
        joblib.dump(scaler, output_dir / "scaler.joblib")
        joblib.dump(pca, output_dir / "pca_model.joblib")
        print(f"[INFO] Saved scaler and PCA to {output_dir}")
    
    return X_reduced


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  FEATURE COMBINATION PIPELINE")
    print("=" * 60)
    
    if args.all_views:
        # Multi-view combination
        exam_ids, X_combined = combine_multi_view(
            vit_dir=Path(args.vit_dir),
            tda_dir=Path(args.tda_dir),
            views=["axial", "coronal", "sagittal"]
        )
    else:
        # Single-view combination
        exam_ids, X_combined = combine_single_view(
            vit_dir=Path(args.vit_dir),
            tda_csv=Path(args.tda_csv),
            view=args.view
        )

    if args.use_pca:
        X_combined = apply_pca(X_combined, n_components=args.n_components, output_dir=out_dir)

    # Save combined features
    np.save(out_dir / "X_combined.npy", X_combined)
    pd.DataFrame({"exam_id": exam_ids}).to_csv(out_dir / "exam_ids.csv", index=False)

    print(f"\n✅ Saved combined features to {out_dir}")
    print(f"   X_combined.npy: {X_combined.shape}")
    print(f"   exam_ids.csv: {len(exam_ids)} samples")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Combine ViT + TDA features (single or multi-view)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-view training (RECOMMENDED)
  python combined_features.py --all_views --use_pca

  # Single-view (axial only)
  python combined_features.py --view axial
        """
    )
    
    # Get script directory for Mac-compatible relative defaults
    _script_dir = Path(__file__).parent.absolute()
    p.add_argument("--vit_dir", default=str(_script_dir / "embeddings_vit"),
                   help="Root directory containing ViT embeddings (with view subdirs)")
    p.add_argument("--tda_dir", default=str(_script_dir / "tda_features"),
                   help="Directory containing TDA feature CSVs (for multi-view mode)")
    p.add_argument("--tda_csv", default=str(_script_dir / "tda_features" / "tda_features_axial.csv"),
                   help="TDA CSV file for single-view mode. tda.py saves per-view files as tda_features_{view}.csv")
    p.add_argument("--out_dir", default=str(_script_dir / "combined_features"),
                   help="Output directory for combined features")
    p.add_argument("--view", default="axial", choices=["axial", "coronal", "sagittal"],
                   help="Single view to use (ignored if --all_views)")
    p.add_argument("--all_views", action="store_true",
                   help="Use all three views (late fusion)")
    p.add_argument("--use_pca", action="store_true", 
                   help="Apply PCA for dimensionality reduction")
    p.add_argument("--n_components", type=int, default=256, 
                   help="Number of PCA components")
    
    args = p.parse_args()
    main(args)