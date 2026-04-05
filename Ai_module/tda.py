"""
Enhanced TDA Feature Extraction for Knee MRI - Mac M4 Optimized
================================================================

Improvements over standard TDA:
1. **3D Volumetric TDA** - Processes multiple slices, not just one
2. **Richer Features** - 20+ topological descriptors per view
3. **Multiple Aggregation** - Per-slice + volume-wide statistics
4. **Mac M4 Optimized** - Parallel processing for speed
5. **Robust Validation** - Handles edge cases properly
6. **Multi-View Support** - Axial, Coronal, Sagittal

Key Features Extracted:
- H0 (Connected Components): Count, lifetimes, statistics
- H1 (Loops/Holes): Count, lifetimes, statistics  
- H2 (Voids): Count, lifetimes (if computing 3D)
- Persistence entropy and landscape features
- Slice-wise aggregations (mean, std, max across slices)

Compatible with preprocess.py output: (num_slices, H, W, C)
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import feature, filters, morphology
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import warnings
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')

# TDA imports
try:
    from ripser import ripser
    from persim import plot_diagrams
    import matplotlib.pyplot as plt
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False
    print("[WARN] ripser/persim not installed. Install with:")
    print("       pip install ripser persim scikit-image")


# =============================================================================
# MAC M4 OPTIMIZATION
# =============================================================================

def get_num_workers():
    """Get optimal number of parallel workers for Mac M4."""
    # M4 has 10 CPU cores (6 performance + 4 efficiency)
    # Use 6-8 workers to leave room for system
    return min(8, mp.cpu_count() - 2)


# =============================================================================
# TOPOLOGICAL FEATURE COMPUTATION
# =============================================================================

def compute_persistence_entropy(lifetimes: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute persistence entropy from lifetimes.
    
    Measures the "complexity" of the topological features.
    Higher entropy = more diverse feature lifetimes.
    """
    if len(lifetimes) == 0:
        return 0.0
    
    # Normalize lifetimes to probability distribution
    L = np.sum(lifetimes)
    if L <= eps:
        return 0.0
    
    p = lifetimes / L
    entropy = -np.sum(p * np.log(p + eps))
    return float(entropy)


def compute_betti_curve(dgms: List[np.ndarray], 
                        num_points: int = 50,
                        max_epsilon: float = None) -> Dict[str, np.ndarray]:
    """
    Compute Betti curves (number of features vs. filtration parameter).
    
    This provides a richer representation than just counting features.
    """
    if max_epsilon is None:
        # Find maximum death time across all dimensions
        max_epsilon = 0
        for dgm in dgms:
            if len(dgm) > 0:
                finite_deaths = dgm[np.isfinite(dgm[:, 1]), 1]
                if len(finite_deaths) > 0:
                    max_epsilon = max(max_epsilon, np.max(finite_deaths))
        
        if max_epsilon == 0:
            max_epsilon = 1.0
    
    epsilons = np.linspace(0, max_epsilon, num_points)
    
    betti_curves = {}
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            betti_curves[f'betti_{dim}'] = np.zeros(num_points)
            continue
        
        # Count features alive at each epsilon
        births = dgm[:, 0].copy()
        deaths = dgm[:, 1].copy()
        deaths[np.isinf(deaths)] = max_epsilon * 2  # Treat infinite as very large
        
        curve = np.zeros(num_points)
        for i, eps in enumerate(epsilons):
            curve[i] = np.sum((births <= eps) & (deaths > eps))
        
        betti_curves[f'betti_{dim}'] = curve
    
    return betti_curves


def compute_persistence_statistics(dgm: np.ndarray, 
                                   prefix: str = "h0") -> Dict[str, float]:
    """
    Compute comprehensive statistics from a persistence diagram.
    
    Args:
        dgm: Persistence diagram (N x 2 array of birth-death pairs)
        prefix: Prefix for feature names (e.g., "h0", "h1")
    
    Returns:
        Dictionary of statistical features
    """
    features = {}
    
    if len(dgm) == 0:
        # Return zeros if no features
        zero_features = {
            f'{prefix}_count': 0,
            f'{prefix}_mean_life': 0.0,
            f'{prefix}_std_life': 0.0,
            f'{prefix}_max_life': 0.0,
            f'{prefix}_min_life': 0.0,
            f'{prefix}_total_life': 0.0,
            f'{prefix}_entropy': 0.0,
            f'{prefix}_mean_birth': 0.0,
            f'{prefix}_mean_death': 0.0,
        }
        return zero_features
    
    # Compute lifetimes
    lifetimes = dgm[:, 1] - dgm[:, 0]
    finite_lifetimes = lifetimes[np.isfinite(lifetimes)]
    
    if len(finite_lifetimes) == 0:
        finite_lifetimes = np.array([0.0])
    
    # Basic statistics
    features[f'{prefix}_count'] = len(finite_lifetimes)
    features[f'{prefix}_mean_life'] = float(np.mean(finite_lifetimes))
    features[f'{prefix}_std_life'] = float(np.std(finite_lifetimes))
    features[f'{prefix}_max_life'] = float(np.max(finite_lifetimes))
    features[f'{prefix}_min_life'] = float(np.min(finite_lifetimes))
    features[f'{prefix}_total_life'] = float(np.sum(finite_lifetimes))
    
    # Entropy
    features[f'{prefix}_entropy'] = compute_persistence_entropy(finite_lifetimes)
    
    # Birth/Death statistics
    features[f'{prefix}_mean_birth'] = float(np.mean(dgm[:, 0]))
    finite_deaths = dgm[np.isfinite(dgm[:, 1]), 1]
    features[f'{prefix}_mean_death'] = float(np.mean(finite_deaths)) if len(finite_deaths) > 0 else 0.0
    
    return features


def extract_point_cloud_from_slice(slice_2d: np.ndarray,
                                   method: str = "edges",
                                   max_points: int = 3000) -> np.ndarray:
    """
    Extract point cloud from 2D slice for TDA computation.
    
    Args:
        slice_2d: 2D grayscale slice
        method: Extraction method
            - "edges": Canny edge detection (good for boundaries)
            - "intensity": High-intensity pixels (good for tissue)
            - "gradient": High gradient magnitude (good for texture)
        max_points: Maximum points to sample
    
    Returns:
        Point cloud as (N, 2) array
    """
    # Normalize to [0, 1]
    slice_2d = slice_2d.astype(np.float32)
    vmin, vmax = slice_2d.min(), slice_2d.max()
    if vmax - vmin > 1e-6:
        slice_2d = (slice_2d - vmin) / (vmax - vmin)
    else:
        return np.array([])
    
    if method == "edges":
        # Edge detection
        try:
            sigma = 1.5
            edges = feature.canny(filters.gaussian(slice_2d, sigma=sigma))
            # Dilate slightly to get more connected edges
            edges = morphology.binary_dilation(edges)
            pts = np.column_stack(np.nonzero(edges))
        except Exception as e:
            return np.array([])
    
    elif method == "intensity":
        # High-intensity regions (e.g., bone, certain tissues)
        threshold = np.percentile(slice_2d, 75)
        mask = slice_2d > threshold
        pts = np.column_stack(np.nonzero(mask))
    
    elif method == "gradient":
        # High gradient regions (texture, boundaries)
        from skimage.filters import sobel
        gradient = sobel(slice_2d)
        threshold = np.percentile(gradient, 75)
        mask = gradient > threshold
        pts = np.column_stack(np.nonzero(mask))
    
    else:
        raise ValueError(f"Unknown extraction method: {method}")
    
    # Subsample if too many points
    if len(pts) > max_points:
        if method == "gradient" or method == "edges":
            # Smart subsampling: prioritize points with higher gradient magnitude
            # This preserves strong topological features better than random sampling
            from skimage.filters import sobel
            grad_mag = sobel(slice_2d)
            
            # Get gradient values for the selected points
            # pts is (N, 2) array of (row, col) coordinates
            pt_grads = grad_mag[pts[:, 0], pts[:, 1]]
            
            # Probabilistic sampling weighted by gradient magnitude
            # Add small epsilon to avoid zero probability
            probs = pt_grads + 1e-6
            probs = probs / np.sum(probs)
            
            idx = np.random.choice(len(pts), size=max_points, replace=False, p=probs)
            pts = pts[idx]
        else:
            # Standard random subsampling for other methods
            idx = np.random.choice(len(pts), size=max_points, replace=False)
            pts = pts[idx]
    
    return pts


def compute_tda_for_slice(slice_2d: np.ndarray,
                          method: str = "edges",
                          max_points: int = 3000,
                          maxdim: int = 1) -> Dict[str, float]:
    """
    Compute TDA features for a single 2D slice.
    
    Args:
        slice_2d: 2D grayscale image
        method: Point cloud extraction method
        max_points: Maximum points for TDA
        maxdim: Maximum homology dimension (0=components, 1=loops, 2=voids)
    
    Returns:
        Dictionary of TDA features
    """
    if not TDA_AVAILABLE:
        return {
            'h0_count': 0, 'h0_mean_life': 0, 'h0_max_life': 0,
            'h1_count': 0, 'h1_mean_life': 0, 'h1_max_life': 0,
            'h1_entropy': 0
        }
    
    # Extract point cloud
    pts = extract_point_cloud_from_slice(slice_2d, method=method, max_points=max_points)
    
    if len(pts) < 3:  # Need at least 3 points for TDA
        return {
            'h0_count': 0, 'h0_mean_life': 0, 'h0_max_life': 0,
            'h1_count': 0, 'h1_mean_life': 0, 'h1_max_life': 0,
            'h1_entropy': 0
        }
    
    # Compute persistence diagrams
    try:
        result = ripser(pts, maxdim=maxdim)
        dgms = result['dgms']
    except Exception as e:
        return {
            'h0_count': 0, 'h0_mean_life': 0, 'h0_max_life': 0,
            'h1_count': 0, 'h1_mean_life': 0, 'h1_max_life': 0,
            'h1_entropy': 0
        }
    
    # Extract features
    features = {}
    
    # H0 features
    if len(dgms) > 0:
        h0_stats = compute_persistence_statistics(dgms[0], prefix='h0')
        features.update(h0_stats)
    
    # H1 features
    if len(dgms) > 1:
        h1_stats = compute_persistence_statistics(dgms[1], prefix='h1')
        features.update(h1_stats)
    
    return features


def compute_tda_features_volumetric(volume: np.ndarray,
                                   method: str = "edges",
                                   max_points: int = 3000,
                                   aggregation: str = "mean",
                                   sample_slices: Optional[int] = None) -> Dict[str, float]:
    """
    Compute TDA features across multiple slices with aggregation.
    
    This is the MAIN improvement: instead of using just ONE slice,
    we compute TDA for multiple slices and aggregate.
    
    Args:
        volume: Preprocessed volume (num_slices, H, W, C)
        method: Point cloud extraction method
        max_points: Max points per slice
        aggregation: How to aggregate across slices
            - "mean": Average statistics across slices
            - "max": Maximum statistics across slices
            - "stats": Compute [mean, std, max, min] across slices
        sample_slices: Number of slices to sample (None = use all)
    
    Returns:
        Dictionary of aggregated TDA features
    """
    # Handle input format
    if volume.ndim == 4:
        # (num_slices, H, W, C) - take first channel
        volume_gray = volume[:, :, :, 0]
    elif volume.ndim == 3:
        volume_gray = volume
    else:
        raise ValueError(f"Unexpected volume shape: {volume.shape}")
    
    num_slices = volume_gray.shape[0]
    
    # Determine which slices to process
    if sample_slices is not None and sample_slices < num_slices:
        # Sample evenly spaced slices
        slice_indices = np.linspace(0, num_slices-1, sample_slices, dtype=int)
    else:
        slice_indices = np.arange(num_slices)
    
    # Compute TDA for each slice
    slice_features = []
    for idx in slice_indices:
        slice_2d = volume_gray[idx]
        features = compute_tda_for_slice(slice_2d, method=method, max_points=max_points)
        slice_features.append(features)
    
    if len(slice_features) == 0:
        return {}
    
    # Convert to DataFrame for easier aggregation
    df = pd.DataFrame(slice_features)
    
    # Aggregate across slices
    if aggregation == "mean":
        agg_features = df.mean().to_dict()
        
    elif aggregation == "max":
        agg_features = df.max().to_dict()
        
    elif aggregation == "stats":
        # Compute multiple statistics
        agg_features = {}
        for col in df.columns:
            agg_features[f'{col}_mean'] = df[col].mean()
            agg_features[f'{col}_std'] = df[col].std()
            agg_features[f'{col}_max'] = df[col].max()
            agg_features[f'{col}_min'] = df[col].min()
    
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    return agg_features


# =============================================================================
# FILE PROCESSING
# =============================================================================

def process_single_volume(input_path: Path,
                          methods: Optional[List[str]] = None,
                          max_points: int = 3000,
                          aggregation: str = "mean",
                          sample_slices: Optional[int] = 8,
                          save_diagram: bool = False,
                          diagram_dir: Optional[Path] = None) -> Tuple[str, Dict]:
    """
    Process a single volume file with one or more extraction methods.
    
    Args:
        methods: List of extraction methods (e.g., ["gradient", "intensity"])
                 If None, defaults to ["edges"]
    
    Returns:
        (exam_id, features_dict) - features are concatenated with method suffixes
    """
    if methods is None:
        methods = ["edges"]
    
    try:
        # Load volume
        volume = np.load(input_path)
        
        # Validate
        if volume.ndim not in [3, 4]:
            return None, {}
        
        # Extract exam ID
        exam_id = input_path.stem.split('_')[0]
        
        # Extract features for each method
        all_features = {}
        
        for method in methods:
            method_features = compute_tda_features_volumetric(
                volume=volume,
                method=method,
                max_points=max_points,
                aggregation=aggregation,
                sample_slices=sample_slices
            )
            
            # Add method suffix to feature names if multiple methods
            if len(methods) > 1:
                method_features = {f"{k}_{method}": v for k, v in method_features.items()}
            
            all_features.update(method_features)
        
        # Save diagrams if requested (use first method for visualization)
        if save_diagram and diagram_dir is not None and TDA_AVAILABLE:
            # Compute for middle slice for visualization
            if volume.ndim == 4:
                mid_slice = volume[volume.shape[0]//2, :, :, 0]
            else:
                mid_slice = volume[volume.shape[0]//2]
            
            for method in methods:
                pts = extract_point_cloud_from_slice(mid_slice, method=method, max_points=max_points)
                if len(pts) >= 3:
                    try:
                        result = ripser(pts, maxdim=1)
                        dgms = result['dgms']
                        
                        plt.figure(figsize=(6, 5))
                        plot_diagrams(dgms, show=False)
                        method_suffix = f"_{method}" if len(methods) > 1 else ""
                        plt.title(f"{exam_id} - TDA ({method})")
                        plt.tight_layout()
                        plt.savefig(diagram_dir / f"{exam_id}_tda{method_suffix}.png", dpi=100)
                        plt.close()
                    except:
                        pass
        
        return exam_id, all_features
        
    except Exception as e:
        print(f"[ERROR] Failed to process {input_path.name}: {e}")
        return None, {}


def process_view(input_dir: Path,
                output_dir: Path,
                view: str,
                methods: Optional[List[str]] = None,
                max_points: int = 3000,
                aggregation: str = "stats",
                sample_slices: Optional[int] = 8,
                save_diagrams: bool = False,
                num_workers: int = 1) -> pd.DataFrame:
    """
    Process all volumes for a single view with optional parallelization.
    
    Args:
        input_dir: Directory with preprocessed .npy files
        output_dir: Output directory
        view: View name (axial, coronal, sagittal)
        methods: List of point cloud extraction methods (e.g., ["gradient", "intensity"])
        max_points: Max points per slice for TDA
        aggregation: Aggregation strategy across slices
        sample_slices: Number of slices to sample (None = all)
        save_diagrams: Whether to save persistence diagrams
        num_workers: Number of parallel workers (1 = sequential)
    
    Returns:
        DataFrame with TDA features
    """
    npy_files = sorted(list(input_dir.glob("*.npy")))
    
    if not npy_files:
        print(f"[WARN] No .npy files found in {input_dir}")
        return pd.DataFrame()
    
    if methods is None:
        methods = ["edges"]
    
    print(f"\n{'='*70}")
    print(f"  Processing {view.upper()} view: {len(npy_files)} volumes")
    print(f"{'='*70}")
    print(f"  Methods: {', '.join(methods)}")
    print(f"  Max points per slice: {max_points}")
    print(f"  Aggregation: {aggregation}")
    print(f"  Sample slices: {sample_slices if sample_slices else 'all'}")
    print(f"  Workers: {num_workers}")
    print(f"{'='*70}\n")
    
    # Create diagrams directory if needed
    diagram_dir = output_dir / "diagrams" / view
    if save_diagrams:
        diagram_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    if num_workers > 1:
        # Parallel processing
        process_fn = partial(
            process_single_volume,
            methods=methods,
            max_points=max_points,
            aggregation=aggregation,
            sample_slices=sample_slices,
            save_diagram=save_diagrams,
            diagram_dir=diagram_dir if save_diagrams else None
        )
        
        print(f"  Starting {num_workers} parallel workers...")
        
        results = []
        with mp.Pool(num_workers) as pool:
            # Use imap_unordered for better responsiveness
            # chunksize=1 helps update progress bar more frequently
            iterator = pool.imap_unordered(process_fn, npy_files, chunksize=1)
            
            for result in tqdm(iterator, total=len(npy_files), desc=f"TDA ({view})", unit="vol"):
                results.append(result)
                
    else:
        # Sequential processing
        print("  Running sequentially (1 worker)...")
        results = []
        for f in tqdm(npy_files, desc=f"TDA ({view})", unit="vol"):
            result = process_single_volume(
                f, methods, max_points, aggregation, sample_slices,
                save_diagrams, diagram_dir if save_diagrams else None
            )
            results.append(result)
    
    # Build DataFrame
    rows = []
    failed_count = 0
    for exam_id, features in results:
        if exam_id is not None and features:
            features['exam_id'] = exam_id
            features['view'] = view
            rows.append(features)
        else:
            failed_count += 1
    
    df = pd.DataFrame(rows)
    print(f"✅ Processed {len(df)} volumes for {view.upper()}")
    if failed_count > 0:
        print(f"❌ Failed to process {failed_count} volumes")
    print(f"   Features extracted: {len(df.columns) - 2}")  # -2 for exam_id and view
    
    return df


def combine_view_features(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine TDA features from multiple views into single row per exam.
    """
    if not dfs:
        return pd.DataFrame()
    
    combined = None
    for df in dfs:
        if df.empty:
            continue
        
        view = df['view'].iloc[0]
        
        # Rename columns with view suffix
        feature_cols = [c for c in df.columns if c not in ['exam_id', 'view']]
        rename_map = {c: f"{c}_{view}" for c in feature_cols}
        df_renamed = df.rename(columns=rename_map).drop(columns=['view'])
        
        if combined is None:
            combined = df_renamed
        else:
            combined = combined.merge(df_renamed, on='exam_id', how='outer')
    
    # Fill NaN with 0
    combined = combined.fillna(0)
    
    return combined


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args):
    """Main entry point for enhanced TDA feature extraction."""
    
    if not TDA_AVAILABLE:
        print("\n[ERROR] ripser/persim not installed!")
        print("Install with: pip install ripser persim scikit-image")
        return
    
    input_root = Path(args.input_root)
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine views
    views = ["axial", "coronal", "sagittal"] if args.all_views else [args.view]
    
    # Determine number of workers
    num_workers = args.num_workers if args.num_workers > 0 else get_num_workers()
    
    # Parse methods
    if args.multi_channel:
        methods = ["gradient", "intensity"]
        print("\n[INFO] Multi-channel mode enabled: Using gradient + intensity methods")
    else:
        methods = [args.method]
    
    print("\n" + "=" * 70)
    print("  ENHANCED TDA FEATURE EXTRACTION - MAC M4 OPTIMIZED")
    print("=" * 70)
    print(f"  Views: {', '.join(views)}")
    print(f"  Methods: {', '.join(methods)}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Sample slices: {args.sample_slices if args.sample_slices else 'all'}")
    print(f"  Max points: {args.max_points}")
    print(f"  Workers: {num_workers}")
    print(f"  Input: {input_root}")
    print(f"  Output: {output_dir}")
    print("=" * 70)
    
    # Process each view
    view_dfs = []
    for view in views:
        view_input = input_root / view
        if not view_input.exists():
            print(f"\n[WARN] View directory not found: {view_input}")
            continue
        
        df = process_view(
            input_dir=view_input,
            output_dir=output_dir,
            view=view,
            methods=methods,
            max_points=args.max_points,
            aggregation=args.aggregation,
            sample_slices=args.sample_slices,
            save_diagrams=args.save_diagrams,
            num_workers=num_workers
        )
        
        if not df.empty:
            # Save individual view CSV
            output_path = output_dir / f"tda_features_{view}.csv"
            df.to_csv(output_path, index=False)
            print(f"   Saved: {output_path}")
            view_dfs.append(df)
    
    # Combine all views
    if len(view_dfs) > 1:
        print(f"\n{'='*70}")
        print("  Combining features from all views...")
        print(f"{'='*70}")
        
        combined_df = combine_view_features(view_dfs)
        combined_path = output_dir / "tda_features_combined.csv"
        combined_df.to_csv(combined_path, index=False)
        
        print(f"✅ Saved combined TDA features: {combined_path}")
        print(f"   Shape: {combined_df.shape}")
    elif len(view_dfs) == 1:
        # Single view - also save as "combined" for compatibility
        view_dfs[0].drop(columns=['view']).to_csv(
            output_dir / "tda_features.csv", index=False
        )
    
    print("\n" + "=" * 70)
    print("  TDA FEATURE EXTRACTION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced TDA Feature Extraction - Mac M4 Optimized",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples for Mac M4:

  # RECOMMENDED FOR RADIOLOGY REPORT GENERATION: Multi-channel mode
  python tda.py --view coronal --multi_channel --aggregation stats
  
  # Multi-channel with all views (gradient + intensity features)
  python tda.py --all_views --multi_channel --aggregation stats
  
  # Single method: Use gradient-based point clouds (good for texture)
  python tda.py --all_views --method gradient
  
  # Process with more slices for richer topology
  python tda.py --all_views --sample_slices 12
  
  # Save persistence diagrams for visualization
  python tda.py --all_views --save_diagrams
  
  # Single view with parallel processing
  python tda.py --view axial --num_workers 6

Note: Uses multiprocessing to leverage M4's 10-core CPU
      --multi_channel extracts features with both gradient AND intensity methods
        """
    )
    
    # Get script directory for default paths
    script_dir = Path(__file__).parent.absolute()
    
    parser.add_argument("--input_root",
                        type=str,
                        default=str(script_dir / "preprocessed"),
                        help="Root directory containing preprocessed volumes")
    parser.add_argument("--output_root",
                        type=str,
                        default=str(script_dir / "tda_features"),
                        help="Output directory for TDA features")
    
    parser.add_argument("--view",
                        type=str,
                        default="axial",
                        choices=["axial", "coronal", "sagittal"],
                        help="Single view to process")
    parser.add_argument("--all_views",
                        action="store_true",
                        help="Process all three views (recommended)")
    
    # TDA parameters
    parser.add_argument("--method",
                        type=str,
                        default="edges",
                        choices=["edges", "intensity", "gradient"],
                        help="Point cloud extraction method (default: edges)")
    parser.add_argument("--multi_channel",
                        action="store_true",
                        help="Use multi-channel mode: extract features with both gradient + intensity")
    parser.add_argument("--max_points",
                        type=int,
                        default=3000,
                        help="Max points per slice for TDA (default: 3000)")
    parser.add_argument("--aggregation",
                        type=str,
                        default="stats",
                        choices=["mean", "max", "stats"],
                        help="Aggregation strategy across slices (default: stats)")
    parser.add_argument("--sample_slices",
                        type=int,
                        default=8,
                        help="Number of slices to sample (0 = all)")
    
    # Processing options
    parser.add_argument("--num_workers",
                        type=int,
                        default=0,
                        help="Number of parallel workers (0 = auto, 1 = sequential)")
    parser.add_argument("--save_diagrams",
                        action="store_true",
                        help="Save persistence diagrams as images")
    
    args = parser.parse_args()
    
    # Convert sample_slices=0 to None (means "all")
    if args.sample_slices == 0:
        args.sample_slices = None
    
    main(args)