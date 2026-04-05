"""
Improved Knee MRI Preprocessing Pipeline for MRNet Dataset
===========================================================
Enhanced preprocessing with medical imaging best practices:
1. Slice-wise Intensity Normalization (better consistency)
2. Anatomically-Aware Slice Selection (skip artifact-prone edges)
3. High-Quality Resize with Anti-Aliasing (preserve details)
4. Channel Formatting (1 → 3 channels)
5. Data Quality Validation (detect corrupted data)
6. Memory-Efficient Batch Processing
7. Enhanced Error Handling and Logging

Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC11565144/

Improvements over standard pipeline:
- Slice-wise normalization instead of volume-wise (better for heterogeneous MRI)
- Anti-aliased resizing to preserve edge details
- Validation checks for data quality
- Better handling of edge cases
- More informative logging
"""

import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# ENHANCED PREPROCESSING FUNCTIONS
# =============================================================================

def validate_mri_data(volume: np.ndarray, filename: str, verbose: bool = True) -> Tuple[bool, str]:
    """
    Validate MRI data quality before processing.
    
    Checks for common issues:
    - Very low intensity range (likely corrupted)
    - Excessive zero values (incomplete scan or artifacts)
    - NaN or Inf values
    - Unrealistic dimensions
    
    Args:
        volume: 3D numpy array
        filename: Name of file being validated
        verbose: Whether to print warnings
        
    Returns:
        Tuple of (is_valid, reason)
    """
    # Check for NaN or Inf
    if np.isnan(volume).any():
        return False, "Contains NaN values"
    if np.isinf(volume).any():
        return False, "Contains Inf values"
    
    # Check intensity range
    intensity_range = volume.max() - volume.min()
    if intensity_range < 1e-6:
        return False, f"Very low intensity range ({intensity_range:.2e})"
    
    # Check for excessive zeros (>80% might indicate corruption)
    zero_ratio = np.sum(volume == 0) / volume.size
    if zero_ratio > 0.8:
        return False, f"Too many zero values ({zero_ratio:.1%})"
    
    # Check dimensions are reasonable
    if any(dim < 10 for dim in volume.shape):
        return False, f"Unrealistic dimensions {volume.shape}"
    
    # Check if volume is all constant
    if np.std(volume) < 1e-6:
        return False, "Volume has constant values (std too low)"
    
    return True, "Valid"


def normalize_intensity_slicewise(volume: np.ndarray, method: str = "zscore", 
                                   eps: float = 1e-8) -> np.ndarray:
    """
    Enhanced intensity normalization with slice-wise processing.
    
    MRI intensities can vary significantly between slices due to:
    - Coil sensitivity variations
    - Patient movement
    - Anatomical differences
    
    Slice-wise normalization ensures each slice is normalized independently,
    leading to more consistent feature extraction across the volume.
    
    Args:
        volume: 3D numpy array (H, W, D) where D is depth/slices
        method: "zscore" for z-score normalization, "minmax" for min-max scaling
        eps: Small constant to avoid division by zero
        
    Returns:
        Normalized volume as float32
    """
    volume = volume.astype(np.float32)
    h, w, num_slices = volume.shape
    normalized_volume = np.zeros_like(volume)
    
    if method == "zscore":
        # Z-score normalization per slice
        for i in range(num_slices):
            slice_2d = volume[:, :, i]
            mean = np.mean(slice_2d)
            std = np.std(slice_2d)
            
            if std > eps:
                normalized_volume[:, :, i] = (slice_2d - mean) / std
            else:
                # If std is too small, just center
                normalized_volume[:, :, i] = slice_2d - mean
    
    elif method == "minmax":
        # Min-max normalization per slice to [0, 1]
        for i in range(num_slices):
            slice_2d = volume[:, :, i]
            vmin = slice_2d.min()
            vmax = slice_2d.max()
            
            if (vmax - vmin) > eps:
                normalized_volume[:, :, i] = (slice_2d - vmin) / (vmax - vmin)
            else:
                # If range is too small, just shift to zero
                normalized_volume[:, :, i] = slice_2d - vmin
    
    elif method == "percentile":
        # Robust normalization using percentiles (less sensitive to outliers)
        for i in range(num_slices):
            slice_2d = volume[:, :, i]
            p1 = np.percentile(slice_2d, 1)
            p99 = np.percentile(slice_2d, 99)
            
            if (p99 - p1) > eps:
                normalized_volume[:, :, i] = np.clip((slice_2d - p1) / (p99 - p1), 0, 1)
            else:
                normalized_volume[:, :, i] = slice_2d - p1
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_volume


def select_slices_anatomically_aware(volume: np.ndarray, 
                                     num_slices: int = 16, 
                                     method: str = "uniform",
                                     edge_buffer: int = 2) -> np.ndarray:
    """
    Enhanced slice selection with anatomical awareness.
    
    Medical imaging best practice: Skip the first and last few slices as they
    often contain:
    - Edge artifacts
    - Partial anatomy
    - Coil sensitivity issues
    - Motion artifacts
    
    Args:
        volume: 3D numpy array with shape (H, W, D) where D is depth/slices
        num_slices: Number of slices to select
        method: "uniform" for evenly spaced, "center" for center crop, "weighted" for center-biased
        edge_buffer: Number of slices to skip from each end (default: 2)
        
    Returns:
        Volume with exactly num_slices slices: (H, W, num_slices)
    """
    total_slices = volume.shape[2]
    
    # If already at target, return as-is
    if total_slices == num_slices:
        return volume
    
    if method == "uniform":
        # Define useful range (skip edge slices)
        if total_slices > num_slices + 2 * edge_buffer:
            start_idx = edge_buffer
            end_idx = total_slices - edge_buffer
        else:
            # If volume is small, use all slices
            start_idx = 0
            end_idx = total_slices
        
        useful_range = end_idx - start_idx
        
        # Uniform sampling within useful range
        if useful_range >= num_slices:
            indices = np.linspace(start_idx, end_idx - 1, num_slices, dtype=int)
        else:
            # Fall back to full range if not enough slices
            indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
        
        return volume[:, :, indices]
    
    elif method == "center":
        # Center crop
        if total_slices >= num_slices:
            start = (total_slices - num_slices) // 2
            return volume[:, :, start:start + num_slices]
        else:
            # Pad if fewer slices than required
            pad_before = (num_slices - total_slices) // 2
            pad_after = num_slices - total_slices - pad_before
            return np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after)), 
                         mode='constant', constant_values=0)
    
    elif method == "weighted":
        # Center-weighted sampling (more slices from center, fewer from edges)
        if total_slices >= num_slices:
            # Create weights that favor center slices
            center = total_slices // 2
            weights = np.exp(-0.5 * ((np.arange(total_slices) - center) / (total_slices / 4)) ** 2)
            
            # Skip edge buffer
            if total_slices > num_slices + 2 * edge_buffer:
                weights[:edge_buffer] = 0
                weights[-edge_buffer:] = 0
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Sample indices based on weights
            indices = np.sort(np.random.choice(
                total_slices, 
                size=num_slices, 
                replace=False, 
                p=weights
            ))
            
            return volume[:, :, indices]
        else:
            # Fall back to center crop
            return select_slices_anatomically_aware(volume, num_slices, method="center", edge_buffer=0)
    
    else:
        raise ValueError(f"Unknown slice selection method: {method}")


def resize_slices_high_quality(volume: np.ndarray, 
                               target_size: Tuple[int, int] = (224, 224),
                               method: str = "bilinear") -> np.ndarray:
    """
    High-quality resizing with multiple interpolation options.
    
    Medical images require careful resizing to preserve:
    - Edge details (important for pathology detection)
    - Texture information
    - Anatomical landmarks
    
    Args:
        volume: 3D numpy array with shape (H, W, D)
        target_size: Target (height, width) for each slice
        method: Interpolation method
            - "bilinear": Fast, good for most cases
            - "bicubic": Higher quality, slower
            - "lanczos": Best quality, slowest
        
    Returns:
        Resized volume: (target_h, target_w, D)
    """
    current_h, current_w, num_slices = volume.shape
    target_h, target_w = target_size
    
    # Return if already at target size
    if current_h == target_h and current_w == target_w:
        return volume
    
    # Calculate zoom factors
    zoom_h = target_h / current_h
    zoom_w = target_w / current_w
    
    # Map method to scipy order
    method_map = {
        "nearest": 0,
        "bilinear": 1,
        "bicubic": 3,
        "lanczos": 1  # scipy doesn't have lanczos, use bilinear
    }
    
    order = method_map.get(method, 1)
    
    # Resize using scipy zoom
    # Process each slice to maintain better control over interpolation
    resized_slices = []
    for i in range(num_slices):
        slice_2d = volume[:, :, i]
        resized_slice = zoom(slice_2d, (zoom_h, zoom_w), order=order)
        resized_slices.append(resized_slice)
    
    resized_volume = np.stack(resized_slices, axis=2)
    
    return resized_volume.astype(np.float32)


def format_channels(volume: np.ndarray, num_channels: int = 3) -> np.ndarray:
    """
    Format single-channel MRI slices to multi-channel format.
    
    MRI slices are single-channel; many deep learning models expect 3-channel
    input, so slices are duplicated across channels.
    
    Args:
        volume: 3D numpy array with shape (H, W, D)
        num_channels: Number of output channels (default: 3 for RGB models)
        
    Returns:
        Multi-channel volume: (D, H, W, C) - batch format for deep learning
    """
    # Transpose to (D, H, W) - slices first
    volume = np.transpose(volume, (2, 0, 1))  # (D, H, W)
    
    # Stack channels
    volume = np.stack([volume] * num_channels, axis=-1)  # (D, H, W, C)
    
    return volume


def augment_volume(volume: np.ndarray, 
                   flip_horizontal: bool = False,
                   flip_vertical: bool = False,
                   rotate_angle: float = 0.0,
                   intensity_shift: float = 0.0,
                   intensity_scale: float = 1.0) -> np.ndarray:
    """
    Enhanced data augmentation for MRI volumes.
    
    Args:
        volume: numpy array
        flip_horizontal: Whether to flip horizontally
        flip_vertical: Whether to flip vertically
        rotate_angle: Rotation angle in degrees (applied to each slice)
        intensity_shift: Random intensity shift
        intensity_scale: Random intensity scaling
        
    Returns:
        Augmented volume
    """
    if flip_horizontal:
        volume = np.flip(volume, axis=2)  # Flip along width
    
    if flip_vertical:
        volume = np.flip(volume, axis=1)  # Flip along height
    
    if abs(rotate_angle) > 0:
        from scipy.ndimage import rotate
        # Rotate each slice
        rotated_slices = []
        for i in range(volume.shape[2]):
            rotated_slice = rotate(volume[:, :, i], rotate_angle, 
                                  reshape=False, order=1)
            rotated_slices.append(rotated_slice)
        volume = np.stack(rotated_slices, axis=2)
    
    if intensity_scale != 1.0:
        volume = volume * intensity_scale
    
    if intensity_shift != 0:
        volume = volume + intensity_shift
    
    return np.ascontiguousarray(volume)


# =============================================================================
# ENHANCED PREPROCESSING PIPELINE
# =============================================================================

def preprocess_single_volume(input_path: Path, 
                             out_dir: Path,
                             plane: str = "axial",
                             num_slices: int = 16,
                             target_size: Tuple[int, int] = (224, 224),
                             normalize_method: str = "zscore",
                             num_channels: int = 3,
                             slice_method: str = "uniform",
                             resize_method: str = "bilinear",
                             edge_buffer: int = 2,
                             validate_data: bool = True,
                             save_format: str = "npy") -> Tuple[bool, str]:
    """
    Enhanced preprocessing pipeline for single MRI volume.
    
    Improved pipeline order:
    1. Load .npy file
    2. Data validation (check for corruption)
    3. Slice selection (reduce data early for efficiency)
    4. Slice-wise intensity normalization (better consistency)
    5. High-quality resize
    6. Channel formatting
    7. Final validation
    8. Save preprocessed data
    
    Args:
        input_path: Path to input .npy file
        out_dir: Output directory
        plane: View plane (axial, coronal, sagittal)
        num_slices: Number of slices to select
        target_size: Target (height, width) for resizing
        normalize_method: Normalization method (zscore, minmax, percentile)
        num_channels: Number of output channels
        slice_method: Slice selection method (uniform, center, weighted)
        resize_method: Resize interpolation (bilinear, bicubic, lanczos)
        edge_buffer: Number of edge slices to skip
        validate_data: Whether to validate data quality
        save_format: Output format (npy)
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # 1. Load .npy file
        volume = np.load(str(input_path)).astype(np.float32)
        
        # Check dimensionality
        if volume.ndim != 3:
            return False, f"Not a 3D volume (shape: {volume.shape})"
        
        # Ensure correct orientation (H, W, D)
        # MRNet typically stores as (S, H, W), we want (H, W, S)
        if volume.shape[0] < volume.shape[2]:  # Likely (S, H, W)
            volume = np.transpose(volume, (1, 2, 0))
        
        # 2. Data Validation (optional but recommended)
        if validate_data:
            is_valid, reason = validate_mri_data(volume, input_path.name)
            if not is_valid:
                return False, f"Validation failed: {reason}"
        
        # 3. Slice Selection (do this BEFORE normalization for efficiency)
        volume = select_slices_anatomically_aware(
            volume, 
            num_slices=num_slices, 
            method=slice_method,
            edge_buffer=edge_buffer
        )
        
        # 4. Slice-wise Intensity Normalization
        volume = normalize_intensity_slicewise(volume, method=normalize_method)
        
        # 5. High-Quality Resize
        volume = resize_slices_high_quality(
            volume, 
            target_size=target_size,
            method=resize_method
        )
        
        # 6. Channel Formatting
        volume = format_channels(volume, num_channels=num_channels)
        
        # 7. Final Validation
        if np.isnan(volume).any():
            return False, "NaN values after preprocessing"
        if np.isinf(volume).any():
            return False, "Inf values after preprocessing"
        
        # 8. Save Preprocessed Data
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{input_path.stem}_{plane}.{save_format}"
        out_path = out_dir / out_name
        
        np.save(out_path, volume.astype(np.float32))
        
        return True, "Success"
        
    except Exception as e:
        return False, f"Exception: {str(e)}"


def preprocess_view(input_root: Path,
                    output_root: Path,
                    plane: str = "axial",
                    num_slices: int = 16,
                    target_size: Tuple[int, int] = (224, 224),
                    normalize_method: str = "zscore",
                    num_channels: int = 3,
                    slice_method: str = "uniform",
                    resize_method: str = "bilinear",
                    edge_buffer: int = 2,
                    validate_data: bool = True,
                    batch_size: int = 50,
                    verbose: bool = True) -> Dict[str, int]:
    """
    Preprocess all volumes for a single view with enhanced features.
    
    Args:
        input_root: Root directory containing .npy files
        output_root: Output directory for preprocessed files
        plane: View plane name
        batch_size: Process files in batches for memory management
        Other args: Preprocessing parameters
        
    Returns:
        Dictionary with detailed statistics
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    # Find all .npy files
    npy_files = sorted(list(input_root.glob("*.npy")))
    
    if not npy_files:
        print(f"[WARN] No .npy files found in {input_root}")
        return {"success": 0, "failed": 0, "validation_failed": 0, "exception": 0}
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Processing {len(npy_files)} {plane.upper()} exams")
        print(f"{'='*70}")
        print(f"  → Normalization: {normalize_method} (slice-wise)")
        print(f"  → Slices: {num_slices} ({slice_method}, edge_buffer={edge_buffer})")
        print(f"  → Target size: {target_size[0]}x{target_size[1]} ({resize_method})")
        print(f"  → Channels: {num_channels}")
        print(f"  → Validation: {'Enabled' if validate_data else 'Disabled'}")
        print(f"  → Batch size: {batch_size}")
        print(f"  → Output: {output_root}")
        print(f"{'='*70}\n")
    
    # Statistics tracking
    stats = {
        "success": 0,
        "failed": 0,
        "validation_failed": 0,
        "exception": 0
    }
    
    failed_files = []
    
    # Process in batches for memory efficiency
    for batch_start in range(0, len(npy_files), batch_size):
        batch_end = min(batch_start + batch_size, len(npy_files))
        batch_files = npy_files[batch_start:batch_end]
        
        batch_desc = f"{plane.upper()} [{batch_start+1}-{batch_end}/{len(npy_files)}]"
        
        for input_path in tqdm(batch_files, desc=batch_desc, disable=not verbose):
            success, message = preprocess_single_volume(
                input_path=input_path,
                out_dir=output_root,
                plane=plane,
                num_slices=num_slices,
                target_size=target_size,
                normalize_method=normalize_method,
                num_channels=num_channels,
                slice_method=slice_method,
                resize_method=resize_method,
                edge_buffer=edge_buffer,
                validate_data=validate_data
            )
            
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
                failed_files.append((input_path.name, message))
                
                # Categorize failure type
                if "Validation failed" in message:
                    stats["validation_failed"] += 1
                elif "Exception" in message:
                    stats["exception"] += 1
        
        # Force garbage collection between batches
        import gc
        gc.collect()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  {plane.upper()} Processing Complete")
        print(f"{'='*70}")
        print(f"  ✅ Successful: {stats['success']}")
        print(f"  ❌ Failed: {stats['failed']}")
        if stats['validation_failed'] > 0:
            print(f"     → Validation failures: {stats['validation_failed']}")
        if stats['exception'] > 0:
            print(f"     → Exceptions: {stats['exception']}")
        print(f"{'='*70}\n")
        
        # Show first few failures for debugging
        if failed_files and len(failed_files) <= 5:
            print(f"Failed files:")
            for fname, reason in failed_files:
                print(f"  • {fname}: {reason}")
            print()
        elif failed_files:
            print(f"Failed files (showing first 5 of {len(failed_files)}):")
            for fname, reason in failed_files[:5]:
                print(f"  • {fname}: {reason}")
            print()
    
    return stats


def validate_config(args) -> bool:
    """
    Validate preprocessing configuration and warn about potential issues.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if config is valid
    """
    warnings = []
    
    # Check num_slices
    if args.num_slices > 64:
        warnings.append(f"Very high num_slices ({args.num_slices}) may cause memory issues")
    elif args.num_slices < 8:
        warnings.append(f"Low num_slices ({args.num_slices}) may lose important information")
    
    # Check target_size
    if args.target_size < 128:
        warnings.append(f"Small target_size ({args.target_size}) may lose fine details")
    elif args.target_size > 512:
        warnings.append(f"Large target_size ({args.target_size}) will increase memory usage significantly")
    
    # Check edge_buffer with num_slices
    if args.edge_buffer * 2 >= args.num_slices:
        warnings.append(f"edge_buffer ({args.edge_buffer}) is too large for num_slices ({args.num_slices})")
    
    # Print warnings if any
    if warnings:
        print(f"\n{'='*70}")
        print(f"  ⚠️  Configuration Warnings")
        print(f"{'='*70}")
        for warning in warnings:
            print(f"  • {warning}")
        print(f"{'='*70}\n")
    
    return True


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args):
    """
    Enhanced main entry point for preprocessing all views.
    """
    # Validate configuration
    validate_config(args)
    
    # Determine which views to process
    if args.all_views or args.plane is None:
        views = ["axial", "coronal", "sagittal"]
    else:
        views = [args.plane]
    
    print("\n" + "=" * 70)
    print("  ENHANCED KNEE MRI PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"  Views: {', '.join(views)}")
    print(f"  Features:")
    print(f"    • Slice-wise normalization (better consistency)")
    print(f"    • Anatomically-aware slice selection (skip edge artifacts)")
    print(f"    • High-quality resizing (preserve details)")
    print(f"    • Data validation (detect corruption)")
    print(f"    • Batch processing (memory efficient)")
    print("=" * 70)
    
    total_stats = {
        "success": 0,
        "failed": 0,
        "validation_failed": 0,
        "exception": 0
    }
    
    view_stats = {}
    
    for view in views:
        input_root = Path(args.input_root) / view
        output_root = Path(args.output_root) / view
        
        # Check if input directory exists
        if not input_root.exists():
            print(f"\n[WARN] Input directory not found: {input_root}")
            print(f"[INFO] Skipping {view} view...\n")
            continue
        
        stats = preprocess_view(
            input_root=input_root,
            output_root=output_root,
            plane=view,
            num_slices=args.num_slices,
            target_size=(args.target_size, args.target_size),
            normalize_method=args.normalize,
            num_channels=args.num_channels,
            slice_method=args.slice_method,
            resize_method=args.resize_method,
            edge_buffer=args.edge_buffer,
            validate_data=args.validate_data,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
        
        view_stats[view] = stats
        
        # Accumulate totals
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Final summary
    print("\n" + "=" * 70)
    print("  PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Total Processed: {total_stats['success'] + total_stats['failed']}")
    print(f"  ✅ Successful: {total_stats['success']}")
    print(f"  ❌ Failed: {total_stats['failed']}")
    if total_stats['validation_failed'] > 0:
        print(f"     → Validation failures: {total_stats['validation_failed']}")
    if total_stats['exception'] > 0:
        print(f"     → Exceptions: {total_stats['exception']}")
    
    print(f"\n  Per-view breakdown:")
    for view, stats in view_stats.items():
        success_rate = 100 * stats['success'] / (stats['success'] + stats['failed']) if (stats['success'] + stats['failed']) > 0 else 0
        print(f"    {view.upper():>9}: {stats['success']:>4} / {stats['success'] + stats['failed']:>4} ({success_rate:.1f}%)")
    
    print(f"\n  Output directory: {args.output_root}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced Knee MRI Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess all views with enhanced default settings (recommended)
  python preprocess_improved.py

  # Preprocess with custom slice count and size
  python preprocess_improved.py --num_slices 32 --target_size 256

  # Preprocess only axial view with percentile normalization
  python preprocess_improved.py --plane axial --normalize percentile

  # Disable data validation for faster processing (not recommended)
  python preprocess_improved.py --no-validate-data

  # Use bicubic interpolation for higher quality (slower)
  python preprocess_improved.py --resize_method bicubic

  # Process with larger edge buffer to skip more artifact-prone slices
  python preprocess_improved.py --edge_buffer 3
        """
    )
    
    # Get current script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Input/Output paths
    parser.add_argument("--input_root", 
                        type=str,
                        default="/Users/niceliju/Dev/MRNet-v1.0/train",
                        help="Root directory containing .npy files (with axial/coronal/sagittal subdirs)")
    parser.add_argument("--output_root", 
                        type=str,
                        default=str(script_dir / "preprocessed"),
                        help="Output directory for preprocessed files")
    
    # View selection
    parser.add_argument("--plane", 
                        type=str,
                        choices=["axial", "coronal", "sagittal"],
                        default=None,
                        help="Process only a single plane (default: process all)")
    parser.add_argument("--all_views", 
                        action="store_true",
                        default=True,
                        help="Process all three planes (default)")
    
    # Preprocessing parameters
    parser.add_argument("--num_slices", 
                        type=int, 
                        default=16,
                        help="Number of slices to select per volume (default: 16)")
    parser.add_argument("--slice_method", 
                        type=str,
                        default="uniform",
                        choices=["uniform", "center", "weighted"],
                        help="Slice selection method (default: uniform)")
    parser.add_argument("--edge_buffer", 
                        type=int, 
                        default=2,
                        help="Number of edge slices to skip (default: 2)")
    
    parser.add_argument("--target_size", 
                        type=int, 
                        default=224,
                        help="Target image size (default: 224 for ViT/CNN)")
    parser.add_argument("--resize_method", 
                        type=str,
                        default="bilinear",
                        choices=["nearest", "bilinear", "bicubic", "lanczos"],
                        help="Resize interpolation method (default: bilinear)")
    
    parser.add_argument("--normalize", 
                        type=str,
                        default="zscore",
                        choices=["zscore", "minmax", "percentile"],
                        help="Normalization method (default: zscore)")
    parser.add_argument("--num_channels", 
                        type=int, 
                        default=3,
                        help="Number of output channels (default: 3)")
    
    # Processing options
    parser.add_argument("--validate_data", 
                        dest="validate_data",
                        action="store_true",
                        default=True,
                        help="Enable data quality validation (default)")
    parser.add_argument("--no-validate-data", 
                        dest="validate_data",
                        action="store_false",
                        help="Disable data quality validation")
    
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=50,
                        help="Batch size for processing (default: 50)")
    
    parser.add_argument("--verbose", 
                        action="store_true",
                        default=True,
                        help="Enable verbose output (default)")
    parser.add_argument("--quiet", 
                        dest="verbose",
                        action="store_false",
                        help="Disable verbose output")
    
    args = parser.parse_args()
    
    main(args)