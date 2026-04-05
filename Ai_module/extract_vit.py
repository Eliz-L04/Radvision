"""
Enhanced ViT Feature Extraction for Knee MRI - Mac M4 Optimized
================================================================
Optimized for Apple Silicon (M4 chip) with:
1. Metal Performance Shaders (MPS) acceleration
2. Efficient memory management for unified memory
3. Optimized batch sizes for M4 GPU
4. Proper integration with enhanced preprocessing pipeline
5. Multiple aggregation strategies
6. Comprehensive logging and validation

Compatible with preprocess_improved.py output: (num_slices, H, W, C)

Hardware: Mac M4 Air (10-core CPU, 10-core GPU)
"""

import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel
from typing import List, Optional, Dict, Tuple
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# MAC M4 DEVICE CONFIGURATION
# =============================================================================

def get_optimal_device():
    """
    Get the optimal device for Mac M4.
    
    Priority:
    1. MPS (Metal Performance Shaders) for M4 GPU
    2. CPU as fallback
    
    Returns:
        Device string and device object
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using Metal Performance Shaders (MPS) - M4 GPU acceleration enabled")
        return "mps", device
    else:
        device = torch.device("cpu")
        print(f"⚠️  MPS not available, using CPU")
        return "cpu", device


def get_optimal_batch_size_for_m4(num_slices: int = 16) -> int:
    """
    Calculate optimal batch size for M4 Air with 16GB unified memory.
    
    M4 Air has unified memory shared between CPU and GPU.
    We need to be conservative to avoid memory pressure.
    
    Args:
        num_slices: Number of slices per volume
        
    Returns:
        Optimal batch size for slice processing
    """
    # M4 can handle larger batches than older Macs
    # But we want to be conservative with 16GB unified memory
    
    if num_slices <= 16:
        return 8  # Process half at a time
    elif num_slices <= 32:
        return 8  # Still safe
    else:
        return 4  # Conservative for larger volumes
    
    # Note: You can increase these if you have 24GB RAM version


# =============================================================================
# VALIDATION & UTILITY FUNCTIONS
# =============================================================================

def validate_preprocessed_volume(volume: np.ndarray, expected_shape: Tuple = None) -> bool:
    """
    Validate that preprocessed volume matches expected format.
    
    Args:
        volume: Loaded volume array
        expected_shape: Expected shape (num_slices, H, W, C) or None for flexible
        
    Returns:
        True if valid, False otherwise
    """
    # Check dimensions
    if volume.ndim != 4:
        print(f"[WARN] Expected 4D volume, got {volume.ndim}D with shape {volume.shape}")
        return False
    
    # Check shape format: (num_slices, H, W, C)
    num_slices, h, w, c = volume.shape
    
    if c not in [1, 3]:
        print(f"[WARN] Expected 1 or 3 channels, got {c}")
        return False
    
    if expected_shape:
        if volume.shape != expected_shape:
            print(f"[WARN] Shape mismatch: expected {expected_shape}, got {volume.shape}")
            return False
    
    # Check for invalid values
    if np.isnan(volume).any() or np.isinf(volume).any():
        print(f"[WARN] Volume contains NaN or Inf values")
        return False
    
    return True


def get_exam_id_from_filename(filename: str, plane: str) -> str:
    """
    Extract exam ID from preprocessed filename.
    
    Examples:
        "0000_axial.npy" → "0000"
        "1234_coronal.npy" → "1234"
    
    Args:
        filename: Name of the file
        plane: View plane (axial, coronal, sagittal)
        
    Returns:
        Exam ID string
    """
    # Remove extension
    name = Path(filename).stem
    
    # Remove plane suffix
    for p in ["axial", "coronal", "sagittal"]:
        name = name.replace(f"_{p}", "")
    
    return name


# =============================================================================
# NORMALIZATION & PREPROCESSING FOR VIT (MAC OPTIMIZED)
# =============================================================================

def prepare_slices_for_vit_m4(volume: np.ndarray, 
                               processor: ViTImageProcessor,
                               device: torch.device) -> torch.Tensor:
    """
    Prepare preprocessed volume slices for ViT input - Mac M4 optimized.
    
    The preprocessed volume is already normalized (z-score), but ViT expects
    ImageNet normalization. We handle this properly here.
    
    Mac M4 optimization:
    - Use numpy operations before GPU transfer (faster on unified memory)
    - Minimize CPU<->GPU transfers
    - Use contiguous memory layout
    
    Args:
        volume: Preprocessed volume (num_slices, H, W, C)
        processor: ViT image processor
        device: Target device (MPS or CPU)
        
    Returns:
        Tensor ready for ViT: (num_slices, 3, H, W)
    """
    num_slices, h, w, c = volume.shape
    
    # Ensure 3 channels (already done in preprocessing, but double-check)
    if c == 1:
        volume = np.repeat(volume, 3, axis=-1)
    elif c != 3:
        volume = volume[..., :3]
    
    # Convert to list of numpy arrays for processor
    # Processor expects values in [0, 255] range
    # Since our preprocessing uses z-score (mean~0, std~1), we need to rescale
    
    # Vectorized rescaling (faster on Mac M4's CPU cores)
    slice_list = []
    for i in range(num_slices):
        slice_data = volume[i].astype(np.float32)
        
        # Rescale from z-score to [0, 255]
        # Assuming z-score range is approximately [-3, 3] for most data
        slice_data = np.clip(slice_data, -3, 3)  # Clip outliers
        slice_data = (slice_data + 3) / 6  # Scale to [0, 1]
        slice_data = (slice_data * 255).astype(np.uint8)
        
        slice_list.append(slice_data)
    
    # Use ViT processor for proper normalization
    inputs = processor(images=slice_list, return_tensors="pt")
    
    # Transfer to device (MPS or CPU)
    pixel_values = inputs['pixel_values'].to(device)
    
    return pixel_values


# =============================================================================
# AGGREGATION STRATEGIES
# =============================================================================

def aggregate_slice_embeddings(embeddings: np.ndarray, 
                               method: str = "mean",
                               num_top_slices: int = None) -> np.ndarray:
    """
    Aggregate embeddings from multiple slices using various strategies.
    
    Args:
        embeddings: Slice embeddings (num_slices, embedding_dim)
        method: Aggregation method
            - "mean": Average pooling across slices (recommended)
            - "max": Max pooling across slices
            - "attention": Attention-weighted pooling (learns importance)
            - "top_k": Average top-k most informative slices
            - "concat": Concatenate all (increases dimensionality)
            - "stats": Concatenate [mean, max, std]
        num_top_slices: For "top_k" method, how many top slices to use
        
    Returns:
        Aggregated embedding
    """
    if method == "mean":
        return np.mean(embeddings, axis=0)
    
    elif method == "max":
        return np.max(embeddings, axis=0)
    
    elif method == "concat":
        # Warning: This increases dimensionality significantly
        return embeddings.flatten()
    
    elif method == "stats":
        # Concatenate statistical features: [mean, max, std]
        mean_emb = np.mean(embeddings, axis=0)
        max_emb = np.max(embeddings, axis=0)
        std_emb = np.std(embeddings, axis=0)
        return np.concatenate([mean_emb, max_emb, std_emb])
    
    elif method == "top_k":
        # Select top-k slices based on L2 norm (proxy for informativeness)
        if num_top_slices is None:
            num_top_slices = max(1, embeddings.shape[0] // 2)
        
        # Compute L2 norm for each slice
        norms = np.linalg.norm(embeddings, axis=1)
        
        # Get indices of top-k slices
        top_indices = np.argsort(norms)[-num_top_slices:]
        
        # Average embeddings from top slices
        return np.mean(embeddings[top_indices], axis=0)
    
    elif method == "attention":
        # Simple attention: weight slices by their magnitude
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        weights = norms / (np.sum(norms) + 1e-8)
        return np.sum(embeddings * weights, axis=0)
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# =============================================================================
# CORE EXTRACTION FUNCTIONS (MAC M4 OPTIMIZED)
# =============================================================================

def extract_volume_embedding_m4(volume: np.ndarray,
                                processor: ViTImageProcessor,
                                model: ViTModel,
                                device: torch.device,
                                aggregation: str = "mean",
                                batch_size: int = 8) -> np.ndarray:
    """
    Extract embedding for a single volume - Mac M4 optimized.
    
    Mac M4 optimizations:
    - Use optimal batch size for unified memory
    - Minimize GPU memory fragmentation
    - Efficient tensor transfers
    
    Args:
        volume: Preprocessed volume (num_slices, H, W, C)
        processor: ViT image processor
        model: ViT model
        device: Device object (MPS or CPU)
        aggregation: Aggregation method
        batch_size: Batch size for processing slices
        
    Returns:
        Volume embedding (embedding_dim,) or larger for concat methods
    """
    num_slices = volume.shape[0]
    all_embeddings = []
    
    # Process slices in batches for efficiency
    for i in range(0, num_slices, batch_size):
        batch_volume = volume[i:i + batch_size]
        
        # Prepare batch for ViT
        pixel_values = prepare_slices_for_vit_m4(batch_volume, processor, device)
        
        # Extract features
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            # Use CLS token (first token) as slice embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        all_embeddings.append(cls_embeddings)
        
        # Clear GPU cache on MPS to avoid memory buildup
        if str(device) == "mps":
            torch.mps.empty_cache()
    
    # Concatenate all batch embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)  # (num_slices, 768)
    
    # Aggregate across slices
    volume_embedding = aggregate_slice_embeddings(all_embeddings, method=aggregation)
    
    return volume_embedding


def extract_embeddings_for_view(input_root: Path,
                                output_root: Path,
                                plane: str = "axial",
                                model_name: str = "google/vit-base-patch16-224",
                                aggregation: str = "mean",
                                batch_size: int = None,
                                validate_inputs: bool = True) -> Dict[str, any]:
    """
    Extract ViT embeddings for all volumes in a single view - Mac M4 optimized.
    
    Args:
        input_root: Root directory containing preprocessed .npy files
        output_root: Output directory for embeddings
        plane: View plane (axial, coronal, sagittal)
        model_name: HuggingFace model name
        aggregation: Aggregation method
        batch_size: Batch size for slice processing (auto if None)
        validate_inputs: Whether to validate input volumes
        
    Returns:
        Dictionary with processing statistics
    """
    # Setup device (MPS for M4)
    device_name, device = get_optimal_device()
    
    print(f"\n{'='*70}")
    print(f"  ViT FEATURE EXTRACTION - {plane.upper()} (Mac M4 Optimized)")
    print(f"{'='*70}")
    print(f"  Model: {model_name}")
    print(f"  Device: {device_name.upper()}")
    print(f"  Hardware: Mac M4 Air (10-core CPU, 10-core GPU)")
    print(f"  Aggregation: {aggregation}")
    
    # Load model
    print(f"\n  Loading ViT model...")
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(device).eval()
    
    # Get input files
    input_plane_dir = input_root / plane
    if not input_plane_dir.exists():
        print(f"[ERROR] Input directory not found: {input_plane_dir}")
        return {"success": 0, "failed": 0}
    
    npy_files = sorted(list(input_plane_dir.glob("*.npy")))
    
    if not npy_files:
        print(f"[ERROR] No .npy files found in {input_plane_dir}")
        return {"success": 0, "failed": 0}
    
    # Auto-detect optimal batch size if not provided
    if batch_size is None:
        # Sample first file to get num_slices
        sample_volume = np.load(npy_files[0])
        num_slices = sample_volume.shape[0]
        batch_size = get_optimal_batch_size_for_m4(num_slices)
        print(f"  Auto-detected optimal batch size: {batch_size}")
    
    print(f"  Batch size: {batch_size}")
    print(f"  Found {len(npy_files)} volumes to process")
    print(f"{'='*70}\n")
    
    # Setup output directory
    output_plane_dir = output_root / plane
    output_plane_dir.mkdir(parents=True, exist_ok=True)
    
    # Processing statistics
    stats = {
        "success": 0,
        "failed": 0,
        "validation_failed": 0
    }
    
    exam_ids = []
    embedding_dims = []
    failed_files = []
    
    # Process each volume
    for npy_path in tqdm(npy_files, desc=f"Extracting {plane.upper()}", 
                         bar_format='{l_bar}{bar:30}{r_bar}'):
        try:
            # Load preprocessed volume
            volume = np.load(npy_path)
            
            # Validate format
            if validate_inputs:
                if not validate_preprocessed_volume(volume):
                    stats["validation_failed"] += 1
                    stats["failed"] += 1
                    failed_files.append((npy_path.name, "Validation failed"))
                    continue
            
            # Extract exam ID
            exam_id = get_exam_id_from_filename(npy_path.name, plane)
            
            # Extract embedding with M4 optimizations
            embedding = extract_volume_embedding_m4(
                volume=volume,
                processor=processor,
                model=model,
                device=device,
                aggregation=aggregation,
                batch_size=batch_size
            )
            
            # Save embedding
            output_path = output_plane_dir / f"{exam_id}_{plane}_vit.npy"
            np.save(output_path, embedding)
            
            # Track metadata
            exam_ids.append(exam_id)
            embedding_dims.append(embedding.shape[0])
            stats["success"] += 1
            
        except Exception as e:
            stats["failed"] += 1
            failed_files.append((npy_path.name, str(e)))
            print(f"\n[ERROR] Failed to process {npy_path.name}: {e}")
    
    # Clear MPS cache at end
    if device_name == "mps":
        torch.mps.empty_cache()
    
    # Save metadata
    metadata = {
        "exam_id": exam_ids,
        "embedding_dim": embedding_dims,
        "plane": [plane] * len(exam_ids),
        "aggregation": [aggregation] * len(exam_ids),
        "device": [device_name] * len(exam_ids)
    }
    
    metadata_path = output_plane_dir / "embedding_metadata.csv"
    pd.DataFrame(metadata).to_csv(metadata_path, index=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  {plane.upper()} Processing Complete")
    print(f"{'='*70}")
    print(f"  ✅ Successful: {stats['success']}")
    print(f"  ❌ Failed: {stats['failed']}")
    if stats['validation_failed'] > 0:
        print(f"     → Validation failures: {stats['validation_failed']}")
    
    if embedding_dims:
        print(f"\n  Embedding dimensions: {embedding_dims[0]}")
    
    print(f"  Output directory: {output_plane_dir}")
    print(f"  Metadata saved: {metadata_path}")
    print(f"{'='*70}\n")
    
    # Show failed files if any
    if failed_files and len(failed_files) <= 5:
        print(f"Failed files:")
        for fname, reason in failed_files:
            print(f"  • {fname}: {reason}")
        print()
    
    return stats


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args):
    """Main entry point for ViT feature extraction - Mac M4 optimized."""
    
    # Print system info
    print("\n" + "=" * 70)
    print("  ENHANCED ViT EXTRACTION - MAC M4 OPTIMIZED")
    print("=" * 70)
    print(f"  Hardware: Mac M4 Air")
    print(f"  CPU: 10-core")
    print(f"  GPU: 10-core (Metal)")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print("=" * 70)
    
    # Determine which views to process
    if args.all_views:
        views = ["axial", "coronal", "sagittal"]
    else:
        views = [args.plane]
    
    print(f"\n  Views to process: {', '.join(views)}")
    print(f"  Model: {args.model_name}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Validation: {'Enabled' if args.validate_inputs else 'Disabled'}")
    
    total_stats = {
        "success": 0,
        "failed": 0,
        "validation_failed": 0
    }
    
    view_stats = {}
    
    for view in views:
        stats = extract_embeddings_for_view(
            input_root=Path(args.input_root),
            output_root=Path(args.output_root),
            plane=view,
            model_name=args.model_name,
            aggregation=args.aggregation,
            batch_size=args.batch_size,
            validate_inputs=args.validate_inputs
        )
        
        view_stats[view] = stats
        
        # Accumulate totals
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Final summary
    print("\n" + "=" * 70)
    print("  FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Total Processed: {total_stats['success'] + total_stats['failed']}")
    print(f"  ✅ Successful: {total_stats['success']}")
    print(f"  ❌ Failed: {total_stats['failed']}")
    
    print(f"\n  Per-view breakdown:")
    for view, stats in view_stats.items():
        success_rate = 100 * stats['success'] / (stats['success'] + stats['failed']) if (stats['success'] + stats['failed']) > 0 else 0
        print(f"    {view.upper():>9}: {stats['success']:>4} / {stats['success'] + stats['failed']:>4} ({success_rate:.1f}%)")
    
    print(f"\n  Output directory: {args.output_root}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced ViT Feature Extraction for Knee MRI - Mac M4 Optimized",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples for Mac M4 Air:
  
  # RECOMMENDED: Extract from all views with MPS acceleration
  python extract_vit_m4.py --all_views
  
  # Use attention aggregation for better features
  python extract_vit_m4.py --all_views --aggregation attention
  
  # Extract with statistical features (mean + max + std)
  python extract_vit_m4.py --all_views --aggregation stats
  
  # Custom batch size (if you have 24GB RAM version)
  python extract_vit_m4.py --all_views --batch_size 12
  
  # Single view processing
  python extract_vit_m4.py --plane axial --aggregation mean

Note: MPS (Metal Performance Shaders) provides ~5-10x speedup over CPU on M4
        """
    )
    
    # Input/Output paths (Mac-friendly defaults)
    parser.add_argument("--input_root",
                        type=str,
                        default="preprocessed",
                        help="Directory containing preprocessed .npy files")
    parser.add_argument("--output_root",
                        type=str,
                        default="embeddings_vit",
                        help="Output directory for embeddings")
    
    # View selection
    parser.add_argument("--plane",
                        type=str,
                        default="axial",
                        choices=["axial", "coronal", "sagittal"],
                        help="Single plane to process (default: axial)")
    parser.add_argument("--all_views",
                        action="store_true",
                        help="Process all three planes (recommended)")
    
    # Model configuration
    parser.add_argument("--model_name",
                        type=str,
                        default="google/vit-base-patch16-224",
                        help="HuggingFace ViT model name")
    parser.add_argument("--aggregation",
                        type=str,
                        default="mean",
                        choices=["mean", "max", "concat", "stats", "top_k", "attention"],
                        help="Slice aggregation method (default: mean)")
    
    # Processing options (Mac M4 optimized defaults)
    parser.add_argument("--batch_size",
                        type=int,
                        default=None,
                        help="Batch size for slice processing (auto-detected if not specified)")
    
    parser.add_argument("--validate_inputs",
                        action="store_true",
                        default=True,
                        help="Validate input volumes (default: True)")
    parser.add_argument("--no-validate-inputs",
                        dest="validate_inputs",
                        action="store_false",
                        help="Skip input validation")
    
    args = parser.parse_args()
    
    main(args)