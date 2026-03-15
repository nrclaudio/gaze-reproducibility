# Batch Size Scaling Guide

This guide explains how to use the new efficient batching implementation to scale beyond `batch_size=1`.

## Overview

The original model was constrained to `batch_size=1` due to inefficient manual batching loops in the graph convolutions. The new implementation provides:

1. **True PyG batching**: Uses PyG's native batching for graph operations
2. **Shared structure optimization**: Reuses the same metabolic network topology across samples  
3. **Gradient accumulation**: Enables very large effective batch sizes without OOM
4. **Memory optimizations**: Efficient tensor operations and memory management

## Quick Start

### Basic Usage

```bash
# Original (batch_size=1)
python train_concat_smiles.py

# Efficient batching (automatically enabled)
python train_concat_smiles.py --batch_size 16

# With gradient accumulation for very large batches
python train_concat_smiles.py --batch_size 8 --gradient_accumulation_steps 4
# Effective batch size = 8 * 4 = 32
```

### Recommended Batch Sizes

For ~700 training samples:

- **Small GPU (8GB)**: `--batch_size 8`
- **Medium GPU (16GB)**: `--batch_size 16` 
- **Large GPU (24GB+)**: `--batch_size 32`

### Command Line Arguments

```bash
--batch_size INT           # Batch size (default: 1, automatically enables batching when > 1)
--gradient_accumulation_steps INT  # Gradient accumulation steps (default: 1)
```

## Performance Benefits

### Expected Improvements

- **Training Speed**: 2-5x faster training with batch_size=16
- **GPU Utilization**: Better parallelization of graph operations
- **Memory Efficiency**: Reduced overhead per sample
- **Gradient Stability**: More stable gradients from averaging across samples

### Why Batching Helps

1. **Eliminates manual loops**: Original code processed each sample individually through graph convolutions
2. **Leverages PyG efficiency**: Uses optimized PyG batching with proper node indexing
3. **Better hardware utilization**: Saturates GPU cores with more parallel operations
4. **Reduced I/O overhead**: Fewer data transfers relative to computation

## Architecture Details

### Key Changes Made

1. **Batching utilities** (`src/data/batching.py`):
   - `collate_metabolic_data()`: PyG-compatible collate function
   - Efficient batching for shared graph structure

2. **Model modifications** (`src/models/concat_gnn_smiles.py`):
   - `_forward_batched()`: Efficient batched forward pass
   - Automatic detection of batched vs single inputs
   - Proper tensor reshaping for PyG convolutions

3. **Loss function updates** (`src/models/losses.py`):
   - Batched loss computation in `MaskedMetaboliteLoss`
   - Compatible with existing physics losses

4. **Training script** (`train_concat_smiles.py`):
   - Gradient accumulation support
   - Configurable batch sizes via CLI

### Shared Structure Optimization

Since all samples use the same metabolic network topology:

- **Shared**: `edge_index`, `edge_attr`, `incidence_matrix`, node types
- **Per-sample**: Gene expression (`x_gene`), metabolite targets (`y`), masks
- **Result**: Memory-efficient batching without graph concatenation overhead

## Testing and Validation

### Numerical Equivalence Test

```bash
python test_batching.py
```

This script verifies that batched processing produces identical results to single-sample processing.

### Batch Size Benchmarking

The test script also benchmarks different batch sizes to find optimal performance for your hardware.

## Troubleshooting

### Out of Memory (OOM) Errors

1. **Reduce batch size**: Start with `--batch_size 4` 
2. **Use gradient accumulation**: `--batch_size 4 --gradient_accumulation_steps 4`
3. **Check GPU memory**: `nvidia-smi` to monitor usage

### Slower than Expected

1. **Check batch size**: Very small batches (2-4) may have overhead
2. **Verify GPU usage**: `nvidia-smi` should show high utilization
3. **Check memory**: Use larger batch sizes if GPU memory allows

### Numerical Differences

1. **Run test script**: `python test_batching.py` to validate
2. **Check random seeds**: Ensure reproducible results
3. **Verify dropout**: Model should be in same mode (train/eval)

## Migration Path

### From batch_size=1 to Larger Batches

1. **Start small**: Try `--batch_size 4` first
2. **Verify results**: Compare metrics with original batch_size=1
3. **Scale up gradually**: Increase batch size until OOM, then back off
4. **Use gradient accumulation**: If you need larger effective batch sizes

### Backwards Compatibility

- Default setting (`batch_size=1`) preserves original behavior
- All existing functionality remains unchanged
- Batching is automatically enabled when `batch_size > 1`
- Spatial transcriptomics falls back to single processing when needed

## Advanced Usage

### Custom Batch Sizes per Phase

```bash
# Fine-tuning with smaller batches for stability
python train_concat_smiles.py --batch_size 4 --pretrained_model model.pt

# Training from scratch with larger batches for speed
python train_concat_smiles.py --batch_size 16
```

### Memory-Constrained Environments

```bash
# Large effective batch size with gradient accumulation
python train_concat_smiles.py --batch_size 2 --gradient_accumulation_steps 16
# Effective batch size = 32, but only 2 samples in GPU memory at once
```

## Future Improvements

### Potential Enhancements

1. **Spatial batching**: Efficient batching for spatial transcriptomics data
2. **Dynamic batching**: Automatically adjust batch size based on available memory
3. **Mixed precision**: FP16 training for larger batches
4. **Distributed training**: Multi-GPU support with larger batch sizes

### Contributing

When adding new features, ensure they support both single and batched inputs:

```python
# Always check for batched input
is_batched = hasattr(data, 'batch_size') and data.batch_size > 1

if is_batched:
    # Handle batched input: [batch_size, ...]
    return process_batched(data)
else:
    # Handle single input: [...]
    return process_single(data)
```

## Summary

The new batching implementation removes the `batch_size=1` constraint and provides significant performance improvements:

- ✅ **Drop-in replacement**: Works with existing training scripts  
- ✅ **Backwards compatible**: Default behavior unchanged
- ✅ **Numerically equivalent**: Identical results to original implementation
- ✅ **Scalable**: Supports batch sizes from 1 to 32+ depending on hardware
- ✅ **Memory efficient**: Gradient accumulation for very large effective batches

Start with `--batch_size 8` and scale up based on your hardware capabilities. Batching is automatically enabled when batch_size > 1.