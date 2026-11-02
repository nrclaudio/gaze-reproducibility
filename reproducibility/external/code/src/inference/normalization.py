"""Gene expression normalization for bulk → spatial domain adaptation.

The primary source of distribution shift between bulk and spatial transcriptomics
is the difference in per-gene expression distributions. This module provides
z-score alignment: spatial gene expression is standardized per-gene using spatial
statistics, then rescaled to match the bulk training distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset


@dataclass
class GeneStatistics:
    """Per-gene mean and standard deviation computed from bulk training data."""

    mean: torch.Tensor  # [gene_dim]
    std: torch.Tensor   # [gene_dim]

    def save(self, path: Union[str, Path]) -> None:
        torch.save({'mean': self.mean, 'std': self.std}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GeneStatistics":
        data = torch.load(path, map_location='cpu', weights_only=True)
        return cls(mean=data['mean'], std=data['std'])


class GeneExpressionNormalizer:
    """Z-score alignment of spatial gene expression to bulk training distribution.

    Given bulk statistics (mean_bulk, std_bulk) and a spatial gene matrix:
        1. Compute per-gene spatial mean/std from the slide.
        2. Standardize: z = (x - mean_spatial) / std_spatial
        3. Rescale:     x' = z * std_bulk + mean_bulk
        4. Preserve zeros (unmeasured genes stay zero).
    """

    def __init__(self, bulk_stats: GeneStatistics) -> None:
        self.bulk_stats = bulk_stats

    @classmethod
    def from_checkpoint(cls, model_dir: Union[str, Path]) -> "GeneExpressionNormalizer":
        """Load normalizer from a trained model directory containing gene_stats.pt."""
        stats_path = Path(model_dir) / 'gene_stats.pt'
        if not stats_path.exists():
            raise FileNotFoundError(
                f"gene_stats.pt not found in {model_dir}. "
                "Run compute_gene_statistics() after training or use "
                "scripts/utils/compute_gene_stats.py to generate it."
            )
        return cls(GeneStatistics.load(stats_path))

    def normalize(self, x_gene: torch.Tensor) -> torch.Tensor:
        """Normalize spatial gene expression to match bulk distribution.

        Args:
            x_gene: [num_spots, gene_dim] spatial gene expression matrix.

        Returns:
            Normalized gene expression with same shape, zeros preserved.
        """
        device = x_gene.device
        bulk_mean = self.bulk_stats.mean.to(device)
        bulk_std = self.bulk_stats.std.to(device)

        # Mask for non-zero entries (zeros = unmeasured)
        nonzero_mask = x_gene != 0

        # Compute spatial per-gene statistics from non-zero entries only
        # Use float64 for numerical stability in mean/std computation
        x_f64 = x_gene.double()
        counts = nonzero_mask.sum(dim=0).double().clamp(min=1)
        spatial_mean = (x_f64 * nonzero_mask).sum(dim=0) / counts

        diff_sq = ((x_f64 - spatial_mean.unsqueeze(0)) * nonzero_mask) ** 2
        spatial_std = (diff_sq.sum(dim=0) / counts.clamp(min=2)).sqrt().float()
        spatial_mean = spatial_mean.float()

        # Avoid division by zero
        spatial_std = spatial_std.clamp(min=1e-8)
        bulk_std_safe = bulk_std.clamp(min=1e-8)

        # Z-score alignment
        z = (x_gene - spatial_mean.unsqueeze(0)) / spatial_std.unsqueeze(0)
        x_aligned = z * bulk_std_safe.unsqueeze(0) + bulk_mean.unsqueeze(0)

        # Preserve zeros
        x_aligned = torch.where(nonzero_mask, x_aligned, x_gene)

        return x_aligned


def compute_gene_statistics(
    dataset: Dataset,
    save_path: Union[str, Path],
) -> GeneStatistics:
    """Compute per-gene mean/std from reaction node gene features in a bulk dataset.

    Iterates over all samples, collects gene expression vectors for reaction nodes,
    and computes running statistics. Saves the result to *save_path*.

    Args:
        dataset: A bulk metabolomics dataset (e.g. ``MetabolicNetworkDataset``).
        save_path: Where to write ``gene_stats.pt``.

    Returns:
        The computed ``GeneStatistics``.
    """
    sum_x = None
    sum_x2 = None
    count = 0

    for idx in range(len(dataset)):
        data = dataset[idx]
        # x_gene may be [num_nodes, gene_dim] or [1, num_nodes, gene_dim]
        x_gene = data.x_gene
        if x_gene.dim() == 3:
            x_gene = x_gene.squeeze(0)

        reaction_indices = data.reaction_indices
        if isinstance(reaction_indices, list):
            reaction_indices = torch.tensor(reaction_indices)

        # Extract gene features at reaction nodes
        gene_features = x_gene[reaction_indices]  # [num_reactions, gene_dim]

        # Only count non-zero entries (zeros = unmeasured)
        nonzero_mask = gene_features != 0
        gene_f = gene_features.double()

        if sum_x is None:
            gene_dim = gene_features.size(1)
            sum_x = torch.zeros(gene_dim, dtype=torch.float64)
            sum_x2 = torch.zeros(gene_dim, dtype=torch.float64)
            count = torch.zeros(gene_dim, dtype=torch.float64)

        sum_x += (gene_f * nonzero_mask).sum(dim=0)
        sum_x2 += ((gene_f ** 2) * nonzero_mask).sum(dim=0)
        count += nonzero_mask.sum(dim=0).double()

    count_safe = count.clamp(min=1)
    mean = (sum_x / count_safe).float()
    variance = (sum_x2 / count_safe - (sum_x / count_safe) ** 2).clamp(min=0)
    std = variance.sqrt().float()

    stats = GeneStatistics(mean=mean, std=std)
    stats.save(save_path)
    print(f"Gene statistics saved to {save_path} (gene_dim={mean.size(0)}, "
          f"mean range=[{mean.min():.4f}, {mean.max():.4f}], "
          f"std range=[{std.min():.4f}, {std.max():.4f}])")

    return stats
