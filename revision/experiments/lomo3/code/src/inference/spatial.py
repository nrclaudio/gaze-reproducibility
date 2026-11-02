"""Spatial transcriptomics inference via bulk-trained metabolic GNN.

Pipeline:
    1. Load slide from ``SpatialTranscriptomicsDataset``.
    2. (Optional) Normalize gene expression to match bulk distribution.
    3. Smooth inputs via sparse adjacency diffusion.
    4. Batch spots through the metabolic GNN using ``model.forward()``.
    5. Return ``[num_spots, num_metabolites]`` predictions + metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm

from .normalization import GeneExpressionNormalizer


@dataclass
class SpatialPrediction:
    """Container for spatial inference results."""

    predictions: torch.Tensor  # [num_spots, num_metabolites]
    coordinates: Optional[np.ndarray] = None  # [num_spots, 2]
    metabolite_names: Optional[list] = field(default_factory=list)
    detection_scores: Optional[torch.Tensor] = None  # [num_spots, num_metabolites]


class SpatialRefiner:
    """Spatial inference by smoothing inputs, running the bulk model, and
    regularizing outputs with spatial graph diffusion.

    This class is model-agnostic: it accepts any ``nn.Module`` whose
    ``forward(data, ...)`` follows the batched interface used by
    ``ConcatGNNWithSMILES`` and ``EnhancedConcatGNNWithSMILES``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cpu",
        normalizer: Optional[GeneExpressionNormalizer] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.normalizer = normalizer
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Input smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def smooth_inputs(
        x_gene: torch.Tensor,
        spatial_edges: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Smooth gene expression using spatial adjacency diffusion.

        Args:
            x_gene: ``[num_spots, gene_dim]`` gene expression matrix.
            spatial_edges: ``[2, num_edges]`` spatial edge index.
            alpha: mixing coefficient (0 = only self, 1 = only neighbours).

        Returns:
            Smoothed gene expression with the same shape.
        """
        num_spots = x_gene.size(0)
        device = x_gene.device
        spatial_edges = spatial_edges.to(device)

        values = torch.ones(spatial_edges.size(1), device=device)
        adj = torch.sparse_coo_tensor(
            spatial_edges, values, (num_spots, num_spots)
        ).coalesce()

        degree = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
        row_indices = adj.indices()[0]
        norm_values = adj.values() / degree[row_indices]

        adj_norm = torch.sparse_coo_tensor(adj.indices(), norm_values, adj.size())

        smoothed = torch.sparse.mm(adj_norm, x_gene)
        return (1 - alpha) * x_gene + alpha * smoothed

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def run_inference(
        self,
        dataset,
        slide_idx: int = 0,
        smooth_input: bool = True,
        input_alpha: float = 0.5,
        spot_batch_size: int = 32,
    ) -> SpatialPrediction:
        """Run full spatial inference pipeline.

        Args:
            dataset: A ``SpatialTranscriptomicsDataset`` instance.
            slide_idx: Index of the slide to process.
            smooth_input: Whether to smooth gene expression before inference.
            input_alpha: Smoothing alpha for input diffusion.
            spot_batch_size: Number of spots to process per forward pass.

        Returns:
            A ``SpatialPrediction`` containing the predictions and metadata.
        """
        data = dataset[slide_idx]
        data = data.to(self.device)

        x_gene = data.x_gene  # [num_spots, gene_dim]

        # 1. Domain adaptation: normalize gene expression
        if self.normalizer is not None:
            x_gene = self.normalizer.normalize(x_gene)

        # 2. Input smoothing
        if smooth_input:
            x_gene = self.smooth_inputs(
                x_gene.cpu(), data.edge_index_spatial.cpu(), alpha=input_alpha
            ).to(self.device)

        # 3. Run bulk model on batches of spots
        num_spots = data.num_spots
        num_metabolic_nodes = data.x_embedding.size(0)
        gene_dim = dataset.gene_dim

        # Precompute GPR mask for reaction nodes
        reaction_mask = self._build_reaction_mask(data, num_metabolic_nodes, gene_dim)

        all_preds: list[torch.Tensor] = []
        all_detection: list[torch.Tensor] = []

        with torch.no_grad():
            for i in tqdm(
                range(0, num_spots, spot_batch_size),
                desc="Spatial inference",
                unit="batch",
            ):
                end = min(i + spot_batch_size, num_spots)
                batch_sz = end - i

                batch_data = self._build_batch(
                    data,
                    x_gene[i:end],
                    batch_sz,
                    num_metabolic_nodes,
                    gene_dim,
                    reaction_mask,
                )

                preds, *_ = self.model(
                    batch_data,
                    return_embeddings=False,
                    compute_physics=False,
                )
                all_preds.append(preds.cpu())

                # Collect detection scores if the model provides them
                det = getattr(self.model, "last_detection_scores", None)
                if det is not None:
                    all_detection.append(det.cpu())

        predictions = torch.cat(all_preds, dim=0)  # [num_spots, num_nodes]
        detection_scores = (
            torch.cat(all_detection, dim=0) if all_detection else None
        )

        # Build result
        coords = data.xy.cpu().numpy() if hasattr(data, "xy") else None
        met_names = (
            list(dataset.metabolite_names)
            if hasattr(dataset, "metabolite_names")
            else []
        )

        return SpatialPrediction(
            predictions=predictions,
            coordinates=coords,
            metabolite_names=met_names,
            detection_scores=detection_scores,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reaction_mask(
        self,
        data,
        num_metabolic_nodes: int,
        gene_dim: int,
    ) -> Optional[torch.Tensor]:
        """Build a ``[num_reactions, gene_dim]`` boolean GPR mask."""
        if not hasattr(data, "reaction_gene_indices"):
            return None

        num_rxns = len(data.reaction_indices)
        mask = torch.zeros(num_rxns, gene_dim, dtype=torch.bool, device=self.device)
        indices = data.reaction_gene_indices.to(self.device)

        valid = indices >= 0
        rows = torch.arange(num_rxns, device=self.device).unsqueeze(1).expand_as(indices)
        mask[rows[valid], indices[valid]] = True
        return mask

    def _build_batch(
        self,
        template_data,
        batch_genes: torch.Tensor,
        batch_sz: int,
        num_nodes: int,
        gene_dim: int,
        reaction_mask: Optional[torch.Tensor],
    ) -> Data:
        """Construct a batched ``Data`` object for ``model.forward()``.

        ``batch_genes`` is ``[batch_sz, gene_dim]`` — the gene expression
        for a subset of spatial spots.  This method maps it onto the
        metabolic graph nodes (reaction nodes only) and expands shared
        features to match the batch dimension.
        """
        # x_gene: [B, num_nodes, gene_dim]
        x_gene = torch.zeros(
            batch_sz, num_nodes, gene_dim, device=self.device
        )
        rxn_idx = template_data.reaction_indices
        x_gene[:, rxn_idx, :] = batch_genes.unsqueeze(1)

        if reaction_mask is not None:
            x_gene[:, rxn_idx, :] *= reaction_mask.unsqueeze(0)

        # Shared features expanded to batch dimension
        x_emb = template_data.x_embedding.unsqueeze(0).expand(batch_sz, -1, -1)
        x_smi = template_data.x_smiles.unsqueeze(0).expand(batch_sz, -1, -1)

        batch_data = Data()
        batch_data.x_embedding = x_emb
        batch_data.x_gene = x_gene
        batch_data.x_smiles = x_smi
        batch_data.edge_index = template_data.edge_index
        batch_data.edge_attr = getattr(template_data, "edge_attr", None)
        batch_data.reaction_indices = template_data.reaction_indices
        batch_data.metabolite_indices = template_data.metabolite_indices
        batch_data.incidence_matrix = getattr(template_data, "incidence_matrix", None)
        batch_data.node_types = getattr(template_data, "node_types", None)
        batch_data.batch_size = batch_sz

        return batch_data
