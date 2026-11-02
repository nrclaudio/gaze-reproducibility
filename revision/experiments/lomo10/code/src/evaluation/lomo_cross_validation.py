"""Leave-One-Metabolite-Out (LOMO) cross-validation for robust generalization testing."""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import os
import json
from typing import Optional, Dict, List
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

from collections import defaultdict

from ..losses.physics import EnhancedPhysicsLoss
from ..data.batching import collate_metabolic_data


def _predict_with_knn(held_out_metabolite, data, model, k=5, temperature=1.0,
                      mode: str = 'smiles', neighbor_map: Optional[dict[int, set[int]]] = None):
    """
    Predict a held-out metabolite using neighbour-based heuristics.

    In graph-based modes the function walks the reaction graph hop by hop,
    gathering measured neighbours until it collects ``k`` candidates (or
    exhausts the component). This allows information to flow beyond the
    immediate 1-hop neighbourhood when the closer neighbours are unlabelled.

    Args:
        held_out_metabolite: Index of the metabolite to predict
        data: PyTorch Geometric data object containing the sample
        model: Trained model to get predictions for neighbor metabolites
        k: Number of nearest neighbors to consider
        temperature: Temperature for similarity weighting (lower = more focused on closest neighbors)
        mode: 'smiles', 'graph_measured', or 'graph_prediction'
        neighbor_map: Direct neighbour mapping (required for graph_* modes)
    
    Returns:
        float: Predicted concentration for the held-out metabolite
    
    Raises:
        ValueError: If no valid neighbors found or invalid inputs
    """
    if not hasattr(data, 'x_smiles'):
        raise ValueError("Data object must have x_smiles attribute for KNN prediction")
    
    if not hasattr(data, 'metabolite_indices'):
        raise ValueError("Data object must have metabolite_indices attribute")
    
    mode = mode or 'smiles'

    # Always obtain model predictions for optional fallback / graph-prediction mode
    with torch.no_grad():
        out, *_ = model(data, return_embeddings=False, compute_physics=False)

        if out.dim() == 2 and out.size(0) == 1:
            out = out.squeeze(0)
        elif out.dim() != 1:
            raise ValueError(f"Expected model output to be 1D or 2D with batch_size=1, got shape {out.shape}")

        max_metabolite_idx = max(data.metabolite_indices).item()
        if out.size(0) <= max_metabolite_idx:
            raise ValueError(f"Model output size {out.size(0)} too small for metabolite index {max_metabolite_idx}")
    
    # Normalise mask shape (can be [num_nodes] or [1, num_nodes] depending on DataLoader)
    mask = data.mask
    if isinstance(mask, torch.Tensor):
        if mask.dim() == 2:
            if mask.size(0) != 1:
                raise ValueError(f"Expected mask with batch dimension 1, got shape {mask.shape}")
            mask = mask.squeeze(0)
        elif mask.dim() != 1:
            raise ValueError(f"Expected mask to be 1D or 2D with batch dimension 1, got {mask.dim()}D")
    else:
        raise ValueError("Data mask must be a torch.Tensor")

    if mode == 'smiles':
        if not hasattr(data, 'x_smiles'):
            raise ValueError("Data object must have x_smiles attribute for SMILES KNN prediction")

        if data.x_smiles.dim() == 3:
            if data.x_smiles.size(0) != 1:
                raise ValueError(f"Expected batch_size=1, got {data.x_smiles.size(0)}")
            held_out_smiles = data.x_smiles[0, held_out_metabolite]
        elif data.x_smiles.dim() == 2:
            held_out_smiles = data.x_smiles[held_out_metabolite]
        else:
            raise ValueError(f"Expected x_smiles to be 2D or 3D, got {data.x_smiles.dim()}D")

        if held_out_smiles.dim() != 1:
            raise ValueError(f"held_out_smiles should be 1D, got {held_out_smiles.dim()}D")

        candidate_metabolites = []
        similarities = []

        for met_idx in data.metabolite_indices:
            met_idx = int(met_idx.item()) if hasattr(met_idx, 'item') else int(met_idx)

            if met_idx == held_out_metabolite or not mask[met_idx]:
                continue

            if data.x_smiles.dim() == 3:
                met_smiles = data.x_smiles[0, met_idx]
            else:
                met_smiles = data.x_smiles[met_idx]

            if met_smiles.dim() != 1:
                raise ValueError(f"met_smiles should be 1D, got {met_smiles.dim()}D")

            similarity = F.cosine_similarity(
                held_out_smiles.unsqueeze(0),
                met_smiles.unsqueeze(0),
                dim=1
            )
            similarity_val = similarity.item()

            if similarity_val > 0:
                candidate_metabolites.append(met_idx)
                similarities.append(similarity_val)

        if len(candidate_metabolites) == 0:
            raise ValueError(f"No valid neighbor metabolites found for metabolite {held_out_metabolite}")

        k_actual = min(k, len(candidate_metabolites))
        neighbor_data = list(zip(candidate_metabolites, similarities))
        neighbor_data.sort(key=lambda x: x[1], reverse=True)
        k_nearest = neighbor_data[:k_actual]

        neighbor_predictions = []
        neighbor_similarities = []

        for met_idx, similarity in k_nearest:
            prediction = out[met_idx].item()
            neighbor_predictions.append(prediction)
            neighbor_similarities.append(similarity)

        if len(neighbor_similarities) == 0:
            raise ValueError("No neighbor similarities available for weight computation")

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        device = next(model.parameters()).device
        similarity_tensor = torch.tensor(neighbor_similarities, dtype=torch.float32, device=device)
        weights = F.softmax(similarity_tensor / temperature, dim=0)

        return float(sum(p * w.item() for p, w in zip(neighbor_predictions, weights)))

    elif mode in {'graph_measured', 'graph_prediction'}:
        if neighbor_map is None:
            raise ValueError("Graph-based KNN requires neighbor_map")

        # Track candidates with their hop distances for proper k-nearest selection
        visited = {int(held_out_metabolite)}
        candidates_with_distance = []  # List of (value, hop_distance) tuples
        current_distance = 1
        frontier = sorted(int(n) for n in neighbor_map.get(int(held_out_metabolite), set()) if int(n) != int(held_out_metabolite))

        while frontier and (k is None or len(candidates_with_distance) < k):
            next_frontier = set()

            for met_idx in frontier:
                # Only mark as visited after we process it
                if met_idx in visited:
                    continue

                visited.add(met_idx)

                # Check if this node has a valid measurement
                if met_idx >= len(mask) or not mask[met_idx]:
                    # Still explore its neighbors even if it doesn't have a measurement
                    next_frontier.update(neighbor_map.get(met_idx, set()))
                    continue

                # Get the value for this metabolite
                if mode == 'graph_measured':
                    y = data.y
                    if y.dim() == 2 and y.size(0) == 1:
                        y = y.squeeze(0)
                    value = float(y[met_idx].cpu().item())
                else:
                    value = float(out[met_idx].item())

                candidates_with_distance.append((value, current_distance))

                # Add this node's neighbors to explore
                next_frontier.update(neighbor_map.get(met_idx, set()))

                # Early exit if we have enough candidates
                if k is not None and len(candidates_with_distance) >= k:
                    break

            if k is not None and len(candidates_with_distance) >= k:
                break

            # Move to next hop distance
            next_frontier -= visited
            frontier = sorted(int(n) for n in next_frontier)
            current_distance += 1

            if not frontier:
                break

        if len(candidates_with_distance) == 0:
            raise ValueError(f"No measured graph neighbors available for metabolite {held_out_metabolite}")

        # Select k-nearest based on hop distance (already ordered by distance due to BFS)
        k_actual = min(k, len(candidates_with_distance)) if k is not None else len(candidates_with_distance)
        selected_candidates = candidates_with_distance[:k_actual]

        # Apply distance-based weighting (inverse distance weighting)
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        # Calculate weights based on inverse hop distance
        distances = [1.0 / dist for _, dist in selected_candidates]
        device = next(model.parameters()).device
        distance_tensor = torch.tensor(distances, dtype=torch.float32, device=device)

        # Apply temperature scaling and softmax for normalized weights
        weights = F.softmax(distance_tensor / temperature, dim=0)

        # Weighted average of values
        weighted_sum = sum(value * w.item() for (value, _), w in zip(selected_candidates, weights))
        return float(weighted_sum)

    else:
        raise ValueError(f"Unknown KNN mode '{mode}'")




def run_lomo_cross_validation(model_class, model_kwargs, dataset, device, output_dir,
                              n_metabolites_to_test=10, min_samples_per_metabolite=20,
                              training_epochs=50, learning_rate=0.001, seed=42,
                              lambda_bal=0.0, lambda_corr=0.0, lambda_var=0.0, lambda_kl=0.0,
                              use_knn_prediction=False, knn_k=5, knn_temperature=1.0,
                              knn_mode: str = 'smiles',
                              pretrained_encoders: Optional[str] = None,
                              compute_neighbor_correlation: bool = False,
                              specific_fold_idx: Optional[int] = None,
                              enable_metabolite_dropout: bool = False,
                              batch_size: int = 1,
                              gradient_accumulation_steps: int = 1,
                              mask_target_smiles: float = 0.0):
    """
    Run Leave-One-Metabolite-Out cross-validation.
    
    For each selected metabolite:
    1. Train a model with that metabolite completely masked out
    2. Evaluate the model's ability to predict that metabolite
    
    Args:
        model_class: Model class to instantiate
        model_kwargs: Arguments for model initialization
        dataset: Full dataset
        device: torch device
        output_dir: Directory to save results
        n_metabolites_to_test: Number of metabolites to test (limited for computational efficiency)
        min_samples_per_metabolite: Minimum samples required for a metabolite to be tested
        training_epochs: Number of epochs to train each model
        learning_rate: Learning rate for training
        seed: Random seed
        lambda_bal: Mass balance loss weight
        lambda_corr: Gene-flux correlation loss weight 
        lambda_var: Flux variance loss weight
        lambda_kl: KL divergence loss weight
        use_knn_prediction: If True, use neighbor-based prediction instead of model prediction
        knn_k: Number of neighbors to consider (applies to all modes)
        knn_temperature: Temperature for similarity weighting in SMILES mode
        knn_mode: 'smiles' (default), 'graph_measured', or 'graph_prediction'
        pretrained_encoders: Optional path to encoder weights from contrastive pretraining
        compute_neighbor_correlation: Whether to log correlation statistics between held-out metabolites
            and their neighbours in the training split
        specific_fold_idx: (Optional) If set, only run this specific index (0 to n-1) from the selected list.
        enable_metabolite_dropout: (Optional) Apply metabolite dropout during training.
        batch_size: Batch size for training.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        mask_target_smiles: Probability of masking target SMILES during training (Blindfold).
    
    Returns:
        Dictionary with LOMO CV results
    """
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output directory
    lomo_dir = os.path.join(output_dir, 'lomo_cross_validation')
    os.makedirs(lomo_dir, exist_ok=True)
    
    # Identify eligible metabolites
    print("Identifying eligible metabolites for LOMO CV...")
    (
        metabolite_sample_counts,
        coverage_scores,
        direct_scores,
        far_scores,
    ) = _compute_metabolite_statistics(dataset)

    eligible_metabolites = [
        met_idx
        for met_idx, count in metabolite_sample_counts.items()
        if count >= min_samples_per_metabolite
    ]

    print(
        f"Found {len(eligible_metabolites)} metabolites with >= {min_samples_per_metabolite} samples"
    )

    # Select subset for testing (computational efficiency)
    if len(eligible_metabolites) > n_metabolites_to_test:
        selected_metabolites = _select_metabolites_by_coverage(
            eligible_metabolites, coverage_scores, n_metabolites_to_test
        )
    else:
        selected_metabolites = eligible_metabolites

    # Validate selected metabolites
    first_data = dataset[0]
    _, direct_neighbors_map, _ = _build_metabolite_neighbor_maps(first_data)
    if hasattr(first_data, 'metabolite_indices'):
        valid_metabolite_indices = set(first_data.metabolite_indices.cpu().numpy().tolist())
    else:
        valid_metabolite_indices = set([i for i, t in enumerate(first_data.node_types) if t == 'metabolite'])
    
    # Filter selected metabolites to ensure they are valid
    selected_metabolites = [m for m in selected_metabolites if m in valid_metabolite_indices]
    print(f"Testing {len(selected_metabolites)} metabolites in LOMO CV")
    
    # --- JOB ARRAY LOGIC ---
    if specific_fold_idx is not None:
        if specific_fold_idx == 999:
            print("🎯 RUNNING UNIVERSAL TRAINING (No permanent mask)")
            target_metabolites = [-1]
            start_fold_idx = 999
        elif specific_fold_idx < 0 or specific_fold_idx >= len(selected_metabolites):
            print(f"⚠️ Fold index {specific_fold_idx} is out of range (0-{len(selected_metabolites)-1}). Skipping.")
            return {'results': [], 'skipped': True}
        else:
            print(f"🎯 RUNNING SINGLE FOLD: Index {specific_fold_idx} / {len(selected_metabolites)}")
            target_metabolites = [selected_metabolites[specific_fold_idx]]
            start_fold_idx = specific_fold_idx
    else:
        target_metabolites = selected_metabolites
        start_fold_idx = 0

    # Store results
    lomo_results = []
    
    # Setup physics loss if needed - MATCH REGULAR TRAINING EXACTLY
    use_physics = any([lambda_bal > 0, lambda_corr > 0, lambda_var > 0, lambda_kl > 0])
    if use_physics:
        physics_criterion = EnhancedPhysicsLoss(
            λ_mse=1.0,  # Same as regular training
            λ_bal=lambda_bal,
            λ_corr=lambda_corr,
            λ_var=lambda_var,
            λ_kl=lambda_kl
        ).to(device)
        print(f"Using physics-informed loss: λ_bal={lambda_bal}, λ_corr={lambda_corr}, λ_var={lambda_var}, λ_kl={lambda_kl}")
    else:
        physics_criterion = None
        print("Using MSE-only loss")
    
    # Run LOMO for selected metabolite(s)
    for i, held_out_metabolite in enumerate(tqdm(target_metabolites, desc="LOMO Folds")):
        current_fold_idx = start_fold_idx + i
        
        fold_result = _run_single_lomo_fold(
            model_class, model_kwargs, dataset, held_out_metabolite,
            device, training_epochs, learning_rate, current_fold_idx,
            physics_criterion, use_physics,
            use_knn_prediction, knn_k, knn_temperature, knn_mode,
            pretrained_encoders,
            direct_neighbors_map,
            compute_neighbor_correlation=compute_neighbor_correlation,
            enable_metabolite_dropout=enable_metabolite_dropout,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            output_dir=lomo_dir,
            mask_target_smiles=mask_target_smiles # Pass down
        )
        
        # Add metabolite information
        fold_result['metabolite_idx'] = held_out_metabolite
        if held_out_metabolite == -1:
            fold_result['metabolite_id'] = "UNIVERSAL_MODEL"
            fold_result['total_samples'] = len(dataset)
            fold_result['coverage_score'] = 1.0
            fold_result['direct_coverage'] = 1.0
            fold_result['far_coverage'] = 1.0
        else:
            fold_result['metabolite_id'] = _get_metabolite_id(dataset, held_out_metabolite)
            fold_result['total_samples'] = metabolite_sample_counts[held_out_metabolite]
            fold_result['coverage_score'] = float(coverage_scores.get(held_out_metabolite, 0.0))
            fold_result['direct_coverage'] = float(direct_scores.get(held_out_metabolite, 0.0))
            fold_result['far_coverage'] = float(far_scores.get(held_out_metabolite, 0.0))
        
        lomo_results.append(fold_result)
        
        # Save individual result immediately (for job arrays)
        if specific_fold_idx is not None:
            fold_file = os.path.join(lomo_dir, f'fold_{current_fold_idx}_result.json')
            with open(fold_file, 'w') as f:
                json.dump(fold_result, f, indent=2)
            print(f"✅ Saved individual fold result to: {fold_file}")
        
        # Save intermediate results (for sequential runs)
        elif (current_fold_idx + 1) % 5 == 0:
            _save_intermediate_results(lomo_results, lomo_dir)
    
    # Analyze and visualize results (only if running sequentially)
    analysis = {}
    if specific_fold_idx is None:
        analysis = analyze_lomo_results(lomo_results, lomo_dir)
        save_lomo_results(lomo_results, analysis, lomo_dir)
    
    return {
        'results': lomo_results,
        'analysis': analysis
    }


def evaluate_model_on_metabolites(model, dataset, device, metabolite_indices=None, batch_size=32):
    """
    Evaluate a model on a set of metabolites (Zero-Shot / Synthetic LOMO).
    
    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        device: torch device
        metabolite_indices: List of metabolite indices to evaluate. If None, uses all.
        batch_size: Batch size for evaluation.
        
    Returns:
        List of result dictionaries
    """
    model.eval()
    results = []
    
    # Identify valid metabolites if not provided
    if metabolite_indices is None:
        (counts, _, _, _) = _compute_metabolite_statistics(dataset)
        # Filter for metabolites with at least 20 samples
        metabolite_indices = [m for m, c in counts.items() if c >= 20]
    
    print(f"Evaluating Zero-Shot generalization on {len(metabolite_indices)} metabolites...")
    
    n_samples = len(dataset)
    for met_idx in tqdm(metabolite_indices, desc="Zero-Shot Eval"):
        held_out_predictions = []
        held_out_targets = []
        
        # MCR cache management
        use_mcr = False
        if hasattr(model, 'config') and hasattr(model.config, 'use_mcr'):
            use_mcr = model.config.use_mcr
        elif hasattr(model, 'use_mcr'):
            use_mcr = model.use_mcr

        if use_mcr and hasattr(model, 'clear_neighbor_cache'):
            first_data = dataset[0]
            model.clear_neighbor_cache()
            model.rebuild_neighbor_cache(
                edge_index=first_data.edge_index,
                metabolite_indices=first_data.metabolite_indices,
                reaction_indices=first_data.reaction_indices,
                exclude_metabolites={met_idx}
            )
        
        # Optimized batch processing
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_metabolic_data,
            num_workers=0
        )

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                
                # Replace embedding with random noise for this metabolite (synthetic LOMO)
                if hasattr(batch, 'x_embedding'):
                    # batch.x_embedding is [B, N, D]
                    # We modify the slice corresponding to the held-out metabolite for all samples in batch
                    # Note: We need to clone the slice to avoid in-place modification issues if we were backpropping,
                    # but here we are in no_grad. However, creating new noise is safe.
                    noise = torch.randn_like(batch.x_embedding[:, met_idx, :]) * 0.01
                    batch.x_embedding[:, met_idx, :] = noise

                if not hasattr(batch, 'num_nodes'):
                    batch.num_nodes = batch.x_embedding.size(1)

                out, *_ = model(batch, return_embeddings=False, compute_physics=False)
                # out is [B, N]
                
                # Extract predictions for the specific metabolite
                batch_preds = out[:, met_idx]
                
                # Handle targets
                if batch.y.dim() > 2: # [B, N, 1]
                    batch_targets = batch.y.squeeze(-1)[:, met_idx]
                else: # [B, N]
                    batch_targets = batch.y[:, met_idx]
                
                # Handle masks
                batch_mask = batch.mask[:, met_idx].bool()
                
                # Filter valid samples
                valid_preds = batch_preds[batch_mask].cpu().numpy()
                valid_targets = batch_targets[batch_mask].cpu().numpy()
                
                if len(valid_preds) > 0:
                    held_out_predictions.extend(valid_preds)
                    held_out_targets.extend(valid_targets)

        # Calculate metrics
        if len(held_out_predictions) > 0:
            held_out_predictions = np.array(held_out_predictions)
            held_out_targets = np.array(held_out_targets)
            
            # Check for NaNs
            if np.isnan(held_out_predictions).any():
                print(f"WARNING: NaNs found in predictions for metabolite {met_idx} ({_get_metabolite_id(dataset, met_idx)})")
                print(f"  Sample indices with NaN pred: {np.where(np.isnan(held_out_predictions))[0]}")
                # Filter out NaNs for metric calculation to prevent crash
                valid_mask = ~np.isnan(held_out_predictions) & ~np.isnan(held_out_targets)
                held_out_predictions = held_out_predictions[valid_mask]
                held_out_targets = held_out_targets[valid_mask]
                
            if np.isnan(held_out_targets).any():
                 print(f"WARNING: NaNs found in targets for metabolite {met_idx} ({_get_metabolite_id(dataset, met_idx)})")
                 # This should theoretically not happen if mask logic is correct, but safety check
                 valid_mask = ~np.isnan(held_out_predictions) & ~np.isnan(held_out_targets)
                 held_out_predictions = held_out_predictions[valid_mask]
                 held_out_targets = held_out_targets[valid_mask]

            if len(held_out_predictions) > 0:
                mse = mean_squared_error(held_out_targets, held_out_predictions)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(held_out_predictions - held_out_targets))
                r2 = r2_score(held_out_targets, held_out_predictions) if np.var(held_out_targets) > 0 else np.nan
            else:
                mse = rmse = mae = r2 = np.nan
        else:
            mse = rmse = mae = r2 = np.nan
            
        results.append({
            'metabolite_idx': met_idx,
            'metabolite_id': _get_metabolite_id(dataset, met_idx),
            'total_samples': len(held_out_predictions),
            'n_test_samples': len(held_out_predictions),
            'mse': mse,
            'r2': r2, 'rmse': rmse, 'mae': mae
        })
        
    return results


def _build_metabolite_neighbor_maps(first_data):
    """Pre-compute direct and extended metabolite neighbours using graph structure."""

    if hasattr(first_data, 'metabolite_indices'):
        metabolite_indices = first_data.metabolite_indices.cpu().numpy()
    else:
        metabolite_indices = np.array(
            [i for i, t in enumerate(first_data.node_types) if t == 'metabolite']
        )

    metabolite_set = set(int(m) for m in metabolite_indices)

    if hasattr(first_data, 'reaction_indices'):
        reaction_indices = first_data.reaction_indices.cpu().numpy()
    else:
        reaction_indices = np.array(
            [i for i, t in enumerate(first_data.node_types) if t == 'reaction']
        )
    reaction_set = set(int(r) for r in reaction_indices)

    edge_index = first_data.edge_index.cpu().numpy()

    reaction_to_metabolites: dict[int, set[int]] = defaultdict(set)
    metabolite_to_reactions: dict[int, set[int]] = defaultdict(set)

    for src, dst in edge_index.T:
        src_i = int(src)
        dst_i = int(dst)
        if src_i in metabolite_set and dst_i in reaction_set:
            reaction_to_metabolites[dst_i].add(src_i)
            metabolite_to_reactions[src_i].add(dst_i)
        elif src_i in reaction_set and dst_i in metabolite_set:
            reaction_to_metabolites[src_i].add(dst_i)
            metabolite_to_reactions[dst_i].add(src_i)

    direct_neighbors: dict[int, set[int]] = {m: set() for m in metabolite_set}

    for metabolite, reactions in metabolite_to_reactions.items():
        for reaction in reactions:
            direct_neighbors[metabolite].update(reaction_to_metabolites.get(reaction, set()))
        direct_neighbors[metabolite].discard(metabolite)

    far_neighbors: dict[int, set[int]] = {m: set() for m in metabolite_set}

    for metabolite, neighbours in direct_neighbors.items():
        extended = set()
        for neighbour in neighbours:
            extended.update(direct_neighbors.get(neighbour, set()))
        extended.discard(metabolite)
        extended.difference_update(neighbours)
        far_neighbors[metabolite] = extended

    return metabolite_indices, direct_neighbors, far_neighbors


def _compute_metabolite_statistics(dataset):
    """Compute sample counts and neighbourhood label coverage for each metabolite."""

    first_data = dataset[0]
    metabolite_indices, neighbor_map, far_neighbor_map = _build_metabolite_neighbor_maps(first_data)

    metabolite_counts: dict[int, int] = {int(m): 0 for m in metabolite_indices}

    coverage_accumulator: dict[int, dict[str, float]] = {
        int(m): {
            'direct_possible': 0.0,
            'direct_labelled': 0.0,
            'far_possible': 0.0,
            'far_labelled': 0.0,
        }
        for m in metabolite_indices
    }

    for idx in range(len(dataset)):
        data = dataset[idx]
        mask = data.mask
        if mask.dim() == 2:
            mask = mask.squeeze(0)
        mask = mask.cpu().numpy()

        for metabolite in metabolite_indices:
            m_idx = int(metabolite)
            if m_idx < len(mask) and mask[m_idx]:
                metabolite_counts[m_idx] += 1

            direct_neighbours = neighbor_map.get(m_idx, set())
            if direct_neighbours:
                coverage_accumulator[m_idx]['direct_possible'] += len(direct_neighbours)
                coverage_accumulator[m_idx]['direct_labelled'] += sum(
                    1 for n in direct_neighbours if n < len(mask) and mask[n]
                )

            far_neighbours = far_neighbor_map.get(m_idx, set())
            if far_neighbours:
                coverage_accumulator[m_idx]['far_possible'] += len(far_neighbours)
                coverage_accumulator[m_idx]['far_labelled'] += sum(
                    1 for n in far_neighbours if n < len(mask) and mask[n]
                )

    coverage_scores: dict[int, float] = {}
    direct_scores: dict[int, float] = {}
    far_scores: dict[int, float] = {}

    for metabolite, stats in coverage_accumulator.items():
        direct_possible = stats['direct_possible']
        far_possible = stats['far_possible']

        direct_ratio = stats['direct_labelled'] / direct_possible if direct_possible > 0 else 0.0
        far_ratio = stats['far_labelled'] / far_possible if far_possible > 0 else 0.0

        direct_scores[metabolite] = direct_ratio
        far_scores[metabolite] = far_ratio
        coverage_scores[metabolite] = 0.5 * (direct_ratio + far_ratio)

    if metabolite_counts:
        counts = list(metabolite_counts.values())
        print(
            "Sample counts per metabolite: "
            f"min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}"
        )

    if direct_scores:
        print(
            "Neighbour coverage stats: "
            f"mean direct={np.mean(list(direct_scores.values())):.3f}, "
            f"mean far={np.mean(list(far_scores.values())):.3f}"
        )
    else:
        print("Neighbour coverage stats: unavailable (no metabolite neighbours found)")

    return metabolite_counts, coverage_scores, direct_scores, far_scores


def _compute_neighbor_correlation_stats(
    dataset,
    sample_indices: List[int],
    target_metabolite: int,
    neighbor_map: Dict[int, set[int]],
) -> Dict[str, object]:
    """Calculate correlation metrics between a metabolite and its neighbours on selected samples."""
    neighbor_ids = sorted(int(n) for n in neighbor_map.get(int(target_metabolite), set()) if int(n) != int(target_metabolite))

    if not neighbor_ids:
        return {
            'target_metabolite': int(target_metabolite),
            'valid_neighbor_count': 0,
            'mean_corr': None,
            'median_corr': None,
            'max_corr': None,
            'top3_mean_corr': None,
            'corr_bins': _bucketize_correlations([]),
            'per_neighbor': {},
            'pair_counts': 0,
            'corr_band': 'no_neighbors'
        }

    per_neighbor: Dict[int, Dict[str, object]] = {}
    all_corrs: List[float] = []
    total_pairs = 0

    for neighbor in neighbor_ids:
        target_values: List[float] = []
        neighbor_values: List[float] = []

        for idx in sample_indices:
            data = dataset[idx]

            if not hasattr(data, 'y') or not hasattr(data, 'mask'):
                continue

            y = data.y
            mask = data.mask

            if y.dim() == 2 and y.size(0) == 1:
                y = y.squeeze(0)
            if mask.dim() == 2 and mask.size(0) == 1:
                mask = mask.squeeze(0)

            if target_metabolite >= y.size(0) or neighbor >= y.size(0):
                continue

            if mask[target_metabolite] and mask[neighbor]:
                target_values.append(float(y[target_metabolite].cpu().item()))
                neighbor_values.append(float(y[neighbor].cpu().item()))

        pair_count = len(target_values)
        total_pairs += pair_count

        if pair_count >= 2:
            corr_matrix = np.corrcoef(target_values, neighbor_values)
            corr_value = corr_matrix[0, 1]
            if np.isnan(corr_value):
                corr = None
            else:
                corr = float(np.clip(corr_value, -1.0, 1.0))
                all_corrs.append(corr)
        else:
            corr = None

        per_neighbor[neighbor] = {
            'correlation': corr,
            'n_pairs': pair_count,
        }

    summary_corrs = all_corrs.copy()

    mean_corr = float(np.mean(summary_corrs)) if summary_corrs else None
    median_corr = float(np.median(summary_corrs)) if summary_corrs else None
    max_corr = float(np.max(summary_corrs)) if summary_corrs else None

    if summary_corrs:
        top_k = min(3, len(summary_corrs))
        top3_mean_corr = float(np.mean(sorted(summary_corrs, reverse=True)[:top_k]))
    else:
        top3_mean_corr = None

    corr_bins = _bucketize_correlations(summary_corrs)
    corr_band = _categorise_mean_correlation(mean_corr, valid_count=len(summary_corrs))

    return {
        'target_metabolite': int(target_metabolite),
        'valid_neighbor_count': len(summary_corrs),
        'mean_corr': mean_corr,
        'median_corr': median_corr,
        'max_corr': max_corr,
        'top3_mean_corr': top3_mean_corr,
        'corr_bins': corr_bins,
        'corr_band': corr_band,
        'pair_counts': total_pairs,
        'per_neighbor': {
            str(idx): {
                'correlation': value['correlation'],
                'n_pairs': value['n_pairs'],
            }
            for idx, value in per_neighbor.items()
        },
        'correlations': summary_corrs,
    }


def _bucketize_correlations(correlations: List[float]) -> Dict[str, int]:
    """Count how many correlations fall into interpretable bins."""
    bins = {
        'negative': 0,
        '0-0.3': 0,
        '0.3-0.6': 0,
        '0.6-0.8': 0,
        '0.8-1.0': 0,
    }

    for corr in correlations:
        if corr < 0:
            bins['negative'] += 1
        elif corr < 0.3:
            bins['0-0.3'] += 1
        elif corr < 0.6:
            bins['0.3-0.6'] += 1
        elif corr < 0.8:
            bins['0.6-0.8'] += 1
        else:
            bins['0.8-1.0'] += 1

    return bins


def _categorise_mean_correlation(mean_corr: Optional[float], *, valid_count: int) -> str:
    """Convert a mean correlation into a qualitative band for reporting."""
    if mean_corr is None or valid_count == 0:
        return 'insufficient_pairs'
    if mean_corr >= 0.8:
        return 'very_high'
    if mean_corr >= 0.6:
        return 'high'
    if mean_corr >= 0.3:
        return 'moderate'
    if mean_corr >= 0:
        return 'low'
    return 'negative'


def _select_metabolites_by_coverage(eligible_metabolites, coverage_scores, n_select, n_quantiles=5):
    """Select metabolites across coverage quantiles for balanced evaluation."""

    if n_select >= len(eligible_metabolites):
        return eligible_metabolites

    sorted_metabolites = sorted(eligible_metabolites, key=lambda m: coverage_scores.get(m, 0.0))

    n_quantiles = min(n_quantiles, n_select, len(sorted_metabolites))
    quantile_bins = np.array_split(sorted_metabolites, n_quantiles)

    # Sort each quantile bin by coverage descending so we can pick deterministically
    bin_lists = [
        sorted((int(m) for m in bin_metabolites), key=lambda m: coverage_scores.get(m, 0.0), reverse=True)
        for bin_metabolites in quantile_bins if len(bin_metabolites) > 0
    ]

    selected = []
    pointers = [0] * len(bin_lists)

    # First pass: take the top-ranked metabolite from each quantile
    for idx, bin_list in enumerate(bin_lists):
        if len(selected) >= n_select:
            break
        if bin_list:
            selected.append(bin_list[0])
            pointers[idx] = 1

    # Subsequent passes: cycle through quantiles, descending within each bin
    while len(selected) < n_select:
        any_added = False
        for idx, bin_list in enumerate(bin_lists):
            if len(selected) >= n_select:
                break
            pointer = pointers[idx]
            if pointer < len(bin_list):
                selected.append(bin_list[pointer])
                pointers[idx] += 1
                any_added = True
        if not any_added:
            break  # All bins exhausted

    return selected[:n_select]


def _run_single_lomo_fold(model_class, model_kwargs, dataset, held_out_metabolite,
                          device, training_epochs, learning_rate, fold_idx,
                          physics_criterion, use_physics,
                          use_knn_prediction=False, knn_k=5, knn_temperature=1.0,
                          knn_mode: str = 'smiles',
                          pretrained_encoders: Optional[str] = None,
                          graph_neighbor_map: Optional[dict[int, set[int]]] = None,
                          compute_neighbor_correlation: bool = False,
                          enable_metabolite_dropout: bool = False,
                          batch_size: int = 1,
                          gradient_accumulation_steps: int = 1,
                          output_dir: Optional[str] = None,
                          mask_target_smiles: float = 0.0):
    """Run a single LOMO fold."""
    
    print(f"\n{'='*60}")
    print(f"LOMO FOLD {fold_idx + 1}: Held-out metabolite index {held_out_metabolite}")
    if enable_metabolite_dropout:
        print(f">>> SYNTHETIC LOMO ENABLED: Applying Metabolite Dropout during training")
    
    # Verify this is actually a metabolite node
    first_data = dataset[0]
    if held_out_metabolite != -1 and hasattr(first_data, 'node_types') and held_out_metabolite < len(first_data.node_types):
        node_type = first_data.node_types[held_out_metabolite]
        print(f"Node type at index {held_out_metabolite}: {node_type}")
        if node_type != 'metabolite':
            print(f"WARNING: Index {held_out_metabolite} is a {node_type} node, not a metabolite!")
    
    if physics_criterion is not None:
        print(f"Using physics-informed loss with criterion: {type(physics_criterion).__name__}")
    else:
        print(f"Using MSE-only loss")
    print(f"{'='*60}")
    
    # Verify held_out_metabolite is valid
    first_data = dataset[0]
    if hasattr(first_data, 'metabolite_indices'):
        metabolite_node_indices = first_data.metabolite_indices.cpu().numpy()
        if held_out_metabolite != -1 and held_out_metabolite not in metabolite_node_indices:
            raise ValueError(f"held_out_metabolite {held_out_metabolite} is not in metabolite_indices")
        if held_out_metabolite != -1:
            print(f"Verified: metabolite node {held_out_metabolite} is a valid metabolite")
        else:
            print(f"Verified: Universal training mode (held_out_metabolite = -1)")

    # Initialize model
    model = model_class(**model_kwargs).to(device)

    if pretrained_encoders:
        try:
            encoder_state = torch.load(pretrained_encoders, map_location=device)
            loaded_components = []

            if isinstance(encoder_state, dict):
                if 'reaction_encoder' in encoder_state and hasattr(model, 'reaction_encoder'):
                    model.reaction_encoder.load_state_dict(encoder_state['reaction_encoder'], strict=False)
                    loaded_components.append('reaction_encoder')

                if 'metabolite_encoder' in encoder_state and hasattr(model, 'metabolite_encoder'):
                    model.metabolite_encoder.load_state_dict(encoder_state['metabolite_encoder'], strict=False)
                    loaded_components.append('metabolite_encoder')

                if 'convs' in encoder_state and hasattr(model, 'convs'):
                    model.convs.load_state_dict(encoder_state['convs'], strict=False)
                    loaded_components.append('convs')

            if loaded_components:
                print(f"Loaded pretrained encoders ({', '.join(loaded_components)}) from {pretrained_encoders}")
            else:
                print(f"Warning: Pretrained encoders at {pretrained_encoders} did not match model components")
        except FileNotFoundError:
            print(f"Warning: Pretrained encoders path not found: {pretrained_encoders}. Proceeding without warm start.")
        except Exception as exc:
            print(f"Warning: Failed to load pretrained encoders from {pretrained_encoders}: {exc}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Clear and rebuild neighbor cache for MCR if applicable
    # Check both Enhanced model style (model.config.use_mcr) and baseline style (model.use_mcr)
    use_mcr = False
    if hasattr(model, 'config') and hasattr(model.config, 'use_mcr'):
        use_mcr = model.config.use_mcr
    elif hasattr(model, 'use_mcr'):
        use_mcr = model.use_mcr

    if use_mcr and hasattr(model, 'clear_neighbor_cache'):
        print("MCR detected – rebuilding neighbor cache without exclusions")
        first_data = dataset[0]
        model.clear_neighbor_cache()
        model.rebuild_neighbor_cache(
            edge_index=first_data.edge_index,
            metabolite_indices=first_data.metabolite_indices,
            reaction_indices=first_data.reaction_indices,
            exclude_metabolites=None
        )
        print("Neighbor cache rebuilt for MCR")
    
    # Split data into train/val (70/30)
    n_samples = len(dataset)
    indices = list(range(n_samples))
    np.random.shuffle(indices)
    
    train_size = int(0.7 * n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create DataLoaders
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_metabolic_data,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_metabolic_data,
        num_workers=4,
        pin_memory=True,
    )

    correlation_stats = None
    if compute_neighbor_correlation and graph_neighbor_map is not None:
        correlation_stats = _compute_neighbor_correlation_stats(
            dataset,
            train_indices,
            held_out_metabolite,
            graph_neighbor_map,
        )
        mean_corr = correlation_stats.get('mean_corr')
        max_corr = correlation_stats.get('max_corr')
        if mean_corr is not None:
            mean_str = f"{mean_corr:.3f}"
            max_str = f"{max_corr:.3f}" if max_corr is not None else 'n/a'
            valid_neighbors = correlation_stats.get('valid_neighbor_count', 0)
            print(
                "Training neighbor correlation - "
                f"mean: {mean_str}, max: {max_str}, valid neighbors: {valid_neighbors}"
            )
        else:
            print("Training neighbor correlation: insufficient overlapping measurements to compute stats")

    # Training with held-out metabolite masked
    train_losses = []
    val_losses = []
    physics_losses = [] if use_physics else None
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Keep track of best model state for universal training
    best_model_state = None
    
    print(f"Training for {training_epochs} epochs with batch_size={batch_size}, accumulation_steps={gradient_accumulation_steps}...")
    
    # Initialize early stopping variables (match normal training)
    patience = 10
    patience_counter = 0
    
    for epoch in range(training_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_physics_losses = {'bal': 0, 'corr': 0, 'var': 0, 'kl': 0} if use_physics else None
        processed_batches = 0
        
        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_epochs} Training", 
                         unit="batch", leave=False)
        
        optimizer.zero_grad()
        for i, data in enumerate(train_pbar):
            data = data.to(device)
            
            # Mask out the held-out metabolite (if specified)
            if held_out_metabolite != -1:
                # data.mask is [batch_size, num_nodes]
                if held_out_metabolite < data.mask.size(1):
                    data.mask[:, held_out_metabolite] = False
                else:
                    if epoch == 0:
                        print(f"WARNING: held_out_metabolite {held_out_metabolite} >= mask length {data.mask.size(1)}. Skipping permanent masking.")
            
            if not hasattr(data, 'num_nodes'):
                data.num_nodes = data.x_embedding.size(1)
            
            # Forward pass
            from ..models.enhanced_concat_gnn_smiles import EnhancedConcatGNNWithSMILES
            fwd_kwargs = {}
            if isinstance(model, EnhancedConcatGNNWithSMILES):
                if enable_metabolite_dropout:
                    fwd_kwargs['dropout_metabolites'] = True
                if mask_target_smiles > 0:
                    fwd_kwargs['mask_target_smiles'] = mask_target_smiles
                
            out, delta, flux_vec, _ = model(data, return_embeddings=False, compute_physics=use_physics, **fwd_kwargs)
            
            # Compute loss only on non-held-out metabolites
            if data.mask.any():
                if use_physics and physics_criterion is not None:
                    total_loss = physics_criterion(
                        c_hat=out,
                        c_true=data.y,
                        mask=data.mask,
                        delta=delta,
                        flux_vec=flux_vec,
                        epoch=epoch,
                        data=data
                    )
                    
                    # Extract physics loss components for tracking
                    if hasattr(physics_criterion, 'last_loss_components'):
                        components = physics_criterion.last_loss_components
                        epoch_physics_losses['bal'] += components.get('Balance', 0)
                        epoch_physics_losses['corr'] += components.get('Corr', 0)
                        epoch_physics_losses['var'] += components.get('Variance', 0)
                        epoch_physics_losses['kl'] += components.get('FluxAlignment', 0)
                        mse_loss_value = components.get('MSE', 0)
                    else:
                        mse_loss_value = total_loss.item()
                else:
                    # Use MSE-only loss
                    total_loss = torch.nn.functional.mse_loss(
                        out[data.mask], 
                        data.y[data.mask]
                    )
                    mse_loss_value = total_loss.item()
                
                # Gradient Accumulation
                scaled_loss = total_loss / gradient_accumulation_steps
                scaled_loss.backward()
                
                if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_train_loss += mse_loss_value
                processed_batches += 1
                
                # Update progress bar
                train_pbar.set_postfix({'Loss': f'{mse_loss_value:.4f}'})
        
        train_pbar.close()
        
        avg_train_loss = epoch_train_loss / max(1, processed_batches)
        train_losses.append(avg_train_loss)
        
        if use_physics:
            avg_physics_losses = {k: v / max(1, processed_batches) for k, v in epoch_physics_losses.items()}
            physics_losses.append(avg_physics_losses)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        val_predictions = []
        val_targets = []
        val_processed_batches = 0
        
        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{training_epochs} Validation",
                       unit="batch", leave=False)
        
        with torch.no_grad():
            for data in val_pbar:
                data = data.to(device)

                if not hasattr(data, 'num_nodes'):
                    data.num_nodes = data.x_embedding.size(1)

                out, *_ = model(data, return_embeddings=False, compute_physics=False)

                if data.mask.any():
                    loss = torch.nn.functional.mse_loss(
                        out[data.mask],
                        data.y[data.mask]
                    )
                    epoch_val_loss += loss.item()
                    val_processed_batches += 1

                    # Collect predictions for R² and RMSE
                    val_predictions.extend(out[data.mask].cpu().numpy())
                    val_targets.extend(data.y[data.mask].cpu().numpy())

                    # Update progress bar
                    val_pbar.set_postfix({'Val_Loss': f'{loss.item():.4f}'})
        
        val_pbar.close()
        
        if val_processed_batches > 0:
            avg_val_loss = epoch_val_loss / val_processed_batches
            val_losses.append(avg_val_loss)
            
            # Calculate R² and RMSE if we have predictions
            if len(val_predictions) > 0:
                val_r2 = r2_score(val_targets, val_predictions)
                val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
            else:
                val_r2 = 0.0
                val_rmse = 0.0
            
            # Track best validation loss - no early stopping logic here yet
        else:
            avg_val_loss = 0
            val_losses.append(0)
            val_r2 = 0.0
            val_rmse = 0.0
            
        # Update learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Print progress every epoch (matching normal training)
        progress_msg = f"Epoch {epoch+1:3d}/{training_epochs} - LR: {new_lr:.6f} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val R²: {val_r2:.4f}, Val RMSE: {val_rmse:.4f}"
        
        # Add loss component details
        if use_physics and physics_losses:
            # Physics-informed training: show all components
            p = physics_losses[-1]
            progress_msg += f" | MSE: {avg_train_loss:.3f}, Bal: {p['bal']:.4f}, Corr: {p['corr']:.3f}, Var: {p['var']:.3f}"
            if p['kl'] > 0:
                progress_msg += f", KL: {p['kl']:.3f}"
        
        print(progress_msg)
        
        # Early stopping (match normal training logic exactly)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # Capture best model state for universal training
            if held_out_metabolite == -1:
                import copy
                best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
            
    # Save the best model if this is universal training
    if held_out_metabolite == -1 and output_dir is not None:
        if best_model_state is None:
            # Fallback to current state if no improvement (unlikely)
            best_model_state = model.state_dict()
            
        model_save_path = os.path.join(output_dir, 'best_model.pt')
        torch.save(best_model_state, model_save_path)
        print(f"\n✅ SAVED UNIVERSAL MODEL to: {model_save_path}")
        
        # Load best state for evaluation
        model.load_state_dict(best_model_state)
    
    # Evaluate on held-out metabolite
    model.eval()
    held_out_predictions = []
    held_out_targets = []
    sample_indices = []
    
    # Debug: Track prediction method
    is_mcr = hasattr(model, 'config') and hasattr(model.config, 'use_mcr') and model.config.use_mcr
    if use_knn_prediction:
        print(f"\n--- KNN Prediction Info for Held-out Metabolite {held_out_metabolite} ---")
        mode_desc = knn_mode or 'smiles'
        print(f"    Mode={mode_desc}, K={knn_k}, temperature={knn_temperature}")
    elif is_mcr:
        print(f"\n--- MCR Debug Info for Held-out Metabolite {held_out_metabolite} ---")
    
    with torch.no_grad():
        for idx in range(n_samples):
            data = dataset[idx].to(device)

            # Handle mask dimension
            mask = data.mask
            if mask.dim() == 2:
                mask = mask.squeeze(0)

            # Check if this sample has the held-out metabolite
            if held_out_metabolite < len(mask) and mask[held_out_metabolite]:
                if not hasattr(data, 'num_nodes'):
                    data.num_nodes = data.x_embedding.size(0)
                
                # Choose prediction method: KNN vs direct model prediction
                if use_knn_prediction:
                    try:
                        pred = _predict_with_knn(
                            held_out_metabolite,
                            data,
                            model,
                            k=knn_k,
                            temperature=knn_temperature,
                            mode=knn_mode,
                            neighbor_map=graph_neighbor_map
                        )
                    except ValueError as e:
                        if len(held_out_predictions) == 0:  # Only print warning once
                            print(f"    KNN prediction failed for sample {idx}: {e}")
                            print(f"    Falling back to direct model prediction...")
                        # Fallback to model prediction
                        out, *_ = model(data, return_embeddings=False, compute_physics=False)
                        if out.dim() == 2 and out.size(0) == 1:
                            out = out.squeeze(0)
                        pred = out[held_out_metabolite].cpu().item()
                else:
                    # Direct model prediction (original method)
                    out, *_ = model(data, return_embeddings=False, compute_physics=False)
                    
                    # Squeeze batch dimension if batch_size=1
                    if out.dim() == 2 and out.size(0) == 1:
                        out = out.squeeze(0)
                    
                    pred = out[held_out_metabolite].cpu().item()
                
                # Get target value (consistent across both methods)
                if data.y.dim() == 2 and data.y.size(0) == 1:
                    data.y = data.y.squeeze(0)
                target = data.y[held_out_metabolite].cpu().item()
                
                # Debug logging for first few predictions
                if (use_knn_prediction or is_mcr) and len(held_out_predictions) < 5:
                    print(f"  Sample {idx}: pred={pred:.4f}, target={target:.4f}, diff={abs(pred-target):.4f}")
                    
                    if use_knn_prediction:
                        # KNN-specific debug info
                        print(f"    Prediction method: KNN ({knn_mode}) with K={knn_k}, temperature={knn_temperature}")
                        # Get model predictions for comparison (if not already failed)
                        try:
                            if 'out' not in locals():
                                out, *_ = model(data, return_embeddings=False, compute_physics=False)
                                if out.dim() == 2 and out.size(0) == 1:
                                    out = out.squeeze(0)
                            model_pred = out[held_out_metabolite].cpu().item()
                            print(f"    Model prediction (for comparison): {model_pred:.4f}")
                            print(f"    KNN vs Model difference: {abs(pred - model_pred):.4f}")
                        except:
                            pass
                    elif is_mcr:
                        # MCR-specific debug info
                        print(f"    Prediction method: MCR (attention-based)")
                        # Also check prediction statistics for all metabolites
                        if 'out' in locals():
                            all_preds = out[data.metabolite_indices].cpu().numpy()
                            print(f"    All metabolite preds - mean={np.mean(all_preds):.4f}, std={np.std(all_preds):.4f}, min={np.min(all_preds):.4f}, max={np.max(all_preds):.4f}")
                        
                        # Check if we can get attention weights for debugging
                        if hasattr(model, 'metabolite_reader') and len(held_out_predictions) == 0:
                            # Try to get attention weights for the held-out metabolite
                            try:
                                # Find the position of held-out metabolite in metabolite_indices
                                met_indices_list = data.metabolite_indices.cpu().numpy().tolist()
                                if held_out_metabolite in met_indices_list:
                                    held_out_pos = met_indices_list.index(held_out_metabolite)
                                    print(f"    Held-out metabolite is at position {held_out_pos} in metabolite list")
                                    # Note: Getting actual attention weights would require modifying forward pass
                                    # For now, just log that MCR is being used
                                    print(f"    MCR is processing this metabolite with attention mechanism")
                            except Exception as e:
                                print(f"    Could not analyze attention: {e}")
                
                held_out_predictions.append(pred)
                held_out_targets.append(target)
                sample_indices.append(idx)
    
    # Calculate metrics
    if len(held_out_predictions) > 0:
        held_out_predictions = np.array(held_out_predictions)
        held_out_targets = np.array(held_out_targets)
        
        mse = mean_squared_error(held_out_targets, held_out_predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(held_out_predictions - held_out_targets))
        
        if np.var(held_out_targets) > 0:
            r2 = r2_score(held_out_targets, held_out_predictions)
        else:
            r2 = np.nan
    else:
        mse = rmse = mae = r2 = np.nan
    
    # Print fold completion summary
    print(f"\n--- FOLD {fold_idx + 1} COMPLETED ---")
    print(f"Best epoch: {best_epoch} (Val loss: {best_val_loss:.4f})")
    print(f"Held-out metabolite {held_out_metabolite} results:")
    print(f"  • Test samples: {len(held_out_predictions)}")
    print(f"  • R²: {r2:.4f}" if not np.isnan(r2) else "  • R²: N/A")
    print(f"  • RMSE: {rmse:.4f}" if not np.isnan(rmse) else "  • RMSE: N/A")
    print(f"  • MAE: {mae:.4f}" if not np.isnan(mae) else "  • MAE: N/A")
    if correlation_stats is not None:
        mean_corr = correlation_stats.get('mean_corr')
        max_corr = correlation_stats.get('max_corr')
        band = correlation_stats.get('corr_band')
        valid_neighbors = correlation_stats.get('valid_neighbor_count', 0)
        mean_str = f"{mean_corr:.3f}" if mean_corr is not None else 'N/A'
        max_str = f"{max_corr:.3f}" if max_corr is not None else 'N/A'
        print(f"  • Train neighbour corr (mean/max): {mean_str}/{max_str} ({band}), valid neighbours: {valid_neighbors}")

    result = {
        'fold_idx': fold_idx,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'predictions': held_out_predictions.tolist() if len(held_out_predictions) > 0 else [],
        'targets': held_out_targets.tolist() if len(held_out_targets) > 0 else [],
        'sample_indices': sample_indices,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_test_samples': len(held_out_predictions),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }
    
    if use_physics:
        result['physics_losses'] = physics_losses

    if correlation_stats is not None:
        result['neighbor_correlation'] = correlation_stats

    return result


def analyze_lomo_results(lomo_results, output_dir):
    """Analyze LOMO cross-validation results."""
    
    # Convert to DataFrame for analysis
    records = []
    for r in lomo_results:
        neighbor_stats = r.get('neighbor_correlation') or {}
        mean_neighbor_corr = neighbor_stats.get('mean_corr')
        median_neighbor_corr = neighbor_stats.get('median_corr')
        max_neighbor_corr = neighbor_stats.get('max_corr')
        top3_neighbor_corr = neighbor_stats.get('top3_mean_corr')

        records.append({
            'metabolite_id': r['metabolite_id'],
            'metabolite_idx': r['metabolite_idx'],
            'total_samples': r['total_samples'],
            'n_test_samples': r['n_test_samples'],
            'mse': r['mse'],
            'rmse': r['rmse'],
            'mae': r['mae'],
            'r2': r['r2'],
            'coverage_score': r.get('coverage_score', np.nan),
            'direct_coverage': r.get('direct_coverage', np.nan),
            'far_coverage': r.get('far_coverage', np.nan),
            'mean_neighbor_corr': mean_neighbor_corr if mean_neighbor_corr is not None else np.nan,
            'median_neighbor_corr': median_neighbor_corr if median_neighbor_corr is not None else np.nan,
            'max_neighbor_corr': max_neighbor_corr if max_neighbor_corr is not None else np.nan,
            'top3_neighbor_corr': top3_neighbor_corr if top3_neighbor_corr is not None else np.nan,
            'neighbor_corr_band': neighbor_stats.get('corr_band'),
            'neighbor_corr_valid_count': neighbor_stats.get('valid_neighbor_count', 0),
            'neighbor_corr_pairs': neighbor_stats.get('pair_counts', 0),
        })

    results_df = pd.DataFrame(records)
    
    # Remove NaN values for statistics
    valid_results = results_df.dropna(subset=['r2'])
    
    # Calculate summary statistics
    analysis = {
        'n_metabolites_tested': len(lomo_results),
        'n_metabolites_with_valid_r2': len(valid_results),
        'mean_r2': valid_results['r2'].mean() if len(valid_results) > 0 else np.nan,
        'median_r2': valid_results['r2'].median() if len(valid_results) > 0 else np.nan,
        'std_r2': valid_results['r2'].std() if len(valid_results) > 0 else np.nan,
        'min_r2': valid_results['r2'].min() if len(valid_results) > 0 else np.nan,
        'max_r2': valid_results['r2'].max() if len(valid_results) > 0 else np.nan,
        'mean_rmse': valid_results['rmse'].mean() if len(valid_results) > 0 else np.nan,
        'median_rmse': valid_results['rmse'].median() if len(valid_results) > 0 else np.nan,
        'std_rmse': valid_results['rmse'].std() if len(valid_results) > 0 else np.nan,
        'metabolites_with_positive_r2': (valid_results['r2'] > 0).sum() if len(valid_results) > 0 else 0,
        'metabolites_with_r2_above_0.5': (valid_results['r2'] > 0.5).sum() if len(valid_results) > 0 else 0,
        'percentage_positive_r2': ((valid_results['r2'] > 0).sum() / len(valid_results) * 100) if len(valid_results) > 0 else 0
    }

    valid_corr_df = valid_results.dropna(subset=['mean_neighbor_corr']) if len(valid_results) > 0 else pd.DataFrame()

    if len(valid_corr_df) > 0:
        analysis['mean_neighbor_corr_overall'] = float(valid_corr_df['mean_neighbor_corr'].mean())
        analysis['median_neighbor_corr_overall'] = float(valid_corr_df['mean_neighbor_corr'].median())
        analysis['max_neighbor_corr_overall'] = float(valid_corr_df['mean_neighbor_corr'].max())

        corr_matrix = valid_corr_df[['mean_neighbor_corr', 'r2']].corr()
        corr_value = corr_matrix.iloc[0, 1]
        if not np.isnan(corr_value):
            analysis['neighbor_corr_r2_correlation'] = float(corr_value)

    band_counts = results_df['neighbor_corr_band'].value_counts(dropna=True).to_dict()
    analysis['neighbor_corr_band_counts'] = {str(k): int(v) for k, v in band_counts.items()}

    # Correlation analyses
    if len(valid_results) > 0:
        correlation = valid_results[['total_samples', 'r2']].corr().iloc[0, 1]
        analysis['sample_count_r2_correlation'] = correlation
        if 'coverage_score' in valid_results.columns:
            coverage_corr = valid_results[['coverage_score', 'r2']].corr().iloc[0, 1]
            analysis['coverage_r2_correlation'] = coverage_corr
            analysis['mean_coverage'] = valid_results['coverage_score'].mean()
            analysis['median_coverage'] = valid_results['coverage_score'].median()
        if 'direct_coverage' in valid_results.columns:
            analysis['mean_direct_coverage'] = valid_results['direct_coverage'].mean()
        if 'far_coverage' in valid_results.columns:
            analysis['mean_far_coverage'] = valid_results['far_coverage'].mean()
    
    # Create visualizations
    create_lomo_plots(results_df, analysis, output_dir)
    
    return analysis


def create_lomo_plots(results_df, analysis, output_dir):
    """Create LOMO cross-validation visualizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    valid_results = results_df.dropna(subset=['r2'])
    
    # 1. R² distribution
    if len(valid_results) > 0:
        axes[0, 0].hist(valid_results['r2'], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('LOMO R² Distribution')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='R²=0')
        axes[0, 0].axvline(x=analysis['median_r2'], color='green', linestyle='--',
                          alpha=0.5, label=f'Median={analysis["median_r2"]:.3f}')
        axes[0, 0].legend()
    
    # 2. RMSE distribution
    if len(valid_results) > 0:
        axes[0, 1].hist(valid_results['rmse'], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('RMSE')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('LOMO RMSE Distribution')
        axes[0, 1].axvline(x=analysis['median_rmse'], color='green', linestyle='--',
                          alpha=0.5, label=f'Median={analysis["median_rmse"]:.3f}')
        axes[0, 1].legend()
    
    # 3. R² vs sample count
    if len(valid_results) > 0:
        axes[0, 2].scatter(valid_results['total_samples'], valid_results['r2'], alpha=0.6)
        axes[0, 2].set_xlabel('Total Samples')
        axes[0, 2].set_ylabel('R² Score')
        axes[0, 2].set_title('LOMO Performance vs Sample Count')
        axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(valid_results['total_samples'], valid_results['r2'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_results['total_samples'].min(), 
                            valid_results['total_samples'].max(), 100)
        axes[0, 2].plot(x_line, p(x_line), "g--", alpha=0.8, 
                       label=f'Trend (corr={analysis.get("sample_count_r2_correlation", 0):.3f})')
        axes[0, 2].legend()
    
    # 4. Performance categories
    categories = ['R² > 0.5', '0 < R² ≤ 0.5', 'R² ≤ 0']
    if len(valid_results) > 0:
        counts = [
            (valid_results['r2'] > 0.5).sum(),
            ((valid_results['r2'] > 0) & (valid_results['r2'] <= 0.5)).sum(),
            (valid_results['r2'] <= 0).sum()
        ]
    else:
        counts = [0, 0, 0]
    
    colors = ['green', 'yellow', 'red']
    axes[1, 0].bar(categories, counts, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('Number of Metabolites')
    axes[1, 0].set_title('LOMO Performance Categories')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. MAE vs RMSE
    if len(valid_results) > 0:
        axes[1, 1].scatter(valid_results['mae'], valid_results['rmse'], alpha=0.6)
        axes[1, 1].set_xlabel('MAE')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('MAE vs RMSE')
        
        # Add diagonal line (RMSE = MAE means no outliers)
        max_val = max(valid_results['mae'].max(), valid_results['rmse'].max())
        axes[1, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='RMSE=MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Coverage vs performance
    if len(valid_results) > 0 and 'coverage_score' in valid_results.columns:
        # Filter out NaNs
        valid_coverage = valid_results.dropna(subset=['coverage_score', 'r2'])
        
        if len(valid_coverage) > 1 and valid_coverage['coverage_score'].std() > 1e-6:
            axes[1, 2].scatter(valid_coverage['coverage_score'], valid_coverage['r2'], alpha=0.6)
            axes[1, 2].set_xlabel('Neighbour Coverage Score')
            axes[1, 2].set_ylabel('R² Score')
            axes[1, 2].set_title('R² vs Neighbour Coverage')
            axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 2].grid(True, alpha=0.3)
            
            try:
                z = np.polyfit(valid_coverage['coverage_score'], valid_coverage['r2'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid_coverage['coverage_score'].min(),
                                    valid_coverage['coverage_score'].max(), 100)
                axes[1, 2].plot(
                    x_line,
                    p(x_line),
                    "g--",
                    alpha=0.8,
                    label=f"Trend (corr={analysis.get('coverage_r2_correlation', 0):.3f})",
                )
                axes[1, 2].legend()
            except Exception as e:
                print(f"Warning: Could not fit trend line for coverage vs R2: {e}")
        else:
            axes[1, 2].text(0.5, 0.5, "Insufficient coverage data\nor constant coverage", 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('R² vs Neighbour Coverage (N/A)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lomo_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    valid_corr_results = valid_results.dropna(subset=['mean_neighbor_corr'])
    if len(valid_corr_results) > 0:
        plt.figure(figsize=(6, 5))
        plt.scatter(valid_corr_results['mean_neighbor_corr'], valid_corr_results['r2'], alpha=0.6)
        plt.xlabel('Mean Train Neighbour Correlation')
        plt.ylabel('R² Score')
        plt.title('R² vs Mean Neighbour Correlation')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.4)
        if len(valid_corr_results) > 1:
            z = np.polyfit(valid_corr_results['mean_neighbor_corr'], valid_corr_results['r2'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_corr_results['mean_neighbor_corr'].min(),
                                valid_corr_results['mean_neighbor_corr'].max(), 100)
            label = f"Trend (corr={analysis.get('neighbor_corr_r2_correlation', 0):.3f})"
            plt.plot(x_line, p(x_line), 'g--', alpha=0.8, label=label)
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lomo_neighbor_corr_vs_r2.png'), dpi=300, bbox_inches='tight')
        plt.close()

    band_counts = analysis.get('neighbor_corr_band_counts', {})
    if band_counts:
        plt.figure(figsize=(6, 4))
        bands = list(band_counts.keys())
        counts = [band_counts[b] for b in bands]
        plt.bar(bands, counts, color='steelblue', alpha=0.7)
        plt.ylabel('Metabolites')
        plt.xlabel('Mean Neighbour Correlation Band')
        plt.title('Distribution of Train Neighbour Correlation Bands')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lomo_neighbor_corr_band_counts.png'), dpi=300, bbox_inches='tight')
        plt.close()


def save_lomo_results(lomo_results, analysis, output_dir):
    """Save LOMO cross-validation results."""
    
    # Save detailed results
    csv_records = []
    for r in lomo_results:
        neighbor_stats = r.get('neighbor_correlation') or {}
        mean_neighbor_corr = neighbor_stats.get('mean_corr')
        median_neighbor_corr = neighbor_stats.get('median_corr')
        max_neighbor_corr = neighbor_stats.get('max_corr')
        top3_neighbor_corr = neighbor_stats.get('top3_mean_corr')

        csv_records.append({
            'metabolite_id': r['metabolite_id'],
            'metabolite_idx': r['metabolite_idx'],
            'total_samples': r['total_samples'],
            'n_test_samples': r['n_test_samples'],
            'mse': r['mse'],
            'rmse': r['rmse'],
            'mae': r['mae'],
            'r2': r['r2'],
            'coverage_score': r.get('coverage_score', np.nan),
            'direct_coverage': r.get('direct_coverage', np.nan),
            'far_coverage': r.get('far_coverage', np.nan),
            'mean_neighbor_corr': mean_neighbor_corr if mean_neighbor_corr is not None else np.nan,
            'median_neighbor_corr': median_neighbor_corr if median_neighbor_corr is not None else np.nan,
            'max_neighbor_corr': max_neighbor_corr if max_neighbor_corr is not None else np.nan,
            'top3_neighbor_corr': top3_neighbor_corr if top3_neighbor_corr is not None else np.nan,
            'neighbor_corr_band': neighbor_stats.get('corr_band'),
            'neighbor_corr_valid_count': neighbor_stats.get('valid_neighbor_count', 0),
            'neighbor_corr_pairs': neighbor_stats.get('pair_counts', 0),
        })

    results_df = pd.DataFrame(csv_records)
    
    results_df.to_csv(os.path.join(output_dir, 'lomo_results.csv'), index=False)
    
    # Save full results with predictions
    with open(os.path.join(output_dir, 'lomo_full_results.json'), 'w') as f:
        json.dump(lomo_results, f, indent=2)
    
    # Save analysis summary
    with open(os.path.join(output_dir, 'lomo_summary.txt'), 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LEAVE-ONE-METABOLITE-OUT CROSS-VALIDATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Metabolites tested: {analysis['n_metabolites_tested']}\n")
        f.write(f"Metabolites with valid R²: {analysis['n_metabolites_with_valid_r2']}\n\n")
        
        f.write("R² STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean R²: {analysis['mean_r2']:.4f}\n")
        f.write(f"Median R²: {analysis['median_r2']:.4f}\n")
        f.write(f"Std Dev R²: {analysis['std_r2']:.4f}\n")
        f.write(f"Min R²: {analysis['min_r2']:.4f}\n")
        f.write(f"Max R²: {analysis['max_r2']:.4f}\n\n")
        
        f.write("RMSE STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean RMSE: {analysis['mean_rmse']:.4f}\n")
        f.write(f"Median RMSE: {analysis['median_rmse']:.4f}\n\n")
        
        f.write("PERFORMANCE BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        f.write(f"Metabolites with R² > 0: {analysis['metabolites_with_positive_r2']} "
               f"({analysis['percentage_positive_r2']:.1f}%)\n")
        f.write(f"Metabolites with R² > 0.5: {analysis['metabolites_with_r2_above_0.5']}\n")
        
        if 'sample_count_r2_correlation' in analysis:
            f.write(f"\nSample count vs R² correlation: {analysis['sample_count_r2_correlation']:.3f}\n")
        if 'coverage_r2_correlation' in analysis:
            f.write(
                f"Neighbour coverage vs R² correlation: {analysis['coverage_r2_correlation']:.3f}\n"
            )
        if 'mean_coverage' in analysis:
            f.write(
                f"Average coverage score: {analysis['mean_coverage']:.3f}"
                f" (median={analysis.get('median_coverage', float('nan')):.3f})\n"
            )
        if 'mean_neighbor_corr_overall' in analysis:
            f.write("\nNEIGHBOUR CORRELATION STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"Mean train neighbour correlation: {analysis['mean_neighbor_corr_overall']:.3f}\n"
            )
            f.write(
                f"Median train neighbour correlation: {analysis['median_neighbor_corr_overall']:.3f}\n"
            )
            f.write(
                f"Max train neighbour correlation: {analysis['max_neighbor_corr_overall']:.3f}\n"
            )
            if 'neighbor_corr_r2_correlation' in analysis:
                f.write(
                    f"Correlation between neighbour corr and R²: {analysis['neighbor_corr_r2_correlation']:.3f}\n"
                )
        band_counts = analysis.get('neighbor_corr_band_counts', {})
        if band_counts:
            f.write("Neighbour correlation bands:\n")
            for band, count in band_counts.items():
                f.write(f"  • {band}: {count}\n")

        f.write("\nINTERPRETATION\n")
        f.write("-" * 40 + "\n")
        
        if analysis['percentage_positive_r2'] > 70:
            f.write("✓ Model shows good generalization to unseen metabolites\n")
        elif analysis['percentage_positive_r2'] > 50:
            f.write("→ Model shows moderate generalization to unseen metabolites\n")
        else:
            f.write("⚠️  Model struggles to generalize to unseen metabolites\n")
        
        if analysis.get('sample_count_r2_correlation', 0) > 0.3:
            f.write("✓ Performance improves with more training samples\n")
        else:
            f.write("→ Performance not strongly correlated with sample count\n")


def _save_intermediate_results(results, output_dir):
    """Save intermediate results during LOMO CV."""
    intermediate_path = os.path.join(output_dir, 'lomo_intermediate.json')
    with open(intermediate_path, 'w') as f:
        json.dump(results, f, indent=2)


def _get_metabolite_id(dataset, met_idx):
    """Helper to get metabolite ID."""
    metabolite_id = f"metabolite_{met_idx}"
    
    if hasattr(dataset, 'graph'):
        if hasattr(dataset.graph, 'hmdb_ids') and met_idx < len(dataset.graph.hmdb_ids):
            if dataset.graph.hmdb_ids[met_idx]:
                metabolite_id = dataset.graph.hmdb_ids[met_idx]
    
    return metabolite_id
