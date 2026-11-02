"""Diagnostic utilities for training analysis and visualization."""

import torch
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader
from src.data.batching import collate_metabolic_data
from src.utils.batch_utils import standardize_batch_data, get_batch_size



def _save_bulk_diagnostics(out, delta, flux_vec, embeddings, data, model_dir):
    """Save diagnostic information for bulk transcriptomics mode"""
    
    # Convert to numpy for saving
    predictions_np = out.cpu().numpy()
    delta_np = delta.cpu().numpy()
    flux_vec_np = flux_vec.cpu().numpy()
    
    # Save raw arrays
    np.save(os.path.join(model_dir, 'bulk_predictions.npy'), predictions_np)
    np.save(os.path.join(model_dir, 'bulk_deltas.npy'), delta_np)
    np.save(os.path.join(model_dir, 'bulk_fluxes.npy'), flux_vec_np)
    
    # Save embeddings if available
    if embeddings is not None:
        embeddings_np = embeddings.cpu().numpy()
        np.save(os.path.join(model_dir, 'bulk_embeddings.npy'), embeddings_np)
    
    # Save targets and mask if available
    if hasattr(data, 'y') and hasattr(data, 'mask'):
        targets_np = data.y.cpu().numpy()
        mask_np = data.mask.cpu().numpy()
        np.save(os.path.join(model_dir, 'bulk_targets.npy'), targets_np)
        np.save(os.path.join(model_dir, 'bulk_mask.npy'), mask_np)
    
    # Create flux analysis
    _analyze_fluxes(flux_vec_np.reshape(1, -1), model_dir, "bulk")
    
    # Save metadata
    metadata = {
        'num_nodes': len(predictions_np),
        'num_reactions': len(flux_vec_np),
        'num_metabolites': len(delta_np),
        'flux_stats': {
            'mean': float(np.mean(flux_vec_np)),
            'std': float(np.std(flux_vec_np)),
            'min': float(np.min(flux_vec_np)),
            'max': float(np.max(flux_vec_np)),
            'nonzero_ratio': float(np.mean(np.abs(flux_vec_np) > 1e-6))
        }
    }
    
    with open(os.path.join(model_dir, 'bulk_diagnostics_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    


def _save_basic_diagnostics(out, embeddings, data, model_dir):
    """Save basic diagnostic information for standard mode"""
    
    # Convert to numpy for saving
    predictions_np = out.cpu().numpy()
    
    # Save raw arrays
    np.save(os.path.join(model_dir, 'basic_predictions.npy'), predictions_np)
    
    # Save embeddings if available
    if embeddings is not None:
        embeddings_np = embeddings.cpu().numpy()
        np.save(os.path.join(model_dir, 'basic_embeddings.npy'), embeddings_np)
    
    # Save targets and mask if available
    if hasattr(data, 'y') and hasattr(data, 'mask'):
        targets_np = data.y.cpu().numpy()
        mask_np = data.mask.cpu().numpy()
        np.save(os.path.join(model_dir, 'basic_targets.npy'), targets_np)
        np.save(os.path.join(model_dir, 'basic_mask.npy'), mask_np)
    
    # Save metadata
    metadata = {
        'num_nodes': len(predictions_np),
    }
    
    with open(os.path.join(model_dir, 'basic_diagnostics_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    


def _save_test_diagnostics(test_fluxes, test_deltas, test_embeddings, test_metabolite_preds,
                          test_preds, test_targets, model_dir, use_enhanced_physics=False):
    """Save comprehensive diagnostic information from test set"""
    
    # Convert lists to arrays - handle batched data properly
    if test_fluxes:
        # Concatenate all flux arrays along the batch dimension
        fluxes_array = np.concatenate(test_fluxes, axis=0)
        np.save(os.path.join(model_dir, 'test_fluxes.npy'), fluxes_array)
        
        # Create comprehensive flux analysis
        _analyze_fluxes(fluxes_array, model_dir, "test")
    
    if test_deltas:
        # Concatenate all delta arrays along the batch dimension
        deltas_array = np.concatenate(test_deltas, axis=0)
        np.save(os.path.join(model_dir, 'test_deltas.npy'), deltas_array)
    
    if test_embeddings:
        # Concatenate all embedding arrays along the batch dimension
        embeddings_array = np.concatenate(test_embeddings, axis=0)
        np.save(os.path.join(model_dir, 'test_embeddings.npy'), embeddings_array)
    
    if test_metabolite_preds:
        # Concatenate all prediction arrays along the batch dimension
        preds_array = np.concatenate(test_metabolite_preds, axis=0)
        np.save(os.path.join(model_dir, 'test_metabolite_predictions.npy'), preds_array)
    
    # Save labeled predictions and targets if available
    if test_preds and test_targets:
        # Concatenate all batches properly for predictions and targets
        if isinstance(test_preds[0], np.ndarray):
            preds_array = np.concatenate(test_preds, axis=0) if len(test_preds) > 0 else np.array([])
        else:
            preds_array = np.array(test_preds)
            
        if isinstance(test_targets[0], np.ndarray):
            targets_array = np.concatenate(test_targets, axis=0) if len(test_targets) > 0 else np.array([])
        else:
            targets_array = np.array(test_targets)
            
        np.save(os.path.join(model_dir, 'test_labeled_predictions.npy'), preds_array)
        np.save(os.path.join(model_dir, 'test_labeled_targets.npy'), targets_array)
    
    # Create comprehensive test summary
    _create_test_summary(test_fluxes, test_deltas, test_preds, test_targets, model_dir, use_enhanced_physics)


def _analyze_fluxes(fluxes_array, model_dir, prefix):
    """Create comprehensive flux analysis and visualizations"""
    
    if len(fluxes_array) == 0:
        return
    
    # Reshape if needed
    if len(fluxes_array.shape) == 1:
        fluxes_array = fluxes_array.reshape(1, -1)
    
    # Flux statistics
    flux_stats = {
        'shape': fluxes_array.shape,
        'mean': float(np.mean(fluxes_array)),
        'std': float(np.std(fluxes_array)), 
        'min': float(np.min(fluxes_array)),
        'max': float(np.max(fluxes_array)),
        'median': float(np.median(fluxes_array)),
        'nonzero_ratio': float(np.mean(np.abs(fluxes_array) > 1e-6)),
        'negative_ratio': float(np.mean(fluxes_array < -1e-6)),
        'positive_ratio': float(np.mean(fluxes_array > 1e-6)),
    }
    
    # Per-reaction statistics
    reaction_stats = {
        'mean_per_reaction': np.mean(fluxes_array, axis=0).tolist(),
        'std_per_reaction': np.std(fluxes_array, axis=0).tolist(),
        'max_per_reaction': np.max(fluxes_array, axis=0).tolist(),
        'min_per_reaction': np.min(fluxes_array, axis=0).tolist(),
    }
    
    # Save statistics
    with open(os.path.join(model_dir, f'{prefix}_flux_analysis.json'), 'w') as f:
        json.dump({'overall': flux_stats, 'per_reaction': reaction_stats}, f, indent=2)
    
    # Create flux histogram
    try:
        plt.figure(figsize=(12, 8))
        
        # Overall flux distribution
        plt.subplot(2, 2, 1)
        plt.hist(fluxes_array.flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Flux Value')
        plt.ylabel('Frequency')
        plt.title(f'{prefix.title()} Flux Distribution')
        plt.yscale('log')
        
        # Nonzero flux distribution
        plt.subplot(2, 2, 2)
        nonzero_fluxes = fluxes_array[np.abs(fluxes_array) > 1e-6]
        if len(nonzero_fluxes) > 0:
            plt.hist(nonzero_fluxes, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Nonzero Flux Value')
            plt.ylabel('Frequency')
            plt.title(f'{prefix.title()} Nonzero Fluxes Only')
        
        # Per-reaction mean fluxes
        plt.subplot(2, 2, 3)
        reaction_means = np.mean(fluxes_array, axis=0)
        plt.plot(reaction_means, 'o-', alpha=0.6)
        plt.xlabel('Reaction Index')
        plt.ylabel('Mean Flux')
        plt.title(f'{prefix.title()} Mean Flux per Reaction')
        
        # Per-reaction nonzero count
        plt.subplot(2, 2, 4)
        nonzero_counts = np.sum(np.abs(fluxes_array) > 1e-6, axis=0)
        plt.plot(nonzero_counts, 'o-', alpha=0.6)
        plt.xlabel('Reaction Index')
        plt.ylabel('Nonzero Count')
        plt.title(f'{prefix.title()} Nonzero Count per Reaction')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f'{prefix}_flux_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        
    except Exception:
        pass
    
    return flux_stats


def _create_test_summary(test_fluxes, test_deltas, test_preds, test_targets, model_dir, use_enhanced_physics=False):
    """Create comprehensive test summary with key diagnostics"""

    summary = {
        'test_mode': 'bulk',
        'physics_mode': 'enhanced' if use_enhanced_physics else 'standard',
        'timestamp': pd.Timestamp.now().isoformat(),
    }
    
    # Add flux diagnostics if available
    if test_fluxes:
        # Concatenate all batches properly
        if isinstance(test_fluxes[0], np.ndarray):
            fluxes_array = np.concatenate(test_fluxes, axis=0)
        else:
            fluxes_array = np.array(test_fluxes)
        
        summary['flux_diagnostics'] = {
            'total_samples': fluxes_array.shape[0] if len(fluxes_array.shape) > 1 else 1,
            'total_reactions': fluxes_array.shape[-1],
            'mean_flux': float(np.mean(fluxes_array)),
            'std_flux': float(np.std(fluxes_array)),
            'nonzero_flux_ratio': float(np.mean(np.abs(fluxes_array) > 1e-6)),
            'max_abs_flux': float(np.max(np.abs(fluxes_array))),
            'reactions_with_nonzero_flux': int(np.sum(np.any(np.abs(fluxes_array) > 1e-6, axis=0) if len(fluxes_array.shape) > 1 else np.abs(fluxes_array) > 1e-6)),
        }
        
        # Flag potential issues
        if summary['flux_diagnostics']['nonzero_flux_ratio'] < 0.01:
            summary['warnings'] = summary.get('warnings', [])
            summary['warnings'].append("Very low nonzero flux ratio (<1%) - model may be predicting mostly zeros")
        
        if summary['flux_diagnostics']['max_abs_flux'] < 1e-3:
            summary['warnings'] = summary.get('warnings', [])
            summary['warnings'].append("Very small flux magnitudes (<1e-3) - check flux scaling")
    
    # Add mass balance diagnostics if available
    if test_deltas:
        # Concatenate all batches properly
        if isinstance(test_deltas[0], np.ndarray):
            deltas_array = np.concatenate(test_deltas, axis=0)
        else:
            deltas_array = np.array(test_deltas)
        
        summary['mass_balance_diagnostics'] = {
            'mean_squared_residual': float(np.mean(deltas_array ** 2)),
            'max_abs_residual': float(np.max(np.abs(deltas_array))),
            'residual_std': float(np.std(deltas_array)),
        }
    
    # Add prediction quality if available
    if test_preds and test_targets:
        # Concatenate all batches properly for predictions and targets
        if isinstance(test_preds[0], np.ndarray):
            preds_array = np.concatenate(test_preds, axis=0) if len(test_preds) > 0 else np.array([])
        else:
            preds_array = np.array(test_preds)
            
        if isinstance(test_targets[0], np.ndarray):
            targets_array = np.concatenate(test_targets, axis=0) if len(test_targets) > 0 else np.array([])
        else:
            targets_array = np.array(test_targets)
            
        summary['prediction_quality'] = {
            'num_labeled_predictions': len(preds_array),
            'r2_score': float(r2_score(targets_array, preds_array)),
            'rmse': float(np.sqrt(mean_squared_error(targets_array, preds_array))),
            'prediction_mean': float(np.mean(preds_array)),
            'target_mean': float(np.mean(targets_array)),
        }
    
    # Save summary
    with open(os.path.join(model_dir, 'test_diagnostics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    


def save_per_metabolite_metrics(model_dir, dataset, model, device, test_indices):
    """
    Generate and save detailed per-metabolite performance metrics
    
    Args:
        model_dir: Directory to save results
        dataset: The full dataset
        model: Trained model
        device: Device to run inference on
        test_indices: Indices of test samples
    """
    
    # First, calculate training sample counts for each metabolite
    train_indices = [i for i in range(len(dataset)) if i not in test_indices]
    metabolite_train_counts = {}
    
    # Count how many training samples each metabolite appears in
    for train_idx in train_indices:
        data = dataset[train_idx]
        mask = data.mask.cpu().numpy()
        
        # Handle both 1D and 2D masks (2D if batched with batch_size=1)
        if mask.ndim == 2 and mask.shape[0] == 1:
            mask = mask[0]  # Remove batch dimension
        
        for i, is_measured in enumerate(mask):
            if is_measured and i in dataset.metabolite_indices:
                # Get metabolite ID
                if hasattr(dataset.graph, 'hmdb_ids') and dataset.graph.hmdb_ids[i]:
                    metabolite_id = dataset.graph.hmdb_ids[i]
                elif hasattr(dataset.graph, 'kegg_ids') and dataset.graph.kegg_ids[i]:
                    metabolite_id = dataset.graph.kegg_ids[i]
                else:
                    metabolite_id = f"metabolite_{i}"
                
                if metabolite_id not in metabolite_train_counts:
                    metabolite_train_counts[metabolite_id] = 0
                metabolite_train_counts[metabolite_id] += 1
    
    print(f"Calculated training counts for {len(metabolite_train_counts)} metabolites")
    
    # Create a dictionary to store predictions and targets for each metabolite
    metabolite_results = {}
    
    # Dictionary to store all predictions across all samples
    all_predictions = {
        'metabolite_id': [],
        'sample_id': [],
        'predicted': [],
        'actual': []
    }
    
    # Define sample size bins for stratified metrics (based on training sample count)
    size_bins = [(5, 25), (26, 75), (76, 200), (201, 400), (401, float('inf'))]
    bin_labels = ['5-25', '26-75', '76-200', '201-400', '401+']
    
    # Dictionaries to store stratified results
    stratified_preds = {label: {'preds': [], 'targets': []} for label in bin_labels}
    stratified_results = {label: {'metabolites': [], 'r2': 0.0, 'rmse': 0.0, 'count': 0} for label in bin_labels}
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dataloader for test samples
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_metabolic_data)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if batch_idx % 100 == 0:
                print(f"Processing sample {batch_idx}/{len(loader)}")
                
            data = data.to(device)
            
            # Standardize data to ensure proper batching
            data = standardize_batch_data(data)
            
            # Set num_nodes to avoid warning - handle both 2D and 3D tensors
            if not hasattr(data, 'num_nodes'):
                if data.x_embedding.dim() == 3:
                    data.num_nodes = data.x_embedding.size(1)  # Middle dimension for batched [1, num_nodes, dim]
                else:
                    data.num_nodes = data.x_embedding.size(0)  # First dimension for unbatched [num_nodes, dim]
            
            # Get model output - only use metabolite predictions, ignore physics outputs
            out, *_ = model(data, return_embeddings=False, compute_physics=False)  # No physics needed for per-metabolite metrics
            
            # Debug output shape
            if batch_idx == 0:
                print(f"DEBUG: Model output shape: {out.shape}")
                print(f"DEBUG: Data x_embedding shape: {data.x_embedding.shape}")
                print(f"DEBUG: Data mask shape: {data.mask.shape}")
                print(f"DEBUG: Data y shape: {data.y.shape}")
            
            # Model now always returns [batch_size, num_nodes]
            # For single sample evaluation, we need to extract the first sample
            if out.dim() == 2 and out.size(0) == 1:
                out = out.squeeze(0)  # [1, num_nodes] -> [num_nodes]
            elif out.dim() != 1:
                print(f"WARNING: Unexpected output shape: {out.shape}")
                out = out[0] if out.dim() == 2 else out.view(-1)
            
            # Get mask, predictions and targets
            mask = data.mask.cpu().numpy()
            y_numpy = data.y.cpu().numpy()
            
            # Handle both 1D and 2D tensors (2D if batched with batch_size=1)
            if mask.ndim == 2 and mask.shape[0] == 1:
                mask = mask[0]  # Remove batch dimension
            if y_numpy.ndim == 2 and y_numpy.shape[0] == 1:
                y_numpy = y_numpy[0]  # Remove batch dimension
            
            # Debug before indexing
            if batch_idx == 0:
                print(f"DEBUG: After processing - out shape: {out.shape}")
                print(f"DEBUG: Mask shape: {mask.shape}")
                print(f"DEBUG: Number of True values in mask: {mask.sum()}")
                print(f"DEBUG: Max index in mask: {np.where(mask)[0].max() if mask.sum() > 0 else -1}")
                
            preds = out.cpu().numpy()[mask]
            targets = y_numpy[mask]
            
            # Get metabolite IDs and store predictions
            # IMPORTANT: Only process indices that are actual metabolite nodes
            all_masked_indices = np.where(mask)[0]
            
            # Get the list of valid metabolite node indices from data
            if hasattr(data, 'metabolite_indices'):
                valid_metabolite_indices = set(data.metabolite_indices.cpu().numpy() if torch.is_tensor(data.metabolite_indices) else data.metabolite_indices)
            else:
                # Fallback: use dataset's metabolite_indices
                valid_metabolite_indices = set(dataset.metabolite_indices)
            
            # Filter to only include actual metabolite nodes
            metabolite_mask = [idx for idx in all_masked_indices if idx in valid_metabolite_indices]
            
            sample_idx = test_indices[batch_idx]
            
            # Create a mapping from full mask indices to filtered prediction indices
            mask_to_pred_idx = {idx: i for i, idx in enumerate(all_masked_indices)}
            
            for met_idx in metabolite_mask:
                # Get the corresponding index in the predictions array
                pred_idx = mask_to_pred_idx[met_idx]
                # Skip NaN values
                if np.isnan(preds[pred_idx]) or np.isnan(targets[pred_idx]):
                    continue
                    
                # Get metabolite ID
                metabolite_id = "unknown"
                
                # Try to find a meaningful ID for this metabolite
                # Note: graph.hmdb_ids and graph.kegg_ids are indexed by the full node index
                if hasattr(dataset, 'graph'):
                    if hasattr(dataset.graph, 'hmdb_ids') and met_idx < len(dataset.graph.hmdb_ids) and dataset.graph.hmdb_ids[met_idx]:
                        metabolite_id = dataset.graph.hmdb_ids[met_idx]
                    elif hasattr(dataset.graph, 'kegg_ids') and met_idx < len(dataset.graph.kegg_ids) and dataset.graph.kegg_ids[met_idx]:
                        metabolite_id = dataset.graph.kegg_ids[met_idx]
                
                if metabolite_id == "unknown":
                    if hasattr(dataset, 'hmdb_to_idx'):
                        for hmdb_id, idx in dataset.hmdb_to_idx.items():
                            if idx == met_idx:
                                metabolite_id = hmdb_id
                                break
                    
                    if metabolite_id == "unknown" and hasattr(dataset, 'kegg_to_idx'):
                        for kegg_id, idx in dataset.kegg_to_idx.items():
                            if idx == met_idx:
                                metabolite_id = kegg_id
                                break
                
                if metabolite_id == "unknown":
                    metabolite_id = f"metabolite_{met_idx}"
                
                # Initialize entry if first time seeing this metabolite
                if metabolite_id not in metabolite_results:
                    metabolite_results[metabolite_id] = {
                        'predictions': [],
                        'targets': [],
                        'sample_indices': []
                    }
                
                # Add prediction and target
                metabolite_results[metabolite_id]['predictions'].append(float(preds[pred_idx]))
                metabolite_results[metabolite_id]['targets'].append(float(targets[pred_idx]))
                metabolite_results[metabolite_id]['sample_indices'].append(sample_idx)
                
                # Also add to all_predictions dictionary
                all_predictions['metabolite_id'].append(metabolite_id)
                all_predictions['sample_id'].append(sample_idx)
                all_predictions['predicted'].append(float(preds[pred_idx]))
                all_predictions['actual'].append(float(targets[pred_idx]))
    
    # Save all predictions
    all_predictions_df = pd.DataFrame(all_predictions)
    all_predictions_df.to_csv(os.path.join(model_dir, 'all_predictions.csv'), index=False)
    
    # Create directories for detailed results
    predictions_dir = os.path.join(model_dir, 'per_metabolite_predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Calculate metrics for each metabolite
    metrics = []
    
    for metabolite_id, results in metabolite_results.items():
        if not results['predictions'] or not results['targets']:
            continue
            
        predictions = np.array(results['predictions'])
        targets = np.array(results['targets'])
        sample_count = len(predictions)
        
        # Skip if we don't have enough samples for reliable metrics
        if sample_count < 5:
            continue
            
        # Basic error metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # Bias and variance metrics
        mean_error = np.mean(predictions - targets)
        std_error = np.std(predictions - targets)
        
        # R-squared calculation
        if np.var(targets) > 0:
            r2 = 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))
        else:
            r2 = np.nan
        
        # Calculate percentile-based metrics
        p25_error = np.percentile(np.abs(predictions - targets), 25)
        p50_error = np.percentile(np.abs(predictions - targets), 50)  # median error
        p75_error = np.percentile(np.abs(predictions - targets), 75)
        
        # Target statistics
        target_mean = np.mean(targets)
        target_std = np.std(targets)
        
        # Prediction statistics
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        
        # Correlation
        if np.std(predictions) > 0 and np.std(targets) > 0:
            correlation = np.corrcoef(predictions, targets)[0, 1]
        else:
            correlation = np.nan
        
        # Store all metrics
        metrics.append({
            'metabolite_id': metabolite_id,
            'sample_count': sample_count,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mean_error': mean_error,
            'std_error': std_error,
            'p25_error': p25_error,
            'p50_error': p50_error,
            'p75_error': p75_error,
            'target_mean': target_mean,
            'target_std': target_std,
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'correlation': correlation,
            'train_sample_count': metabolite_train_counts.get(metabolite_id, 0)
        })
        
        # Add to stratified metrics based on training sample size
        train_sample_count = metabolite_train_counts.get(metabolite_id, 0)
        for i, (min_size, max_size) in enumerate(size_bins):
            if min_size <= train_sample_count <= max_size:
                bin_label = bin_labels[i]
                stratified_results[bin_label]['metabolites'].append(metabolite_id)
                stratified_results[bin_label]['count'] += 1
                # Add ALL predictions for this metabolite to the appropriate bin
                stratified_preds[bin_label]['preds'].extend(predictions.tolist())
                stratified_preds[bin_label]['targets'].extend(targets.tolist())
                break
        
        # For metabolites with reasonable sample size, save individual predictions
        # and create scatter plots
        if sample_count >= 20:
            # Save to CSV
            metabolite_df = pd.DataFrame({
                'sample_id': results['sample_indices'],
                'predicted': predictions,
                'actual': targets
            })
            
            # Handle special characters in metabolite IDs for filenames
            safe_id = metabolite_id.replace(':', '_').replace('/', '_').replace('(', '_').replace(')', '_')
            metabolite_file = os.path.join(predictions_dir, f"{safe_id}.csv")
            metabolite_df.to_csv(metabolite_file, index=False)
            
            # Generate scatter plot
            plt.figure(figsize=(8, 8))
            plt.scatter(targets, predictions, alpha=0.6)
            
            # Add diagonal line (perfect predictions)
            min_val = min(min(targets), min(predictions))
            max_val = max(max(targets), max(predictions))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Add title and labels
            plt.title(f"{metabolite_id}\nSamples: {sample_count}, R²: {r2:.4f}, RMSE: {rmse:.4f}")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(predictions_dir, f"{safe_id}.png"))
            plt.close()
    
    # Convert to DataFrame and sort by sample count (descending)
    metrics_df = pd.DataFrame(metrics)
    
    # Check if we have any metrics
    if len(metrics_df) == 0:
        print("Warning: No metabolite metrics were collected. This might be due to insufficient samples.")
        # Create empty report indicating no metrics available
        html_content = """
        <html>
        <head><title>Per-Metabolite Metrics Report</title></head>
        <body>
            <h1>Per-Metabolite Metrics Report</h1>
            <p>No metabolite metrics were collected. This typically happens when:</p>
            <ul>
                <li>Test set is too small</li>
                <li>No metabolites have sufficient samples (minimum 5 required)</li>
                <li>Data processing errors</li>
            </ul>
        </body>
        </html>
        """
        report_file = os.path.join(model_dir, 'per_metabolite_report.html')
        with open(report_file, 'w') as f:
            f.write(html_content)
        return
    
    metrics_df = metrics_df.sort_values('sample_count', ascending=False)
    
    # Save to CSV
    metrics_file = os.path.join(model_dir, 'per_metabolite_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)
    
    # Create a summary of the best and worst predicted metabolites
    # Sort by R2 for metabolites with at least 30 samples
    common_metabolites = metrics_df[metrics_df['sample_count'] >= 30].copy()
    if len(common_metabolites) > 0:
        best_r2 = common_metabolites.sort_values('r2', ascending=False).head(10)
        worst_r2 = common_metabolites.sort_values('r2', ascending=True).head(10)
        
        # Save summaries
        best_file = os.path.join(model_dir, 'best_predicted_metabolites.csv')
        worst_file = os.path.join(model_dir, 'worst_predicted_metabolites.csv')
        
        best_r2.to_csv(best_file, index=False)
        worst_r2.to_csv(worst_file, index=False)
        
    
    # Create figures directory
    figures_dir = os.path.join(model_dir, 'metabolite_figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Calculate stratified metrics
    for bin_label in bin_labels:
        bin_preds = stratified_preds[bin_label]['preds']
        bin_targets = stratified_preds[bin_label]['targets']
        
        if len(bin_preds) > 1:  # Need at least 2 predictions for R2
            bin_r2 = r2_score(bin_targets, bin_preds)
            bin_rmse = np.sqrt(mean_squared_error(bin_targets, bin_preds))
            
            stratified_results[bin_label]['r2'] = float(bin_r2)
            stratified_results[bin_label]['rmse'] = float(bin_rmse)
            
            print(f"Training Sample Size {bin_label} (Metabolites: {stratified_results[bin_label]['count']}):")
            print(f"  R2: {bin_r2:.4f}")
            print(f"  RMSE: {bin_rmse:.4f}")
            
            # Create scatter plot for this bin
            plt.figure(figsize=(10, 8))
            plt.scatter(bin_targets, bin_preds, alpha=0.5, s=10)
            
            # Add diagonal line (perfect predictions)
            min_val = min(min(bin_targets), min(bin_preds))
            max_val = max(max(bin_targets), max(bin_preds))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title(f"Predictions for Metabolites with {bin_label} Samples\n"
                     f"Metabolites: {stratified_results[bin_label]['count']}, R²: {bin_r2:.4f}, RMSE: {bin_rmse:.4f}")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'stratified_{bin_label.replace("-", "_")}.png'))
            plt.close()
    
    # Save stratified results
    with open(os.path.join(model_dir, 'stratified_metrics.json'), 'w') as f:
        json.dump(stratified_results, f, indent=2)
    
    # Generate a simple HTML report with visualizations
    try:
        # Create histograms of R2 and RMSE
        plt.figure(figsize=(10, 6))
        plt.hist(metrics_df['r2'].dropna(), bins=20)
        plt.xlabel('R² Score')
        plt.ylabel('Number of Metabolites')
        plt.title('Distribution of R² Scores Across Metabolites')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'r2_distribution.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.hist(metrics_df['rmse'].dropna(), bins=20)
        plt.xlabel('RMSE')
        plt.ylabel('Number of Metabolites')
        plt.title('Distribution of RMSE Across Metabolites')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'rmse_distribution.png'))
        plt.close()
        
        # Scatter plot of sample count vs. R2
        plt.figure(figsize=(10, 6))
        plt.scatter(metrics_df['sample_count'], metrics_df['r2'], alpha=0.5)
        plt.xlabel('Number of Samples')
        plt.ylabel('R² Score')
        plt.title('Prediction Quality vs. Sample Size')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'sample_count_vs_r2.png'))
        plt.close()
        
        # Generate simple HTML report
        html_content = f"""
        <html>
        <head>
            <title>Metabolite Prediction Analysis (Concat GNN + SMILES)</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .figure {{ margin: 20px 0; }}
                .figure img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Metabolite Prediction Analysis (Concat GNN + SMILES)</h1>
            <p>Analysis of model performance across {len(metrics_df)} metabolites.</p>
            
            <h2>Stratified Performance by Sample Size</h2>
            <table>
                <tr>
                    <th>Sample Size</th>
                    <th>Number of Metabolites</th>
                    <th>R²</th>
                    <th>RMSE</th>
                </tr>
                {"".join([f"<tr><td>{label}</td><td>{stratified_results[label]['count']}</td><td>{stratified_results[label]['r2']:.4f}</td><td>{stratified_results[label]['rmse']:.4f}</td></tr>" for label in bin_labels if stratified_results[label]['count'] > 0])}
            </table>
            
            {"".join([f'<div class="figure"><img src="metabolite_figures/stratified_{label.replace("-", "_")}.png" alt="Predictions for {label} samples"><p>Aggregate predictions for metabolites with {label} samples.</p></div>' for label in bin_labels if stratified_results[label]['count'] > 0])}
            
            <h2>Performance Distribution</h2>
            <div class="figure">
                <img src="metabolite_figures/r2_distribution.png" alt="R² Distribution">
                <p>Distribution of R² values across metabolites.</p>
            </div>
            
            <div class="figure">
                <img src="metabolite_figures/rmse_distribution.png" alt="RMSE Distribution">
                <p>Distribution of RMSE values across metabolites.</p>
            </div>
            
            <div class="figure">
                <img src="metabolite_figures/sample_count_vs_r2.png" alt="Sample Count vs R²">
                <p>Relationship between sample count and prediction quality.</p>
            </div>
            
            <h2>Summary Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
                <tr>
                    <td>R²</td>
                    <td>{metrics_df['r2'].mean():.4f}</td>
                    <td>{metrics_df['r2'].median():.4f}</td>
                    <td>{metrics_df['r2'].min():.4f}</td>
                    <td>{metrics_df['r2'].max():.4f}</td>
                </tr>
                <tr>
                    <td>RMSE</td>
                    <td>{metrics_df['rmse'].mean():.4f}</td>
                    <td>{metrics_df['rmse'].median():.4f}</td>
                    <td>{metrics_df['rmse'].min():.4f}</td>
                    <td>{metrics_df['rmse'].max():.4f}</td>
                </tr>
                <tr>
                    <td>Sample Count</td>
                    <td>{metrics_df['sample_count'].mean():.1f}</td>
                    <td>{metrics_df['sample_count'].median():.1f}</td>
                    <td>{metrics_df['sample_count'].min()}</td>
                    <td>{metrics_df['sample_count'].max()}</td>
                </tr>
            </table>
            
            <h2>Top 10 Best Predicted Metabolites</h2>
            <table>
                <tr>
                    <th>Metabolite ID</th>
                    <th>Sample Count</th>
                    <th>R²</th>
                    <th>RMSE</th>
                </tr>
                {"".join([f"<tr><td>{row['metabolite_id']}</td><td>{row['sample_count']}</td><td>{row['r2']:.4f}</td><td>{row['rmse']:.4f}</td></tr>" for _, row in common_metabolites.sort_values('r2', ascending=False).head(10).iterrows()]) if len(common_metabolites) > 0 else "<tr><td colspan='4'>No metabolites with sufficient samples</td></tr>"}
            </table>
            
            <h2>Top 10 Worst Predicted Metabolites</h2>
            <table>
                <tr>
                    <th>Metabolite ID</th>
                    <th>Sample Count</th>
                    <th>R²</th>
                    <th>RMSE</th>
                </tr>
                {"".join([f"<tr><td>{row['metabolite_id']}</td><td>{row['sample_count']}</td><td>{row['r2']:.4f}</td><td>{row['rmse']:.4f}</td></tr>" for _, row in common_metabolites.sort_values('r2', ascending=True).head(10).iterrows()]) if len(common_metabolites) > 0 else "<tr><td colspan='4'>No metabolites with sufficient samples</td></tr>"}
            </table>
            
            <p>For complete details, see the CSV files in the results directory.</p>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(os.path.join(model_dir, 'metabolite_report.html'), 'w') as f:
            f.write(html_content)
    except Exception:
        pass
    
    return metrics_df