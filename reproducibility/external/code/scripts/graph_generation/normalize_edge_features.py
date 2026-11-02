import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def normalize_kcat_values(graph_path, output_path=None, visualize=True):
    """
    Normalize kcat values in a graph and save the normalized version
    
    Args:
        graph_path: Path to the graph with edge attributes
        output_path: Path to save normalized graph (if None, uses original path with _normalized suffix)
        visualize: Whether to create visualization of the normalization
    
    Returns:
        Normalized graph and normalization stats
    """
    print(f"Loading graph from {graph_path}")
    graph = torch.load(graph_path, weights_only=False)
    
    if not hasattr(graph, 'edge_attr') or graph.edge_attr is None:
        print("Graph does not have edge attributes. Nothing to normalize.")
        return graph, None
    
    print(f"Graph has {graph.edge_attr.size(0)} edges with {graph.edge_attr.size(1)} attributes per edge")
    
    # Extract kcat values (assuming they're in the first dimension of edge_attr)
    kcat_values = graph.edge_attr[:, 0].numpy()
    
    # Print some basic statistics
    print(f"kcat value statistics before normalization:")
    print(f"  Min: {kcat_values.min()}")
    print(f"  Max: {kcat_values.max()}")
    print(f"  Mean: {kcat_values.mean()}")
    print(f"  Median: {np.median(kcat_values)}")
    print(f"  Std: {kcat_values.std()}")
    
    # Count default values (where kcat == 1)
    default_count = np.sum(kcat_values == 1.0)
    print(f"  Number of default values (kcat=1): {default_count} ({default_count/len(kcat_values)*100:.2f}%)")
    
    # Identify non-default values (where kcat != 1)
    non_default_mask = kcat_values != 1.0
    non_default_values = kcat_values[non_default_mask]
    
    if visualize:
        # Create a directory for visualizations
        vis_dir = os.path.join(os.path.dirname(graph_path), "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Visualize distribution of kcat values
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(kcat_values, bins=50, alpha=0.7)
        plt.title('Distribution of Original kcat Values')
        plt.xlabel('kcat')
        plt.ylabel('Count')
        plt.yscale('log')  # Log scale for better visibility
        
        plt.subplot(2, 2, 2)
        plt.hist(non_default_values, bins=50, alpha=0.7)
        plt.title('Distribution of Non-default kcat Values')
        plt.xlabel('kcat')
        plt.ylabel('Count')
        plt.yscale('log')
    
    # Only normalize if we have non-default values
    if np.sum(non_default_mask) > 0:
        # Log10 transformation with small epsilon to avoid log(0)
        epsilon = 1e-10
        log_kcat = np.log10(np.maximum(kcat_values, epsilon))
        
        # Min-max scaling on the log values to get [0,1] range
        log_min = np.min(log_kcat[non_default_mask])  # This will be negative (around -4 for your min value)
        log_max = np.max(log_kcat[non_default_mask])  # This will be around 4.4 for your max
        
        # Scale to [0, 1] range
        normalized_kcat = (log_kcat - log_min) / (log_max - log_min)
        
        # Store the normalized value for the default
        default_normalized = (np.log10(1.0) - log_min) / (log_max - log_min)  # Will be around 0.5
        
        if visualize:
            plt.subplot(2, 2, 3)
            plt.hist(log_kcat, bins=50, alpha=0.7)
            plt.title('Distribution after Log Transformation')
            plt.xlabel('log(kcat)')
            plt.ylabel('Count')
            
            plt.subplot(2, 2, 4)
            plt.hist(normalized_kcat, bins=50, alpha=0.7)
            plt.title('Distribution after Normalization')
            plt.xlabel('normalized kcat')
            plt.ylabel('Count')
            
            plt.tight_layout()
            vis_path = os.path.join(vis_dir, f"kcat_normalization_{os.path.basename(graph_path).split('.')[0]}.png")
            plt.savefig(vis_path)
            print(f"Saved visualization to {vis_path}")
            plt.close()
        
        # Print statistics after normalization
        print(f"kcat value statistics after normalization:")
        print(f"  Min: {normalized_kcat.min()}")
        print(f"  Max: {normalized_kcat.max()}")
        print(f"  Mean: {normalized_kcat.mean()}")
        print(f"  Median: {np.median(normalized_kcat)}")
        print(f"  Std: {normalized_kcat.std()}")
        
        # Create a new edge_attr tensor
        new_edge_attr = graph.edge_attr.clone()
        new_edge_attr[:, 0] = torch.tensor(normalized_kcat, dtype=torch.float)
        
        # Update the graph
        graph.edge_attr = new_edge_attr
        graph.edge_attr_stats = {
            'log_min': float(log_min),
            'log_max': float(log_max),
            'transform': 'log10+minmax',
            'default_value': 1.0,
            'normalized_default': float(default_normalized),
            'original_min': float(kcat_values.min()),
            'original_max': float(kcat_values.max()),
            'original_mean': float(kcat_values.mean()),
            'original_std': float(kcat_values.std()),
            'default_count': int(default_count),
            'total_edges': int(len(kcat_values)),
            'non_default_count': int(np.sum(non_default_mask))
        }
        
        # Save normalized graph
        if output_path is None:
            base, ext = os.path.splitext(graph_path)
            output_path = f"{base}_normalized{ext}"
        
        print(f"Saving normalized graph to {output_path}")
        torch.save(graph, output_path)
        
        return graph, graph.edge_attr_stats
    else:
        print("No non-default kcat values found. Using original graph.")
        return graph, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Normalize edge features in graph')
    parser.add_argument('--embeddings', action='store_true', 
                      help='Normalize graph with embeddings (default is gene features)')
    parser.add_argument('--input-dir', type=str, default=None,
                      help='Custom input directory (overrides embeddings flag)')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Custom output directory')
    
    args = parser.parse_args()
    
    # Set paths based on arguments
    if args.input_dir:
        graph_path = os.path.join(args.input_dir, 'merged_metabolic_network.pt')
        graph_type = 'custom'
    elif args.embeddings:
        graph_path = '/home/cnovellarausell/data1/Metabolomics-RNA/source-data/PyG_merged_graph_with_embeddings/merged_metabolic_network.pt'
        graph_type = 'embeddings'
    else:
        graph_path = '/home/cnovellarausell/data1/Metabolomics-RNA/source-data/PyG_merged_graph_with_gene_features/merged_metabolic_network.pt' 
        graph_type = 'gene_features'
    
    # Set output path
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_dir = os.path.dirname(graph_path)
        output_dir = os.path.join(base_dir, f"normalized_{graph_type}")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'merged_metabolic_network.pt')
    
    # Run normalization
    normalized_graph, stats = normalize_kcat_values(graph_path, output_path)
    
    if stats:
        # Save stats as JSON for reference
        import json
        stats_path = os.path.join(output_dir, 'normalization_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved normalization stats to {stats_path}") 