import os
import sys
import json
import glob
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
import networkx as nx

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.gene_parsing import parse_gene_rule_simple as parse_gene_rule

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Initialize enzyme embeddings for metabolic graphs')
    parser.add_argument('--identifier', type=str, default='', help='Suffix for directories (e.g. masked)')
    args = parser.parse_args()

    suffix = f"_{args.identifier}" if args.identifier else ""
    output_base_path = "/home/cnovellarausell/data1/concat-smiles-model/data/graphs"
    input_dir = f"{output_base_path}/PyG_Graphs_with_SMILES{suffix}"
    output_dir = f"{output_base_path}/PyG_Graphs_with_SMILES_with_embeddings{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Initializing enzyme embeddings from: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    add_enzyme_embeddings(input_dir, output_dir)

def add_enzyme_embeddings(input_dir, output_dir):
    """
    Add EC2Vec embeddings to reaction nodes.
    Only include reactions that have measured gene expressions.
    """
    source_data_path = "/home/cnovellarausell/data1/Metabolomics-RNA/source-data"
    # Load EC number embeddings
    ec_embeddings_df = pd.read_csv(f'{source_data_path}/embedded_EC_number.csv')
    # ... dictionary logic ...
    ec_embeddings_dict = {}
    for _, row in ec_embeddings_df.iterrows():
        ec_number = row['EC_number']
        embedding = row.values[1:].astype(np.float32)
        ec_embeddings_dict[ec_number] = embedding
    
    # Load RNA expression data
    import mudata as mu
    mdata = mu.read_h5mu(f'{source_data_path}/mudata_correct_HMDB_new_no_negatives.h5mu')
    gene_expression = mdata['rna']
    
    # Create mapping from gene ID to expression values
    gene_to_expression = {}
    for i, gene_id in enumerate(gene_expression.var['ensembl_gene_id']):
        mean_expr = gene_expression.X[:, i].mean()
        gene_to_expression[gene_id] = mean_expr
    
    # Get all JSON files in input directory
    json_files = glob.glob(f'{input_dir}/*.json')
    
    # Statistics tracking
    total_pathways = 0
    skipped_pathways = 0
    reactions_removed_count = 0
    reactions_kept_count = 0
    pathways_with_no_reactions = 0
    
    # Process each JSON file
    for json_file in json_files:
        total_pathways += 1
        base_name = os.path.basename(json_file)
        pathway_name = base_name.replace("_metadata.json", "")
        
        # Construct filenames
        pkl_file = f'{input_dir}/{pathway_name}_nx.pkl'
        pyg_file = f'{input_dir}/{pathway_name}.pt'
        output_pyg_file = f'{output_dir}/{pathway_name}.pt'
        output_pkl_file = f'{output_dir}/{pathway_name}_nx.pkl'
        
        # Load JSON metadata
        with open(json_file, 'r') as f:
            metadata = json.load(f)

        # Store original node mapping BEFORE any filtering modifies it.
        # ec_numbers is indexed by original node indices, so we need this
        # to correctly look up EC numbers after node removal remaps indices.
        if 'node_mapping' in metadata and 'original_node_mapping' not in metadata:
            metadata['original_node_mapping'] = dict(metadata['node_mapping'])

        # Load PyG Data object
        if os.path.exists(pyg_file):
            try:
                data = torch.load(pyg_file, weights_only=False)
            except Exception as e:
                print(f"  Error loading PyG file: {e}")
                skipped_pathways += 1
                continue
        else:
            print(f"  PyG file not found: {pyg_file}")
            skipped_pathways += 1
            continue
        
        # Skip if required attributes are missing
        if not hasattr(data, 'node_types') or not hasattr(data, 'gene_rules'):
            print(f"  Skipping {pathway_name}: Missing required attributes")
            skipped_pathways += 1
            continue
            
        if data.num_nodes == 0:
            print(f"  Skipping {pathway_name}: Graph contains no nodes")
            skipped_pathways += 1
            continue
        
        # Identify reactions with no measured genes
        nodes_to_keep = []
        reactions_without_measured_genes = []
        reaction_count = 0
        reaction_with_measured_gene_count = 0
        
        for i in range(data.num_nodes):
            keep_node = True
            
            if data.node_types[i] == 'reaction':
                reaction_count += 1
                reaction_id = data.node_ids[i] if hasattr(data, 'node_ids') else f"reaction_{i}"
                gene_rule = data.gene_rules[i]
                genes = parse_gene_rule(gene_rule)
                
                # Check if any gene is measured
                measured_count = 0
                if len(genes) > 0:
                    measured_count = sum(1 for gene in genes if gene in gene_to_expression)
                    if measured_count == 0:
                        # This reaction has genes but none are measured - remove it
                        keep_node = False
                        reactions_without_measured_genes.append((reaction_id, gene_rule))
                        reactions_removed_count += 1
                    else:
                        reaction_with_measured_gene_count += 1
                        reactions_kept_count += 1
                # Also handle reactions with no genes - they should be removed
                elif len(genes) == 0:
                    keep_node = False
                    reactions_without_measured_genes.append((reaction_id, "No genes in rule"))
                    reactions_removed_count += 1
            
            nodes_to_keep.append(keep_node)
        
        # Skip if all reactions would be removed
        if reaction_count > 0 and reaction_with_measured_gene_count == 0:
            print(f"  Skipping {pathway_name}: All {reaction_count} reactions have unmeasured genes")
            skipped_pathways += 1
            pathways_with_no_reactions += 1
            continue
        
        # If there are reactions to remove, create a new graph without them
        if not all(nodes_to_keep):
            # Get indices of nodes to keep
            keep_indices = torch.tensor([i for i, keep in enumerate(nodes_to_keep) if keep])
            
            # Count how many reactions and metabolites are being kept/removed
            reactions_kept = sum(1 for i, keep in enumerate(nodes_to_keep) 
                                if keep and data.node_types[i] == 'reaction')
            metabolites_kept = sum(1 for i, keep in enumerate(nodes_to_keep) 
                                  if keep and data.node_types[i] == 'metabolite')
            reactions_removed = sum(1 for i, keep in enumerate(nodes_to_keep) 
                                   if not keep and data.node_types[i] == 'reaction')
            
            print(f"  Graph modification for {pathway_name}:")
            print(f"    - Keeping {reactions_kept} reactions with measured genes")
            print(f"    - Keeping {metabolites_kept} metabolites")
            print(f"    - Removing {reactions_removed} reactions without measured genes")
            
            # Create mapping from old indices to new indices
            old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(keep_indices)}
            
            # Filter edge_index to keep only edges between remaining nodes
            mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
            edges_kept = 0
            edges_removed = 0
            
            for i in range(data.edge_index.size(1)):
                src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
                if src in old_to_new and dst in old_to_new:
                    mask[i] = True
                    edges_kept += 1
                else:
                    edges_removed += 1
            
            print(f"    - Keeping {edges_kept} edges between remaining nodes")
            print(f"    - Removing {edges_removed} edges connected to removed nodes")
            
            # Update edge_index with new node indices
            new_edge_index = data.edge_index[:, mask].clone()
            for i in range(new_edge_index.size(1)):
                new_edge_index[0, i] = old_to_new[new_edge_index[0, i].item()]
                new_edge_index[1, i] = old_to_new[new_edge_index[1, i].item()]
            
            # Update all node-related attributes
            new_node_types = [data.node_types[i] for i, keep in enumerate(nodes_to_keep) if keep]
            
            # Transfer other node attributes
            new_data = Data(edge_index=new_edge_index)
            new_data.node_types = new_node_types
            
            if hasattr(data, 'node_ids'):
                new_data.node_ids = [data.node_ids[i] for i, keep in enumerate(nodes_to_keep) if keep]
            
            if hasattr(data, 'gene_rules'):
                new_data.gene_rules = [data.gene_rules[i] for i, keep in enumerate(nodes_to_keep) if keep]
            
            # Transfer edge attributes if they exist (must be filtered by edge mask)
            edge_tensor_keys = set()
            if hasattr(data, 'edge_type'):
                new_data.edge_type = data.edge_type[mask]
                edge_tensor_keys.add('edge_type')
            if hasattr(data, 'edge_attr'):
                new_data.edge_attr = data.edge_attr[mask]
                edge_tensor_keys.add('edge_attr')
            if hasattr(data, 'edge_attr_raw'):
                new_data.edge_attr_raw = data.edge_attr_raw[mask]
                edge_tensor_keys.add('edge_attr_raw')
            if hasattr(data, 'is_reverse_edge'):
                new_data.is_reverse_edge = data.is_reverse_edge[mask]
                edge_tensor_keys.add('is_reverse_edge')

            # Other graph-level attributes (including existing x if present)
            skip_keys = {'edge_index', 'node_types', 'node_ids', 'gene_rules'} | edge_tensor_keys
            for key, value in data:
                if key not in skip_keys:
                    new_data[key] = value
            
            # Replace the original data with the filtered one
            data = new_data
            
            # Update the metadata node mapping to match the new node indices
            new_node_mapping = {}
            for node_id, old_idx in metadata.get('node_mapping', {}).items():
                if old_idx in old_to_new:
                    new_node_mapping[node_id] = old_to_new[old_idx]
            
            metadata['node_mapping'] = new_node_mapping
            

        
        # Initialize node features tensor for EC embeddings
        embedding_dim = len(next(iter(ec_embeddings_dict.values())))
        ec_embeddings = torch.zeros((data.num_nodes, embedding_dim), dtype=torch.float32)
        
        # Process each node for EC embeddings
        for node_id, node_idx in metadata.get('node_mapping', {}).items():
            # Skip nodes that were removed in filtering
            if not hasattr(data, 'node_ids') or node_id not in data.node_ids:
                continue
                
            # Find the new index if we've remapped nodes
            if hasattr(data, 'node_ids'):
                node_idx = data.node_ids.index(node_id)
            
            # Check if this is a reaction node (typically starts with 'MAR')
            if node_id.startswith('MAR'):
                # Get EC numbers for this reaction
                ec_numbers = []
                # Check if ec_numbers field exists in metadata
                if 'ec_numbers' in metadata:
                    # Find the original index of this node in the metadata
                    orig_node_index = metadata.get('original_node_mapping', {}).get(node_id, 
                                     metadata.get('node_mapping', {}).get(node_id))
                    
                    if orig_node_index is not None and orig_node_index < len(metadata['ec_numbers']):
                        ec_entry = metadata['ec_numbers'][orig_node_index]
                        if ec_entry:  # Skip empty entries
                            for ec in ec_entry.split(';'):
                                ec = ec.strip()
                                if ec:  # Only add non-empty strings
                                    ec_numbers.append(ec)
                
                # If EC numbers found, compute embedding
                if ec_numbers:
                    embeddings = []
                    for ec in ec_numbers:
                        if ec in ec_embeddings_dict:
                            embeddings.append(ec_embeddings_dict[ec])
                    
                    if embeddings:
                        # Average the embeddings if multiple EC numbers
                        avg_embedding = np.mean(embeddings, axis=0)
                        ec_embeddings[node_idx] = torch.tensor(avg_embedding, dtype=torch.float32)
        
        # Keep existing features if available and combine with EC embeddings
        if hasattr(data, 'x') and data.x is not None:
            # Combine existing features with EC embeddings
            combined_features = torch.cat([data.x, ec_embeddings], dim=1)
            data.x = combined_features
            data.feature_type = 'gene_expression_and_ec_embeddings'
        else:
            # Just use EC embeddings
            data.x = ec_embeddings
            data.feature_type = 'ec_embeddings'
        
        # Save the updated PyG Data object
        print(data)
        torch.save(data, output_pyg_file)
        
        # Save the updated metadata
        with open(f'{output_dir}/{pathway_name}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save the updated pickle file if it exists
        if os.path.exists(pkl_file):
            try:
                with open(pkl_file, 'rb') as f:
                    nx_graph = pickle.load(f)
                
                # Remove nodes that were filtered out
                if not all(nodes_to_keep):
                    nodes_to_remove = [data.node_ids[i] for i, keep in enumerate(nodes_to_keep) if not keep]
                    for node_id in nodes_to_remove:
                        if node_id in nx_graph:
                            nx_graph.remove_node(node_id)
                
                # Update the networkx graph with new node features
                for i, node_id in enumerate(data.node_ids):
                    if node_id in nx_graph:
                        # Add the embedding as a node attribute
                        nx_graph.nodes[node_id]['ec_embedding'] = ec_embeddings[i].numpy()
                        if 'gene_features' in nx_graph.nodes[node_id]:
                            combined = torch.cat([
                                torch.tensor(nx_graph.nodes[node_id]['gene_features']), 
                                ec_embeddings[i]
                            ]).numpy()
                            nx_graph.nodes[node_id]['combined_features'] = combined
                
                # Save the updated networkx graph
                with open(output_pkl_file, 'wb') as f:
                    pickle.dump(nx_graph, f)
            except Exception as e:
                print(f"  Error updating NetworkX graph: {e}")
    
    # Print summary statistics
    print("\nEC Embedding initialization complete!")
    print("Summary:")
    print(f"  Total pathways: {total_pathways}")
    print(f"  Processed pathways: {total_pathways - skipped_pathways}")
    print(f"  Skipped pathways: {skipped_pathways}")
    print(f"  Pathways with no valid reactions: {pathways_with_no_reactions}")
    print(f"  Reactions kept: {reactions_kept_count}")
    print(f"  Reactions removed: {reactions_removed_count}")


# parse_gene_rule is imported from src.utils.gene_parsing

if __name__ == '__main__':
    main()
