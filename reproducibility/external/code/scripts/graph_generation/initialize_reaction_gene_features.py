import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import pickle
import json

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.gene_parsing import parse_gene_rule_simple as parse_gene_rule

def initialize_reaction_gene_features():
    """
    Initialize reaction gene features for all graphs in the dataset.
    This replaces the EC number embeddings with mean gene expression features
    for the reactions that have associated genes.
    """
    source_data_path = '/home/cnovellarausell/data1/Metabolomics-RNA/source-data'
    output_base_path = '/home/cnovellarausell/data1/concat-smiles-model/data/graphs'
    
    input_dir = f'{output_base_path}/PyG_Graphs_with_SMILES'
    output_dir = f'{output_base_path}/PyG_Graphs_with_SMILES_with_gene_features'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the mudata file to get the gene expression data
    import mudata as mu
    mdata = mu.read_h5mu(f'{source_data_path}/mudata_correct_HMDB_new_no_negatives.h5mu')
    gene_expression = mdata['rna']
    
    # Create mapping from gene ID to expression values
    gene_to_expression = {}
    for i, gene_id in enumerate(gene_expression.var['ensembl_gene_id']):
        # Get mean expression across all samples for this gene
        mean_expr = gene_expression.X[:, i].mean()
        gene_to_expression[gene_id] = mean_expr
    
    print(f"Loaded expression data for {len(gene_to_expression)} genes")
    
    # Get all PyG files from input directory
    pyg_files = glob.glob(os.path.join(input_dir, '*.pt'))
    print(f"Found {len(pyg_files)} graph files to process")
    
    # First pass - collect all unique MEASURED genes across all pathways
    measured_genes = set()
    skipped_graphs = []
    reactions_missing_all_genes = []
    
    for pyg_file in pyg_files:
        pathway_name = os.path.basename(pyg_file).replace('.pt', '')
        
        try:
            data = torch.load(pyg_file, weights_only=False)
            
            # Skip empty graphs or those without required attributes
            if not hasattr(data, 'node_types'):
                print(f"  Skipping {pathway_name}: Missing 'node_types' attribute required for processing")
                skipped_graphs.append((pathway_name, "Missing 'node_types' attribute"))
                continue
                
            if not hasattr(data, 'gene_rules'):
                print(f"  Skipping {pathway_name}: Missing 'gene_rules' attribute required for gene analysis")
                skipped_graphs.append((pathway_name, "Missing 'gene_rules' attribute"))
                continue
                
            if data.num_nodes == 0:
                print(f"  Skipping {pathway_name}: Graph contains no nodes")
                skipped_graphs.append((pathway_name, "Empty graph (0 nodes)"))
                continue
            
            # Collect genes from all reactions
            for i in range(data.num_nodes):
                if data.node_types[i] == 'reaction':
                    reaction_id = data.node_ids[i] if hasattr(data, 'node_ids') else f"reaction_{i}"
                    gene_rule = data.gene_rules[i]
                    genes = parse_gene_rule(gene_rule)
                    
                    # Count measured genes for this reaction
                    measured_count = 0
                    for gene in genes:
                        if gene in gene_to_expression:
                            measured_genes.add(gene)
                            measured_count += 1
                    
                    # Check if none of the genes were measured
                    if len(genes) > 0 and measured_count == 0:
                        reactions_missing_all_genes.append((pathway_name, reaction_id, gene_rule))
        
        except Exception as e:
            print(f"  Skipping {pathway_name}: Failed to load graph - {str(e)}")
            skipped_graphs.append((pathway_name, f"Error loading graph: {str(e)}"))
            continue
    
    # Create mapping from gene to index (only for measured genes)
    gene_to_index = {gene: i for i, gene in enumerate(sorted(measured_genes))}
    index_to_gene = {i: gene for gene, i in gene_to_index.items()}
    
    total_genes = len(measured_genes)
    print(f"Total unique measured genes across all pathways: {total_genes}")
    print(f"Found {len(reactions_missing_all_genes)} reactions with no measured genes")
    
    if total_genes == 0:
        print("Error: No measured genes found in any reaction. Check gene rule parsing or expression data.")
        return
    
    # Save gene mapping for reference
    with open(os.path.join(output_dir, 'gene_mapping.json'), 'w') as f:
        json.dump({
            'gene_to_index': gene_to_index,
            'index_to_gene': index_to_gene
        }, f, indent=2)
    
    # Save report on reactions missing all genes
    with open(os.path.join(output_dir, 'reactions_missing_all_genes.json'), 'w') as f:
        json.dump(reactions_missing_all_genes, f, indent=2)
    
    # Second pass - update graphs with measured gene expression features
    processed_graphs = []
    reactions_removed_count = 0
    skipped_pathway_details = {}  # Track details for each skipped pathway
    
    for pyg_file in pyg_files:
        pathway_name = os.path.basename(pyg_file).replace('.pt', '')
        print(f"Processing {pathway_name}")
        
        # Check if this pathway was skipped in the first pass
        if any(path == pathway_name for path, _ in skipped_graphs):
            reason = next(reason for path, reason in skipped_graphs if path == pathway_name)
            print(f"  Skipping {pathway_name}: Previously identified issue - {reason}")
            continue
        
        try:
            data = torch.load(pyg_file, weights_only=False)
            
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
                    # Important: Also handle reactions with no genes - they should be removed
                    elif len(genes) == 0:
                        keep_node = False
                        reactions_without_measured_genes.append((reaction_id, "No genes in rule"))
                        reactions_removed_count += 1
                
                nodes_to_keep.append(keep_node)
            
            # Skip if all reactions would be removed
            if reaction_count > 0 and reaction_with_measured_gene_count == 0:
                skip_reason = f"All {reaction_count} reactions have unmeasured genes"
                print(f"  Skipping {pathway_name}: {skip_reason}")
                skipped_graphs.append((pathway_name, skip_reason))
                skipped_pathway_details[pathway_name] = {
                    "reaction_count": reaction_count,
                    "reactions_without_gene_data": [{"id": rxn[0], "rule": rxn[1]} for rxn in reactions_without_measured_genes],
                    "reason": skip_reason
                }
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
                print(f"    - Keeping {reactions_kept} reactions with at least one measured gene")
                print(f"    - Keeping {metabolites_kept} metabolites")
                print(f"    - Removing {reactions_removed} reactions with no measured genes")
                
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

                # Other graph-level attributes
                skip_keys = {'edge_index', 'node_types', 'node_ids', 'gene_rules', 'x'} | edge_tensor_keys
                for key, value in data:
                    if key not in skip_keys:
                        new_data[key] = value
                
                # Replace the original data with the filtered one
                data = new_data
                
                # Save information about removed reactions
                with open(os.path.join(output_dir, f"{pathway_name}_removed_reactions.json"), 'w') as f:
                    json.dump({
                        "pathway": pathway_name,
                        "reactions_removed": len(reactions_without_measured_genes),
                        "reactions_kept": reactions_kept,
                        "removed_reaction_details": reactions_without_measured_genes
                    }, f, indent=2)
            
            # Create new feature matrix - for each node:
            # - If metabolite: zero vector
            # - If reaction: has expression values at positions of its measured genes
            x = torch.zeros((data.num_nodes, total_genes), dtype=torch.float32)
            
            # Track which genes are found/used for each reaction (for reporting)
            reaction_genes_info = []
            
            for i in range(data.num_nodes):
                if data.node_types[i] == 'reaction':
                    reaction_id = data.node_ids[i] if hasattr(data, 'node_ids') else f"reaction_{i}"
                    gene_rule = data.gene_rules[i]
                    genes = parse_gene_rule(gene_rule)
                    
                    measured_genes_used = []
                    unmeasured_genes = []
                    measured_count = 0
                    
                    # Fill in expression values for genes that are measured
                    for gene_id in genes:
                        if gene_id in gene_to_expression:
                            gene_idx = gene_to_index[gene_id]
                            x[i, gene_idx] = gene_to_expression[gene_id]
                            measured_genes_used.append(gene_id)
                            measured_count += 1
                        else:
                            unmeasured_genes.append(gene_id)
                    
                    # Store gene info for this reaction
                    reaction_genes_info.append({
                        'reaction_id': reaction_id,
                        'gene_rule': gene_rule,
                        'total_genes': len(genes),
                        'measured_genes': measured_count,
                        'measured_gene_ids': measured_genes_used,
                        'unmeasured_gene_ids': unmeasured_genes
                    })
                    
                    # Handle the error more gracefully instead of crashing
                    if measured_count == 0:
                        print(f"  Warning: Reaction {reaction_id} has no measured genes despite filtering")
                        # Add to the list of problematic reactions to debug later
                        with open(os.path.join(output_dir, 'problematic_reactions.txt'), 'a') as f:
                            f.write(f"{pathway_name}, {reaction_id}, '{gene_rule}'\n")
            
            # Set the new feature matrix
            data.x = x
            
            # Add metadata about gene feature representation
            data.feature_type = 'measured_gene_expression'
            data.total_unique_measured_genes = total_genes
            
            # Save the updated graph
            torch.save(data, os.path.join(output_dir, f"{pathway_name}.pt"))
            
            # Save gene information as JSON for reference
            with open(os.path.join(output_dir, f"{pathway_name}_gene_info.json"), 'w') as f:
                json.dump(reaction_genes_info, f, indent=2)
            
            # Also update and save NetworkX graph if it exists
            nx_file = os.path.join(input_dir, f"{pathway_name}_nx.pkl")
            if os.path.exists(nx_file):
                with open(nx_file, 'rb') as f:
                    G = pickle.load(f)
                
                # Remove nodes that correspond to reactions with no measured genes
                if not all(nodes_to_keep):
                    # Get node IDs to remove
                    nodes_to_remove = [list(G.nodes())[i] for i, keep in enumerate(nodes_to_keep) if not keep]
                    for node_id in nodes_to_remove:
                        if node_id in G:
                            G.remove_node(node_id)
                
                # Update node features in the remaining nodes
                for i in range(data.num_nodes):
                    if i < len(G.nodes()):
                        node_id = list(G.nodes())[i]
                        G.nodes[node_id]['gene_features'] = x[i].tolist()
                
                # Save updated NetworkX graph
                with open(os.path.join(output_dir, f"{pathway_name}_nx.pkl"), 'wb') as f:
                    pickle.dump(G, f)
            
            print(f"  Successfully processed {pathway_name}: {len(reaction_genes_info)} reactions with measured genes")
            processed_graphs.append(pathway_name)
            
        except Exception as e:
            error_message = f"Error during processing: {str(e)}"
            print(f"  Skipping {pathway_name}: {error_message}")
            skipped_graphs.append((pathway_name, error_message))
            skipped_pathway_details[pathway_name] = {
                "error": str(e),
                "traceback": str(e.__traceback__)
            }
            continue
    
    # Save detailed information about skipped pathways
    with open(os.path.join(output_dir, 'skipped_pathways_details.json'), 'w') as f:
        json.dump(skipped_pathway_details, f, indent=2)
    
    # Generate and save summary statistics for genes
    gene_stats = {
        'gene_id': [],
        'expression_value': [],
        'in_pathways_count': [],
        'in_reactions_count': []
    }
    
    # Track counts for each gene
    gene_pathway_counts = {gene: 0 for gene in measured_genes}
    gene_reaction_counts = {gene: 0 for gene in measured_genes}
    
    # Reprocess all graphs to count gene usage
    for pyg_file in glob.glob(os.path.join(output_dir, '*.pt')):
        if pyg_file.endswith('_gene_info.json'):
            continue
        
        pathway_name = os.path.basename(pyg_file).replace('.pt', '')
        gene_info_file = os.path.join(output_dir, f"{pathway_name}_gene_info.json")
        
        if os.path.exists(gene_info_file):
            with open(gene_info_file, 'r') as f:
                reaction_info = json.load(f)
                
                # Track genes found in this pathway
                pathway_genes = set()
                
                for reaction in reaction_info:
                    for gene in reaction['measured_gene_ids']:
                        pathway_genes.add(gene)
                        gene_reaction_counts[gene] += 1
                
                # Update pathway counts
                for gene in pathway_genes:
                    gene_pathway_counts[gene] += 1
    
    # Create gene statistics dataframe
    for gene in measured_genes:
        gene_stats['gene_id'].append(gene)
        gene_stats['expression_value'].append(gene_to_expression[gene])
        gene_stats['in_pathways_count'].append(gene_pathway_counts[gene])
        gene_stats['in_reactions_count'].append(gene_reaction_counts[gene])
    
    # Save gene statistics
    gene_stats_df = pd.DataFrame(gene_stats)
    gene_stats_df.to_csv(os.path.join(output_dir, 'gene_usage_stats.csv'), index=False)
    
    # Overall statistics
    stats = {
        'total_graphs': len(pyg_files),
        'processed_graphs': len(processed_graphs),
        'skipped_graphs': len(skipped_graphs),
        'total_unique_measured_genes': total_genes,
        'reactions_missing_all_genes': len(reactions_missing_all_genes),
        'reactions_removed_count': reactions_removed_count,
        'top_used_genes': gene_stats_df.sort_values('in_reactions_count', ascending=False)['gene_id'].head(10).tolist()
    }
    
    # Create a more comprehensive skipped graphs report
    skipped_reasons = {}
    for pathway, reason in skipped_graphs:
        if reason not in skipped_reasons:
            skipped_reasons[reason] = []
        skipped_reasons[reason].append(pathway)
    
    # Save statistics
    with open(os.path.join(output_dir, 'gene_features_stats.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        
        # Add list of skipped graphs with grouped reasons
        f.write("\nSkipped Pathways by Reason:\n")
        for reason, pathways in skipped_reasons.items():
            f.write(f"  Reason: {reason}\n")
            f.write(f"  Count: {len(pathways)}\n")
            f.write(f"  Pathways: {', '.join(pathways[:5])}")
            if len(pathways) > 5:
                f.write(f" and {len(pathways) - 5} more")
            f.write("\n\n")
    
    print("\nGene feature initialization complete!")
    print("Summary:")
    for key, value in stats.items():
        if key != 'top_used_genes':
            print(f"  {key}: {value}")
    
    print("\nTop reasons for skipping pathways:")
    for reason, pathways in sorted(skipped_reasons.items(), key=lambda x: len(x[1]), reverse=True)[:3]:
        print(f"  - {reason}: {len(pathways)} pathways")
    
    print("\nTop 10 most used genes:")
    for i, gene in enumerate(stats['top_used_genes']):
        count = gene_reaction_counts[gene]
        pathways = gene_pathway_counts[gene]
        print(f"  {i+1}. {gene}: Used in {count} reactions across {pathways} pathways")
    
    return stats


# parse_gene_rule is imported from src.utils.gene_parsing

if __name__ == "__main__":
    initialize_reaction_gene_features()