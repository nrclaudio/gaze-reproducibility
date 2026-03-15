import requests
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
import time
import pandas as pd
from requests.exceptions import RequestException, Timeout
import csv
import numpy as np
import seaborn as sns
from collections import Counter
import re
import mudata as mu
import pickle  # Add this import

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.gene_parsing import parse_gene_rule_simple as parse_gene_rule

# Define the base URL for the API
base_url = "https://metabolicatlas.org/api/v2"

# Global variables for rate limiting
RATE_LIMIT = 5  # requests per second
LAST_REQUEST_TIME = 0

# Function to get the list of subsystems
def get_subsystems(model="HumanGem"):
    endpoint = f"{base_url}/maps/listing?model={model}"
    response = requests.get(endpoint)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to retrieve subsystems: {response.status_code}")

# Function to get the details of a specific subsystem
def get_subsystem_details(subsystem_id, model="HumanGem"):
    endpoint = f"{base_url}/subsystems/{subsystem_id}?model={model}"
    response = requests.get(endpoint)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to retrieve subsystem details: {response.status_code}")
    
    

def get_external_ids(metabolite, metabolites_df):
    metabolite_row = metabolites_df.loc[metabolites_df['mets'] == metabolite['id']]
    if metabolite_row.empty:
        return metabolite['id'], None, None  # Return the metabolite ID if not found in the table
    else:
        hmdb_id = metabolite_row['metHMDBID'].values[0] if not pd.isna(metabolite_row['metHMDBID'].values[0]) else None
        kegg_id = metabolite_row['metKEGGID'].values[0] if not pd.isna(metabolite_row['metKEGGID'].values[0]) else None
        return metabolite['id'], hmdb_id, kegg_id

def get_kcat_values(ec_number, kegg_id, organism='hsa'):
    if not kegg_id:
        return None
    url = f"https://metabolicatlas.org/api/v2/gotenzymes/enzymes?filters%5Bec_number%5D={ec_number}&filters%5Bcompound%5D={kegg_id}&filters%5Borganism%5D={organism}&pagination%5BisAscending%5D=true"
    response = requests.get(url)
    if response.status_code == 200:
        data_kcat = response.json()
        kcat_values = [enzyme['kcat_values'] for enzyme in data_kcat['enzymes'] if enzyme.get('kcat_values') is not None]
        return sum(kcat_values) / len(kcat_values) if kcat_values else None
    else:
        print(f"Failed to retrieve kcat values: {response.status_code}")
        return None

# Function to get reactions related to a specific subsystem
def get_subsystem_reactions(subsystem_id, model="HumanGem"):
    endpoint = f"{base_url}/subsystems/{subsystem_id}/related-reactions?model={model}"
    response = requests.get(endpoint)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to retrieve subsystem reactions: {response.status_code}")


# Function to build a graph from subsystem details, focusing on genes and metabolites
def build_graph(reactions, metabolites_df, measured_metabolites_hmdb, measured_metabolites_kegg, measured_gene_ids, ignore_hubs=True):
    G = nx.DiGraph()
    
    # Hub metabolites to ignore if ignore_hubs is True
    blacklist_ids = {
        "HMDB0000538", "HMDB0001341", "HMDB0000045", "HMDB0001273", "HMDB0001201", "HMDB0001397",
        "HMDB0000082", "HMDB0000300", "HMDB0000095", "HMDB0240312", "HMDB0000285", "HMDB0000295", 
        "HMDB0000288", "HMDB0001429", "HMDB0000250", "HMDB0000902", "HMDB0001487", "HMDB0000217", 
        "HMDB0304435", "HMDB0000221", "HMDB0001248", "HMDB0002111", "HMDB0001967", "HMDB0000051", 
        "HMDB0041827", "HMDB0001423", "HMDB0001448"
    }
    blacklist_names = {
        "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "CTP", "CDP", "CMP", "UTP", "UDP", "UMP",
        "PPI", "PI", "PHOSPHATE", "PYROPHOSPHATE", "NAD+", "NADH", "NADP+", "NADPH", 
        "FAD", "FADH2", "H2O2", "O2", "H+", "H2O", "CO2", "NH3", "SULFATE", "COA"
    }
    whitelist_ids = {
        "HMDB0001206", "HMDB0000062", "HMDB0001175", "HMDB0000148", "HMDB0000208", 
        "HMDB0000243", "HMDB0000190", "HMDB0000094", "HMDB0000156", "HMDB0000254", 
        "HMDB0000134", "HMDB0001185", "HMDB0000939"
    }
    
    # Track statistics for filtering
    total_reactions = 0
    skipped_no_ec = 0
    skipped_no_gene_rule = 0
    skipped_no_measured_genes = 0
    skipped_insufficient_edges = 0
    
    for reaction in reactions:
        total_reactions += 1
        reaction_id = reaction['id']
        ec = reaction.get('ec', '')
        gene_rule = reaction.get('geneRule', '')
        
        # Skip reactions without EC numbers
        if not ec:
            print(f"Reaction {reaction_id} has no EC number")
            skipped_no_ec += 1
            continue
            
        # Skip reactions without gene rules
        if not gene_rule:
            print(f"Reaction {reaction_id} has no gene rule")
            skipped_no_gene_rule += 1
            continue
            
        # Parse gene IDs from gene rule
        gene_ids = parse_gene_rule(gene_rule)
        
        # Check which genes are measured
        measured_genes = check_measured_genes(gene_ids, measured_gene_ids)
        
        # Skip reactions with no measured genes
        if not measured_genes:
            print(f"Reaction {reaction_id} has no measured genes")
            skipped_no_measured_genes += 1
            continue
        
        incoming_edges = 0
        outgoing_edges = 0
        edges_to_add = []
        
        for metabolite in reaction['metabolites']:
            metabolite_id = metabolite['id']
            # Get external IDs for identification
            met_id, hmdb_id, kegg_id = get_external_ids(metabolite, metabolites_df)
            
            # Hub Filtering Logic
            if ignore_hubs:
                is_hub = False
                if hmdb_id in blacklist_ids: is_hub = True
                if metabolite['name'].upper() in blacklist_names: is_hub = True
                
                # Protect key carbon metabolites (whitelist)
                if hmdb_id in whitelist_ids: is_hub = False
                
                if is_hub:
                    continue

            if 'pool' in metabolite['name']:
                continue
            if '[protein]' in metabolite['name']:
                continue
            if 'antigen' in metabolite['name']:
                continue
            
            if not hmdb_id and not kegg_id:
                continue

            # Check if this metabolite is measured in the dataset
            is_measured = (hmdb_id in measured_metabolites_hmdb) or (kegg_id in measured_metabolites_kegg)
            
            # Extract stoichiometry coefficient from API data
            stoichiometry = metabolite.get('stoichiometry', 1)  # Default to 1 if missing
            
            # Validate stoichiometry value (API should provide positive values)
            if not isinstance(stoichiometry, (int, float)) or stoichiometry == 0:
                print(f"Warning: Invalid stoichiometry {stoichiometry} for {metabolite.get('name')}, defaulting to 1")
                stoichiometry = 1
            
            # Store BOTH raw and tanh-squashed stoichiometric coefficients.
            # Raw coefficients preserve exact ratios for the physics mass-balance loss (Sv = 0).
            # Tanh-squashed coefficients stay bounded in [-1, 1] for stable GNN message passing.
            if metabolite['outgoing']:
                # Metabolite -> Reaction (reactant, negative stoichiometry)
                raw_stoich = -stoichiometry
                squashed_stoich = float(np.tanh(raw_stoich))
                edges_to_add.append((metabolite_id, reaction_id, raw_stoich, squashed_stoich))
                incoming_edges += 1
            else:
                # Reaction -> Metabolite (product, positive stoichiometry)
                raw_stoich = stoichiometry
                squashed_stoich = float(np.tanh(raw_stoich))
                edges_to_add.append((reaction_id, metabolite_id, raw_stoich, squashed_stoich))
                outgoing_edges += 1
        
        # Only add the reaction and its edges if it has at least one incoming and one outgoing edge
        if incoming_edges > 0 and outgoing_edges > 0:
            # Add reaction node
            G.add_node(reaction_id, id=reaction_id, type='reaction', name=reaction['id'], 
                       ec=ec, reversible=reaction.get('reversible', False),
                       gene_rule=gene_rule, gene_ids=gene_ids, 
                       measured_genes=measured_genes,
                       is_measured=len(measured_genes) > 0)
            
            # Add metabolite nodes and edges
            # Store edges with both raw and squashed stoichiometry
            for edge in edges_to_add:
                source, target, raw_stoich, squashed_stoich = edge
                if source not in G:
                    _, hmdb_id, kegg_id = get_external_ids({'id': source}, metabolites_df)
                    is_measured = (hmdb_id in measured_metabolites_hmdb) or (kegg_id in measured_metabolites_kegg)
                    G.add_node(source, id=source, hmdb_id=hmdb_id, kegg_id=kegg_id, type='metabolite',
                               name=next((m['name'] for m in reaction['metabolites'] if m['id'] == source), ''),
                               is_currency=next((m['isCurrency'] for m in reaction['metabolites'] if m['id'] == source), False),
                               is_measured=is_measured)
                if target not in G:
                    _, hmdb_id, kegg_id = get_external_ids({'id': target}, metabolites_df)
                    is_measured = (hmdb_id in measured_metabolites_hmdb) or (kegg_id in measured_metabolites_kegg)
                    G.add_node(target, id=target, hmdb_id=hmdb_id, kegg_id=kegg_id, type='metabolite',
                               name=next((m['name'] for m in reaction['metabolites'] if m['id'] == target), ''),
                               is_currency=next((m['isCurrency'] for m in reaction['metabolites'] if m['id'] == target), False),
                               is_measured=is_measured)

                # Store both raw and squashed stoichiometry, plus is_reverse_edge flag
                if ec and kegg_id:
                    kcat = get_kcat_values(ec, kegg_id)
                    G.add_edge(source, target, weight=squashed_stoich,
                               raw_stoichiometry=float(raw_stoich),
                               is_reverse_edge=False, kcat=kcat)
                else:
                    G.add_edge(source, target, weight=squashed_stoich,
                               raw_stoichiometry=float(raw_stoich),
                               is_reverse_edge=False, kcat=None)

            # If the reaction is reversible, add edges in both directions
            # Reverse edges get negated stoichiometry and is_reverse_edge=True
            if reaction.get('reversible', False):
                for edge in edges_to_add:
                    source, target, raw_stoich, squashed_stoich = edge
                    # Add reverse edge with opposite stoichiometry sign
                    original_edge_data = G[source][target]
                    G.add_edge(target, source, weight=float(-squashed_stoich),
                               raw_stoichiometry=float(-raw_stoich),
                               is_reverse_edge=True,
                               kcat=original_edge_data.get('kcat'))
        else:
            skipped_insufficient_edges += 1
    
    # Print filtering statistics
    print(f"Reaction filtering stats:")
    print(f"  Total reactions: {total_reactions}")
    print(f"  Skipped (no EC number): {skipped_no_ec}")
    print(f"  Skipped (no gene rule): {skipped_no_gene_rule}")
    print(f"  Skipped (no measured genes): {skipped_no_measured_genes}")
    print(f"  Skipped (insufficient edges): {skipped_insufficient_edges}")
    print(f"  Included in graph: {len([n for n in G.nodes if G.nodes[n]['type'] == 'reaction'])}")
    
    return G

def plot_graph(G, title):
    plt.figure(figsize=(20, 16))
    
    if len(G) == 0:
        plt.text(0.5, 0.5, "Empty Graph", ha='center', va='center', fontsize=20)
        plt.title(title, fontsize=16)
        plt.axis('off')
    else:
        try:
            # Create a copy of the graph without weights for layout purposes
            G_layout = G.copy()
            for u, v, d in G_layout.edges(data=True):
                d.clear()  # Remove all edge attributes, including weight

            pos = nx.spring_layout(G_layout, k=0.5, iterations=50)
            
            # Prepare node labels
            node_labels = {}
            for node, data in G.nodes(data=True):
                if data['type'] == 'metabolite':
                    node_labels[node] = data['name']
                else:  # reaction
                    node_labels[node] = node  # Keep reaction IDs as is
            
            # Draw nodes
            metabolite_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'metabolite']
            reaction_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'reaction']
            
            nx.draw_networkx_nodes(G, pos, nodelist=metabolite_nodes, node_color="lightblue", node_size=3000, alpha=0.8)
            nx.draw_networkx_nodes(G, pos, nodelist=reaction_nodes, node_color="lightgreen", node_size=2000, alpha=0.8)
            
            # Draw edges (without considering weights)
            nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, node_labels, font_size=8, font_weight="bold")
            
            plt.title(title, fontsize=16)
            plt.axis('off')
        
        except Exception as e:
            print(f"Error plotting graph for {title}: {str(e)}")
            plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=12, wrap=True)
            plt.title(title, fontsize=16)
            plt.axis('off')
    
    # Ensure the results directory exists
    results_dir = "/home/cnovellarausell/data1/Metabolomics-RNA/source-data/Graphs_identified"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the plot
    plt.savefig(os.path.join(results_dir, f"{title}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Function to save the graph for PyTorch Geometric
def save_graph_for_pyg_to_path(G, subsystem_id, results_dir):
    import torch
    import numpy as np
    from torch_geometric.data import Data
    
    # Create node mapping (original ID to index)
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    
    # Create edge index tensor (2 x num_edges)
    edge_index = []
    edge_weights_squashed = []
    edge_weights_raw = []
    edge_is_reverse = []
    for source, target, data in G.edges(data=True):
        edge_index.append([node_mapping[source], node_mapping[target]])
        # Squashed weight for GNN message passing
        weight = data.get('weight', 1.0)
        if weight is None:
            weight = 1.0
        edge_weights_squashed.append(weight)
        # Raw stoichiometry for physics loss incidence matrix
        raw = data.get('raw_stoichiometry', weight)
        if raw is None:
            raw = weight
        edge_weights_raw.append(raw)
        # Reverse edge flag
        edge_is_reverse.append(data.get('is_reverse_edge', False))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # edge_attr stores squashed weights (used by GNN message passing)
    edge_attr = torch.tensor(edge_weights_squashed, dtype=torch.float).view(-1, 1)
    # edge_attr_raw stores raw stoichiometry (used by physics loss)
    edge_attr_raw = torch.tensor(edge_weights_raw, dtype=torch.float).view(-1, 1)
    # Boolean tensor marking reverse edges
    is_reverse_edge = torch.tensor(edge_is_reverse, dtype=torch.bool)
    
    # Create node features and collect metadata
    x = []
    hmdb_ids = []
    kegg_ids = []
    node_types = []
    node_names = []
    node_ids = []
    measured_flags = []
    gene_rules = []
    reversible_flags = []
    ec_numbers = []
    
    for node, data in G.nodes(data=True):
        node_ids.append(data.get('id', ''))
        node_names.append(data.get('name', ''))
        is_metabolite = 1 if data['type'] == 'metabolite' else 0
        is_reaction = 1 if data['type'] == 'reaction' else 0
        is_measured = 1 if data.get('is_measured', False) else 0
        
        # Features: [is_metabolite, is_reaction, is_measured]
        feature = [is_metabolite, is_reaction, is_measured]
        x.append(feature)
        
        node_types.append(data['type'])
        measured_flags.append(is_measured)
        
        # Store metabolite-specific data
        if data['type'] == 'metabolite':
            hmdb_ids.append(data.get('hmdb_id', ''))
            kegg_ids.append(data.get('kegg_id', ''))
            gene_rules.append('')  # Empty for metabolites
            reversible_flags.append(0)  # Not applicable
            ec_numbers.append('')  # Not applicable
        # Store reaction-specific data
        else:  # reaction
            hmdb_ids.append('')  # Not applicable
            kegg_ids.append('')  # Not applicable
            gene_rules.append(data.get('gene_rule', ''))
            reversible_flags.append(1 if data.get('reversible', False) else 0)
            ec_numbers.append(data.get('ec', ''))
    
    # Convert to tensor
    x = torch.tensor(x, dtype=torch.float)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Add raw stoichiometry and reverse edge flag
    data.edge_attr_raw = edge_attr_raw
    data.is_reverse_edge = is_reverse_edge

    # Add metadata as attributes to the Data object
    data.node_ids = node_ids
    data.node_names = node_names
    data.node_types = node_types
    data.hmdb_ids = hmdb_ids
    data.kegg_ids = kegg_ids
    data.gene_rules = gene_rules
    data.measured_flags = measured_flags
    data.reversible_flags = reversible_flags
    data.ec_numbers = ec_numbers
    data.subsystem_id = subsystem_id
    
    # Ensure the results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Save the PyG Data object
    torch.save(data, os.path.join(results_dir, f"{subsystem_id}.pt"))
    
    # Save NetworkX graph for complete reference
    with open(os.path.join(results_dir, f"{subsystem_id}_nx.pkl"), 'wb') as f:
        pickle.dump(G, f)
    
    # Save metadata as JSON for easier inspection
    import json
    with open(os.path.join(results_dir, f"{subsystem_id}_metadata.json"), 'w') as f:
        # Convert node_mapping keys to strings since JSON doesn't support non-string keys
        str_node_mapping = {str(k): v for k, v in node_mapping.items()}
        json_metadata = {
            'node_mapping': str_node_mapping,
            'node_ids': node_ids,
            'node_names': node_names,
            'node_types': node_types,
            'hmdb_ids': [str(id) if id is not None else "" for id in hmdb_ids],
            'kegg_ids': [str(id) if id is not None else "" for id in kegg_ids],
            'gene_rules': gene_rules,
            'measured_flags': measured_flags,
            'reversible_flags': reversible_flags,
            'ec_numbers': ec_numbers,
            'subsystem_id': subsystem_id
        }
        json.dump(json_metadata, f, indent=2)
    
    return data


# New function to get metabolites without HMDB or KEGG IDs
def get_metabolites_without_ids(G):
    return [node for node, data in G.nodes(data=True) 
            if data['type'] == 'metabolite' and (not data['hmdb_id'] or not data['kegg_id'])]

# New function to calculate percentage of metabolites without IDs
def calculate_percentage_without_ids(G):
    metabolites = [node for node, data in G.nodes(data=True) if data['type'] == 'metabolite']
    without_ids = get_metabolites_without_ids(G)
    return len(without_ids) / len(metabolites) * 100 if metabolites else 0

# New function to check if a gene rule contains only OR operations
def is_only_or_operation(gene_rule):
    # Remove spaces and parentheses
    gene_rule = gene_rule.replace(" ", "").replace("(", "").replace(")", "")
    # Check if there are any AND operations
    return "and" not in gene_rule.lower()


# parse_gene_rule is imported from src.utils.gene_parsing

# New function to check which genes are measured
def check_measured_genes(gene_ids, measured_gene_ids):
    return [gene_id for gene_id in gene_ids if gene_id in measured_gene_ids]

# New function to save matched metabolites
def save_matched_metabolites(G, subsystem_id):
    matched_metabolites = []
    for node, data in G.nodes(data=True):
        if data['type'] == 'metabolite' and data.get('is_measured', False):
            matched_metabolites.append({
                'subsystem_id': subsystem_id,
                'name': data['name'],
                'id': data['id'],
                'hmdb_id': data['hmdb_id'],
                'kegg_id': data['kegg_id']
            })
    
    return matched_metabolites

# New function to save matched genes
def save_matched_genes(G, subsystem_id):
    matched_genes = []
    for node, data in G.nodes(data=True):
        if data['type'] == 'reaction':
            measured_genes = data.get('measured_genes', [])
            for gene_id in measured_genes:
                matched_genes.append({
                    'subsystem_id': subsystem_id,
                    'reaction_id': node,
                    'gene_id': gene_id
                })
    
    return matched_genes

# New function to calculate coverage statistics
def calculate_coverage_stats(G, subsystem_id):
    total_metabolites = 0
    measured_metabolites = 0
    total_reactions = 0
    reactions_with_measured_genes = 0
    total_genes = set()
    measured_genes = set()
    
    for node, data in G.nodes(data=True):
        if data['type'] == 'metabolite':
            total_metabolites += 1
            if data.get('is_measured', False):
                measured_metabolites += 1
        elif data['type'] == 'reaction':
            total_reactions += 1
            if data.get('is_measured', False):
                reactions_with_measured_genes += 1
            
            # Add genes to sets
            gene_ids = data.get('gene_ids', [])
            total_genes.update(gene_ids)
            measured_gene_ids = data.get('measured_genes', [])
            measured_genes.update(measured_gene_ids)
    
    return {
        'subsystem_id': subsystem_id,
        'total_metabolites': total_metabolites,
        'measured_metabolites': measured_metabolites,
        'metabolite_coverage': measured_metabolites / total_metabolites if total_metabolites > 0 else 0,
        'total_reactions': total_reactions,
        'reactions_with_measured_genes': reactions_with_measured_genes,
        'reaction_coverage': reactions_with_measured_genes / total_reactions if total_reactions > 0 else 0,
        'total_genes': len(total_genes),
        'measured_genes': len(measured_genes),
        'gene_coverage': len(measured_genes) / len(total_genes) if total_genes else 0
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build metabolic graphs with optional hub masking')
    parser.add_argument('--identifier', type=str, default='', help='Suffix for output directories (e.g. masked)')
    parser.add_argument('--no-masking', action='store_true', help='Disable hub masking')
    args = parser.parse_args()

    # Paths
    source_data_path = '/home/cnovellarausell/data1/Metabolomics-RNA/source-data'
    output_base_path = '/home/cnovellarausell/data1/concat-smiles-model/data/graphs'
    
    # Load measured metabolites from mudata
    mdata = mu.read_h5mu(f'{source_data_path}/mudata_correct_HMDB_new_no_negatives.h5mu')

    measured_metabolites_hmdb = set(mdata['metabolomics'].var['HMDB_trailing_zeros'])
    measured_metabolites_kegg = set(mdata['metabolomics'].var['KEGG'])
    measured_gene_ids = set(mdata['rna'].var['ensembl_gene_id'])
    
    metabolites_df = pd.read_csv(f'{source_data_path}/metabolites.tsv', sep='\t')
    subsystems = get_subsystems()
    
    suffix = f"_{args.identifier}" if args.identifier else ""
    pyg_results_dir = f"{output_base_path}/PyG_Graphs{suffix}"
    viz_results_dir = f"{output_base_path}/Graphs_identified{suffix}"
    
    os.makedirs(pyg_results_dir, exist_ok=True)
    os.makedirs(viz_results_dir, exist_ok=True)
    
    all_matched_metabolites = []
    all_matched_genes = []
    all_coverage_stats = []
    
    for subsystem in subsystems['subsystems']:
        subsystem_id = subsystem['id']
        if 'o_glycan' in subsystem_id:
            continue
        print(f"Processing {subsystem_id}...")
        subsystem_reactions = get_subsystem_reactions(subsystem_id)
    
        G = build_graph(subsystem_reactions, metabolites_df, measured_metabolites_hmdb, 
                        measured_metabolites_kegg, measured_gene_ids, ignore_hubs=not args.no_masking)
        
        if len(G.nodes) > 0:
            # We need to update save_graph_for_pyg to accept custom path
            save_graph_for_pyg_to_path(G, subsystem_id, pyg_results_dir)
            # plot_graph_to_path(G, subsystem_id, viz_results_dir)
            
            # Save matched metabolites and genes
            matched_metabolites = save_matched_metabolites(G, subsystem_id)
            all_matched_metabolites.extend(matched_metabolites)
            
            matched_genes = save_matched_genes(G, subsystem_id)
            all_matched_genes.extend(matched_genes)
            
            # Calculate and save coverage statistics
            coverage_stats = calculate_coverage_stats(G, subsystem_id)
            all_coverage_stats.append(coverage_stats)
    
    # Save results with identifier
    results_dir = pyg_results_dir # Save summary files inside the graph directory
    matched_metabolites_df = pd.DataFrame(all_matched_metabolites)
    matched_metabolites_df.to_csv(os.path.join(results_dir, f"all_matched_metabolites{suffix}.csv"), index=False)
    
    matched_genes_df = pd.DataFrame(all_matched_genes)
    matched_genes_df.to_csv(os.path.join(results_dir, f"all_matched_genes{suffix}.csv"), index=False)
    
    coverage_stats_df = pd.DataFrame(all_coverage_stats)
    coverage_stats_df.to_csv(os.path.join(results_dir, f"all_coverage_stats{suffix}.csv"), index=False)
    
    print(f"Finished. Graphs saved to {pyg_results_dir}")
    
    # Create summary statistics
    overall_metabolite_coverage = sum(stats['measured_metabolites'] for stats in all_coverage_stats) / sum(stats['total_metabolites'] for stats in all_coverage_stats) if sum(stats['total_metabolites'] for stats in all_coverage_stats) > 0 else 0
    overall_reaction_coverage = sum(stats['reactions_with_measured_genes'] for stats in all_coverage_stats) / sum(stats['total_reactions'] for stats in all_coverage_stats) if sum(stats['total_reactions'] for stats in all_coverage_stats) > 0 else 0
    
    # Get unique genes and metabolites across all subsystems
    unique_metabolites = set()
    for item in all_matched_metabolites:
        unique_metabolites.add((item['name'], item['hmdb_id'], item['kegg_id']))
    
    unique_genes = set()
    for item in all_matched_genes:
        unique_genes.add(item['gene_id'])
    
    # Save summary statistics
    with open(os.path.join(results_dir, "coverage_summary.txt"), 'w') as f:
        f.write(f"Overall Metabolite Coverage: {overall_metabolite_coverage:.2%}\n")
        f.write(f"Overall Reaction Coverage: {overall_reaction_coverage:.2%}\n")
        f.write(f"Total Unique Measured Metabolites: {len(unique_metabolites)}\n")
        f.write(f"Total Unique Measured Genes: {len(unique_genes)}\n")
    
    # Write unique metabolites to file
    with open(os.path.join(results_dir, "unique_measured_metabolites.txt"), 'w') as f:
        f.write("Name\tHMDB_ID\tKEGG_ID\n")
        for name, hmdb_id, kegg_id in unique_metabolites:
            f.write(f"{name}\t{hmdb_id or 'NA'}\t{kegg_id or 'NA'}\n")
    
    # Write unique genes to file
    with open(os.path.join(results_dir, "unique_measured_genes.txt"), 'w') as f:
        for gene_id in unique_genes:
            f.write(f"{gene_id}\n")

    # Print summary
    print(f"Total unique measured metabolites across all graphs: {len(unique_metabolites)}")
    print(f"Total unique measured genes across all graphs: {len(unique_genes)}")
    print(f"Overall metabolite coverage: {overall_metabolite_coverage:.2%}")
    print(f"Overall reaction coverage: {overall_reaction_coverage:.2%}")

if __name__ == "__main__":
    main()
