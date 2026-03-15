#!/usr/bin/env python3
"""
Deep analysis of LOMO results to understand WHY prediction works (or fails).
Correlates performance (R2) with graph topology and chemical properties.
Uses Graph Index lookup to resolve SMILES and RNA mappings.
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils import to_networkx
import argparse
from pathlib import Path
import re
from scipy.stats import mannwhitneyu
import mudata as mu
import requests

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.utils.data_paths import get_graph_path, get_mudata_path
from torch_geometric.data import Data

# Allow PyG objects for torch.load
try:
    from torch_geometric.data.data import DataEdgeAttr
    torch.serialization.add_safe_globals([Data, DataEdgeAttr])
except (ImportError, AttributeError):
    torch.serialization.add_safe_globals([Data])

from rdkit import Chem

def count_carbons(smiles):
    """Count carbon atoms in a SMILES string using RDKit."""
    if not smiles or not isinstance(smiles, str) or smiles in ["N/A", "", "nan", "None"]:
        return 0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    except Exception:
        pass
    return 0

def get_smiles_from_api(metabolite_id, cache):
    """Fetch SMILES from Metabolic Atlas API with local caching."""
    if metabolite_id in cache:
        return cache[metabolite_id]
    if not str(metabolite_id).startswith('M_'):
        return "N/A"
    try:
        url = f"https://metabolicatlas.org/api/v2/metabolites/{metabolite_id}?model=HumanGem"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            smiles = response.json().get('smiles', "N/A")
            cache[metabolite_id] = smiles
            return smiles
    except Exception:
        pass
    return "N/A"

def clean_id(s):
    return str(s).lower().strip()

def analyze_mechanisms(results_dir, graph_data, mdata, output_dir, api_cache):
    print(f"Analyzing folder: {results_dir.name}")
    
    # 1. Build Mappings from MuData
    met_mod_key = next((k for k in mdata.mod.keys() if k.lower() in ['metabolomics', 'metabolite']), 'metabolomics')
    met_mod = mdata.mod[met_mod_key]
    
    id_to_smiles = {}
    if 'SMILES' in met_mod.var.columns:
        # a) By Index
        for idx, s in zip(met_mod.var_names, met_mod.var['SMILES']):
            if s and str(s) != "nan":
                id_to_smiles[clean_id(idx)] = s
        # b) By HMDB
        hmdb_col = next((c for c in ['HMDB_trailing_zeros', 'HMDB', 'hmdb'] if c in met_mod.var.columns), None)
        if hmdb_col:
            for h, s in zip(met_mod.var[hmdb_col], met_mod.var['SMILES']):
                if h and s and str(s) != "nan":
                    id_to_smiles[clean_id(h)] = s

    # 2. Load LOMO Results
    lomo_cv_dir = results_dir / 'lomo_cross_validation'
    fold_files = list(lomo_cv_dir.glob('fold_*_result.json'))
    
    if not fold_files:
        return

    fold_results = []
    for f in fold_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                if data.get('fold_idx') != 999 and data.get('metabolite_idx') != -1:
                    fold_results.append(data)
        except Exception:
            continue

    if not fold_results:
        return

    df = pd.DataFrame([{
        'metabolite_idx': r['metabolite_idx'],
        'R2': r['r2'],
        'result_id': r.get('metabolite_id', '')
    } for r in fold_results]).dropna(subset=['R2'])
    
    # 3. Graph Analysis
    G = to_networkx(graph_data, to_undirected=False)
    enz_indices = [i for i, t in enumerate(graph_data.node_types) if t == 'reaction']
    G_undir = G.to_undirected()
    
    metrics = []
    for idx, row in df.iterrows():
        met_idx = int(row['metabolite_idx'])
        degree = G.degree(met_idx)
        
        # Proximity
        min_dist_to_enz = 999
        try:
            for enz_idx in enz_indices:
                try:
                    d = nx.shortest_path_length(G_undir, source=met_idx, target=enz_idx)
                    if d < min_dist_to_enz:
                        min_dist_to_enz = d
                    if min_dist_to_enz == 1:
                        break
                except nx.NetworkXNoPath:
                    continue
        except Exception:
            pass
        
        # --- ROBUST SMILES RETRIEVAL ---
        # Get true identity from graph
        h_id = graph_data.hmdb_ids[met_idx] if hasattr(graph_data, 'hmdb_ids') else None
        name = graph_data.node_names[met_idx] if hasattr(graph_data, 'node_names') else None
        
        smiles = "N/A"
        # Try Mapping MuData
        for candidate in [h_id, name, row['result_id']]:
            if candidate and smiles == "N/A":
                smiles = id_to_smiles.get(clean_id(candidate), "N/A")
        
        # Try Graph/API
        if smiles == "N/A" and hasattr(graph_data, 'node_ids'):
            m_id = graph_data.node_ids[met_idx]
            if str(m_id).startswith('M_'):
                smiles = get_smiles_from_api(m_id, api_cache)
        
        if smiles == "N/A" and hasattr(graph_data, 'x_smiles_strings'):
            smiles = graph_data.x_smiles_strings[met_idx]

        carbons = count_carbons(smiles)
        
        metrics.append({
            'metabolite_idx': met_idx,
            'Degree': degree,
            'Dist_to_Enzyme': min_dist_to_enz if min_dist_to_enz < 50 else 50,
            'Carbon_Count': carbons,
            'True_ID': str(h_id if h_id else name)
        })
        
    topo_df = pd.DataFrame(metrics)
    full_df = pd.merge(df, topo_df, on='metabolite_idx')
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    
    # plots
    corr_cols = ['R2', 'Degree', 'Dist_to_Enzyme', 'Carbon_Count']
    corr_cols = [c for c in corr_cols if c in full_df.columns and full_df[c].nunique() > 1]
    
    if len(corr_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(full_df[corr_cols].corr(), annot=True, cmap='RdBu_r', center=0, fmt=".2f")
        plt.title(f"Feature Correlations ({results_dir.name})")
        plt.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'), dpi=300)
        plt.close()

    # Updated Plotting Logic: Always show all 3 factors
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    full_df['Group'] = full_df['R2'].apply(lambda x: 'Positive R²' if x > 0 else 'Negative R²')
    
    # 1. Degree
    if full_df['Degree'].nunique() > 1:
        sns.boxplot(data=full_df, x='Group', y='Degree', ax=axes[0], palette='Set2')
        sns.stripplot(data=full_df, x='Group', y='Degree', ax=axes[0], color='black', alpha=0.3)
    axes[0].set_title('Topological Connectivity (Degree)')
    
    # 2. Distance
    if full_df['Dist_to_Enzyme'].nunique() > 1:
        sns.boxplot(data=full_df, x='Group', y='Dist_to_Enzyme', ax=axes[1], palette='Set2')
        sns.stripplot(data=full_df, x='Group', y='Dist_to_Enzyme', ax=axes[1], color='black', alpha=0.3)
    axes[1].set_title('Proximity to Measured Enzymes')
    
    # 3. Carbon Count (Chemical Complexity)
    if full_df['Carbon_Count'].nunique() > 1:
        sns.boxplot(data=full_df, x='Group', y='Carbon_Count', ax=axes[2], palette='Set2')
        sns.stripplot(data=full_df, x='Group', y='Carbon_Count', ax=axes[2], color='black', alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "Carbon Count Constant (0)", ha='center', va='center')
    axes[2].set_title('Chemical Complexity (Carbons)')
    
    plt.suptitle(f"Comparative Analysis: {results_dir.name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'success_factors_boxplot.png'), dpi=300)
    plt.close()
    
    full_df.to_csv(os.path.join(output_dir, 'prediction_success_factors.csv'), index=False)
    
    # Statistical Table
    pos = full_df[full_df['R2'] > 0]
    neg = full_df[full_df['R2'] <= 0]
    
    def get_p(col):
        if len(pos) > 0 and len(neg) > 0 and full_df[col].nunique() > 1:
            try:
                return mannwhitneyu(pos[col], neg[col], alternative='two-sided').pvalue
            except Exception:
                return 1.0
        return 1.0

    print(f"\n--- STATS: {results_dir.name} ---")
    print(f"Positive Group (N={len(pos)}) vs Negative Group (N={len(neg)})")
    comparison = pd.DataFrame({
        'Metric': ['Degree', 'Dist', 'Carbons'],
        'Pos_Mean': [pos['Degree'].mean(), pos['Dist_to_Enzyme'].mean(), pos['Carbon_Count'].mean()],
        'Neg_Mean': [neg['Degree'].mean(), neg['Dist_to_Enzyme'].mean(), neg['Carbon_Count'].mean()],
        'P-Val': [get_p('Degree'), get_p('Dist_to_Enzyme'), get_p('Carbon_Count')]
    })
    print(comparison.to_markdown(index=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--out', type=str, default='results/mechanism_analysis')
    args = parser.parse_args()
    
    # Cache
    cache_path = Path("metabolite_smiles_cache.json")
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            api_cache = json.load(f)
    else:
        api_cache = {}

    # MuData (Populated)
    mdata_path = get_mudata_path()
    pop_path = Path(mdata_path).parent / "mudata_with_smiles.h5mu"
    if pop_path.exists():
        mdata_path = str(pop_path)
    print(f"Using MuData: {mdata_path}")
    try:
        mdata = mu.read(mdata_path)
    except Exception:
        mu.set_options(pull_on_update=False)
        mdata = mu.read(mdata_path)

    # Graph
    graph_data = torch.load(get_graph_path(), map_location='cpu', weights_only=False)

    base_dir = Path(args.dir)
    if (base_dir / 'lomo_cross_validation').exists():
        analyze_mechanisms(base_dir, graph_data, mdata, args.out, api_cache)
    else:
        for sub_path in sorted(base_dir.iterdir()):
            if sub_path.is_dir() and (sub_path / 'lomo_cross_validation').exists():
                analyze_mechanisms(sub_path, graph_data, mdata, str(Path(args.out) / sub_path.name), api_cache)

    with open(cache_path, 'w') as f:
        json.dump(api_cache, f)

if __name__ == "__main__":
    main()
