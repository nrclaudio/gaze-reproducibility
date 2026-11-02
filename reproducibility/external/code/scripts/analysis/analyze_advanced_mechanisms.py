#!/usr/bin/env python3
"""
Advanced Mechanism Analysis for LOMO.
Probes Chemical Density and Enzyme Signal Strength with fuzzy biological mapping.
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
import mudata as mu
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils import to_networkx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
from tqdm import tqdm
import argparse
from pathlib import Path
import re

# Add project root to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
sys.path.insert(0, _project_root)
from src.utils.data_paths import get_graph_path, get_mudata_path
from torch_geometric.data import Data

# Allow PyG objects
try:
    from torch_geometric.data.data import DataEdgeAttr
    torch.serialization.add_safe_globals([Data, DataEdgeAttr])
except (ImportError, AttributeError):
    torch.serialization.add_safe_globals([Data])

def clean_name(name):
    """Normalize names for fuzzy matching: lowercase, remove special chars and D/L prefixes."""
    if not name:
        return ""
    s = str(name).lower().strip()
    s = re.sub(r'^[dl]-', '', s)
    s = re.sub(r'[^a-z0-9]', '', s)
    return s

def analyze_advanced(results_dir, graph, mdata, output_dir):
    print(f"\nAnalyzing: {results_dir.name}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load Modalities
    met_key = next((k for k in mdata.mod.keys() if k.lower() in ['metabolomics', 'metabolite']), None)
    rna_key = next((k for k in mdata.mod.keys() if k.lower() in ['rna', 'transcriptomics', 'gene']), None)

    if not met_key:
        return
    met_mod = mdata.mod[met_key]
    rna_mod = mdata.mod[rna_key] if rna_key else None

    # Load Results
    lomo_cv_dir = results_dir / 'lomo_cross_validation'
    fold_files = list(lomo_cv_dir.glob('fold_*_result.json'))
    if not fold_files:
        return

    results = []
    for f in fold_files:
        try:
            with open(f, 'r') as file:
                d = json.load(file)
                if d.get('fold_idx') != 999:
                    results.append(d)
        except Exception:
            pass

    if not results:
        return
    df = pd.DataFrame([{
        'met_idx': r['metabolite_idx'],
        'R2': r['r2'],
        'met_id': r.get('metabolite_id', '')
    } for r in results]).dropna(subset=['R2'])
    
    # Resolve IDs via Graph Index
    true_ids = []
    for idx in df['met_idx']:
        h_id = graph.hmdb_ids[idx] if hasattr(graph, 'hmdb_ids') else None
        if not h_id and hasattr(graph, 'node_names'):
            h_id = graph.node_names[idx]
        true_ids.append(str(h_id) if h_id else "N/A")
    df['True_ID'] = true_ids

    # 2. Chemical Density
    if 'SMILES' in met_mod.var.columns:
        # Build robust Mapping: CleanName -> SMILES
        clean_to_smiles = {}
        # a) Map by Index (Row Name)
        for name, s in zip(met_mod.var_names, met_mod.var['SMILES']):
            if s and str(s) != "nan" and len(str(s)) > 1:
                clean_to_smiles[clean_name(name)] = s
        
        # b) Map by HMDB Col
        hmdb_col = next((c for c in ['HMDB_trailing_zeros', 'HMDB', 'hmdb'] if c in met_mod.var.columns), None)
        if hmdb_col:
            for h, s in zip(met_mod.var[hmdb_col], met_mod.var['SMILES']):
                if h and s and str(s) != "nan" and len(str(s)) > 1:
                    clean_to_smiles[clean_name(h)] = s

        all_library_smiles = list(set(clean_to_smiles.values()))
        
        # Precompute library fingerprints for speed
        library_fps = []
        for s in all_library_smiles:
            try:
                m = Chem.MolFromSmiles(s)
                if m:
                    library_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2))
            except Exception:
                pass

        densities = []
        mapped_smiles_count = 0
        for _, row in df.iterrows():
            target_smiles = clean_to_smiles.get(clean_name(row['True_ID']), "N/A")
            
            # Fallback to graph strings
            if target_smiles == "N/A" and hasattr(graph, 'x_smiles_strings'):
                target_smiles = graph.x_smiles_strings[row['met_idx']]
            
            if target_smiles != "N/A" and len(str(target_smiles)) > 1:
                try:
                    t_mol = Chem.MolFromSmiles(str(target_smiles))
                    t_fp = AllChem.GetMorganFingerprintAsBitVect(t_mol, 2)
                    sims = DataStructs.BulkTanimotoSimilarity(t_fp, library_fps)
                    # Exclude self matches
                    sims = [s for s in sims if s < 0.99]
                    densities.append(max(sims) if sims else 0)
                    mapped_smiles_count += 1
                except Exception:
                    densities.append(0)
            else:
                densities.append(0)
        df['Chemical_Support'] = densities
        print(f"  Chemical Support: Mapped {mapped_smiles_count}/{len(df)} metabolites to SMILES.")

    # 3. Enzyme Signal Strength
    if rna_mod is not None:
        X_rna = rna_mod.X.toarray() if hasattr(rna_mod.X, 'toarray') else rna_mod.X
        variances = np.var(X_rna, axis=0)
        gene_var_map = dict(zip(rna_mod.var_names, variances))
        
        # Symbol Map
        symbol_col = next((c for c in rna_mod.var.columns if 'symbol' in c.lower() or 'gene_name' in c.lower()), None)
        if symbol_col:
            gene_var_map.update(dict(zip(rna_mod.var[symbol_col], variances)))
        
        # Ensembl Map
        ensembl_col = next((c for c in rna_mod.var.columns if 'ensembl' in c.lower()), None)
        if ensembl_col:
            gene_var_map.update(dict(zip(rna_mod.var[ensembl_col], variances)))
        
        rxn_vars = {}
        enz_indices = [i for i, t in enumerate(graph.node_types) if t == 'reaction']
        mapped_rxns = 0
        for r_idx in enz_indices:
            rule = str(graph.gene_rules[r_idx]) if hasattr(graph, 'gene_rules') else ""
            genes = re.findall(r'ENSG\d+|[A-Z0-9]+', rule)
            genes = [g for g in genes if len(g) > 2 and g.lower() not in ['and', 'or']]
            val = max([gene_var_map.get(g, 0) for g in genes]) if genes else 0
            rxn_vars[r_idx] = val
            if val > 0:
                mapped_rxns += 1
        
        G = to_networkx(graph, to_undirected=True)
        met_signal = []
        for _, row in df.iterrows():
            neighbors = [n for n in G.neighbors(row['met_idx']) if n in rxn_vars]
            met_signal.append(np.mean([rxn_vars[n] for n in neighbors]) if neighbors else 0)
        df['Enzyme_Signal'] = met_signal
        
        non_zero_signal = (df['Enzyme_Signal'] > 0).sum()
        print(f"  Enzyme Signal: Mapped {non_zero_signal}/{len(df)} metabolites to RNA neighbors.")

    # 4. Plots
    out = Path(output_dir)
    sns.set_style("whitegrid")
    for col, color, label in [('Chemical_Support', 'red', 'Chemical Support'), ('Enzyme_Signal', 'green', 'Enzyme Signal')]:
        if col in df.columns and df[col].nunique() > 1:
            plt.figure(figsize=(8, 6))
            sns.regplot(data=df, x=col, y='R2', color=color, scatter_kws={'alpha': 0.5})
            plt.title(f"{label} vs R2\n{results_dir.name}")
            plt.savefig(out / f"{col.lower()}.png", dpi=300)
            plt.close()

    df.to_csv(out / "advanced_metrics.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--out', type=str, default='results/advanced_analysis')
    args = parser.parse_args()
    
    graph = torch.load(get_graph_path(), map_location='cpu', weights_only=False)
    mdata_path = "/data1/cnovellarausell/concat-smiles-model/data/raw/mudata_with_smiles.h5mu"
    if not os.path.exists(mdata_path):
        mdata_path = str(Path(get_mudata_path()).parent / "mudata_with_smiles.h5mu")
    
    try:
        import h5py
        mods = {}
        with h5py.File(mdata_path, 'r') as f:
            if 'mod' in f:
                for k in f['mod'].keys():
                    mods[k] = mu.read_h5ad(mdata_path, mod=k)
        mdata = mu.MuData(mods)
    except Exception:
        mdata = mu.read(mdata_path)

    base_dir = Path(args.dir)
    if (base_dir / 'lomo_cross_validation').exists():
        analyze_advanced(base_dir, graph, mdata, args.out)
    else:
        for sub_path in sorted(base_dir.iterdir()):
            if sub_path.is_dir() and (sub_path / 'lomo_cross_validation').exists():
                analyze_advanced(sub_path, graph, mdata, str(Path(args.out) / sub_path.name))

if __name__ == "__main__":
    main()