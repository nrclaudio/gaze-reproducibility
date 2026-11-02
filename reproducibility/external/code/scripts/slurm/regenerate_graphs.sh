#!/bin/bash
#SBATCH --job-name=regenerate_graphs
#SBATCH --output=logs/regenerate_graphs_%j.out
#SBATCH --error=logs/regenerate_graphs_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=cpu-zen4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Activate environment
source ~/.bashrc
conda activate MetaboliteRNA

echo "Starting Graph Regeneration Pipeline..."
echo "Date: $(date)"
echo "Node: $(hostname)"

# 0. Backup existing graphs
BACKUP_DIR="data/graphs_backup_$(date +%Y%m%d_%H%M%S)"
echo ">>> Step 0: Backing up existing graphs to ${BACKUP_DIR}"
if [ -d "data/graphs" ]; then
    cp -r data/graphs "${BACKUP_DIR}"
    echo "Backup created successfully."
else
    echo "No existing data/graphs directory found. Skipping backup."
fi

# 1. Generate base pathway graphs with tanh-squashed edge weights
echo ">>> Step 1: Running metabolicatlas_api.py (Building Base Graphs)"
python scripts/graph_generation/metabolicatlas_api.py
if [ $? -ne 0 ]; then echo "Step 1 Failed"; exit 1; fi

# 2. Add SMILES embeddings to metabolites
echo ">>> Step 2: Running generate_metabolite_embeddings.py"
python scripts/graph_generation/generate_metabolite_embeddings.py
if [ $? -ne 0 ]; then echo "Step 2 Failed"; exit 1; fi

# 3. Add Enzyme embeddings
echo ">>> Step 3: Running initialize_enzyme_embeddings.py"
python scripts/graph_generation/initialize_enzyme_embeddings.py
if [ $? -ne 0 ]; then echo "Step 3 Failed"; exit 1; fi

# 4. Add Reaction Gene Features (optional but good for completeness)
echo ">>> Step 4: Running initialize_reaction_gene_features.py"
python scripts/graph_generation/initialize_reaction_gene_features.py
if [ $? -ne 0 ]; then echo "Step 4 Failed"; exit 1; fi

# 5. Merge everything into the final training graphs
echo ">>> Step 5: Running create_all_merged_graphs.py"
python scripts/graph_generation/create_all_merged_graphs.py
if [ $? -ne 0 ]; then echo "Step 5 Failed"; exit 1; fi

echo "✅ Graph Regeneration Complete!"
