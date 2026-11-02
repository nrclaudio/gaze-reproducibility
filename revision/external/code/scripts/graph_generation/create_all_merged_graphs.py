#!/usr/bin/env python3
"""
Script to create all merged graphs needed by different training scripts.

This script generates merged graphs for all processing stages:
- PyG_merged_graph_with_embeddings (for train.py, train_film.py) 
- PyG_merged_graph_with_SMILES_with_embeddings (for train_concat_smiles.py, train_smiles.py)
- PyG_merged_graph_with_gene_features (for train.py with gene features)

Usage:
    python create_all_merged_graphs.py
"""

import os
import sys
from merge_graphs import merge_graphs

def create_all_merged_graphs():
    """Create merged graphs for all processing stages needed by training scripts"""
    
    # Define the stages that training scripts need
    required_stages = [
        'with_embeddings',           # For train.py, train_film.py (no SMILES)
        'with_SMILES_with_embeddings', # For train_concat_smiles.py, train_smiles.py
        'with_gene_features'         # For train.py with gene features
    ]
    
    print("Creating merged graphs for all required processing stages...")
    print("=" * 60)
    
    successful_stages = []
    failed_stages = []
    
    for stage in required_stages:
        print(f"\n🔄 Processing stage: {stage}")
        print("-" * 40)
        
        try:
            merged_data = merge_graphs(input_stage=stage)
            
            if merged_data is not None:
                print(f"✅ Successfully created merged graph for: {stage}")
                successful_stages.append(stage)
            else:
                print(f"❌ Failed to create merged graph for: {stage}")
                failed_stages.append(stage)
                
        except FileNotFoundError as e:
            print(f"❌ Input directory not found for {stage}: {e}")
            failed_stages.append(stage)
        except Exception as e:
            print(f"❌ Error processing {stage}: {e}")
            failed_stages.append(stage)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"✅ Successfully created: {len(successful_stages)}/{len(required_stages)} merged graphs")
    for stage in successful_stages:
        print(f"   - {stage}")
    
    if failed_stages:
        print(f"\n❌ Failed to create: {len(failed_stages)} merged graphs")
        for stage in failed_stages:
            print(f"   - {stage}")
        print("\n💡 Make sure you have run the preprocessing scripts to generate the input graphs:")
        print("   1. python metabolicatlas_api.py")
        print("   2. python generate_metabolite_embeddings.py") 
        print("   3. python initialize_enzyme_embeddings.py")
        print("   4. python initialize_reaction_gene_features.py")
    
    print("\n🎯 Training scripts can now use the appropriate merged graphs:")
    print("   - train.py → PyG_merged_graph_with_embeddings/")
    print("   - train_film.py → PyG_merged_graph_with_embeddings/")
    print("   - train_concat_smiles.py → PyG_merged_graph_with_SMILES_with_embeddings/")
    print("   - train_smiles.py → PyG_merged_graph_with_SMILES_with_embeddings/")
    print("   - train.py (gene features) → PyG_merged_graph_with_gene_features/")
    
    return len(failed_stages) == 0

if __name__ == "__main__":
    success = create_all_merged_graphs()
    sys.exit(0 if success else 1) 