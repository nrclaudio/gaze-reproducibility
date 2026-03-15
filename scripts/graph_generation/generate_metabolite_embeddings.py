import os
import torch
import pickle
import requests
import time
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem
import numpy as np

# Load ChemBERTa model and tokenizer
model_name = "DeepChem/ChemBERTa-77M-MLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()  # Set to evaluation mode

# Define the base URL for the API
base_url = "https://metabolicatlas.org/api/v2"

# Rate limiting parameters
RATE_LIMIT = 5  # requests per second
LAST_REQUEST_TIME = 0

def get_metabolite_details(metabolite_id, model="HumanGem"):
    """Get metabolite details including SMILES from the Metabolic Atlas API"""
    global LAST_REQUEST_TIME
    
    # Rate limiting
    current_time = time.time()
    time_since_last_request = current_time - LAST_REQUEST_TIME
    if time_since_last_request < 1/RATE_LIMIT:
        time.sleep(1/RATE_LIMIT - time_since_last_request)
    
    endpoint = f"{base_url}/metabolites/{metabolite_id}?model={model}"
    response = requests.get(endpoint)
    LAST_REQUEST_TIME = time.time()
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve metabolite details for {metabolite_id}: {response.status_code}")
        return None

def generate_embedding(smiles):
    """Generate embedding from SMILES using ChemBERTa with mean pooling"""
    if not smiles:
        return None
    
    # Check if SMILES is valid
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None
    
    # Tokenize and generate embedding
    try:
        inputs = tokenizer(smiles, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use mean pooling instead of [CLS] token
        # Get attention mask to avoid padding tokens
        attention_mask = inputs['attention_mask']
        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Sum embeddings, taking into account the mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        # Sum the mask to get the actual number of tokens
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # Divide to get the mean
        pooled_output = sum_embeddings / sum_mask
        return pooled_output.numpy().flatten()
    except Exception as e:
        print(f"Error generating embedding for {smiles}: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate SMILES embeddings for metabolic graphs')
    parser.add_argument('--identifier', type=str, default='', help='Suffix for directories (e.g. masked)')
    args = parser.parse_args()

    suffix = f"_{args.identifier}" if args.identifier else ""
    output_base_path = "/home/cnovellarausell/data1/concat-smiles-model/data/graphs"
    input_dir = f"{output_base_path}/PyG_Graphs{suffix}"
    output_dir = f"{output_base_path}/PyG_Graphs_with_SMILES{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing graphs from: {input_dir}")
    print(f"Saving graphs to: {output_dir}")
    
    # Run processing (refactored to take directories)
    process_graphs(input_dir, output_dir)

def process_graphs(input_dir, output_dir):
    """Process all graphs, add SMILES and embeddings"""
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} not found.")
        return

    # Get list of all PyG graph files
    graph_files = [f for f in os.listdir(input_dir) if f.endswith('.pt')]
    
    # Dictionary to cache metabolite details and embeddings
    metabolite_cache = {}
    
    # Add counters for SMILES statistics
    total_metabolites = 0
    metabolites_without_smiles = 0
    unique_metabolites = set()
    unique_metabolites_without_smiles = set()
    
    for graph_file in tqdm(graph_files, desc="Processing graphs"):
        # Load the PyG graph
        graph_path = os.path.join(input_dir, graph_file)
        data = torch.load(graph_path, weights_only=False)
        
        # Also load the NetworkX graph for reference
        nx_path = os.path.join(input_dir, graph_file.replace('.pt', '_nx.pkl'))
        if os.path.exists(nx_path):
            with open(nx_path, 'rb') as f:
                G = pickle.load(f)
        else:
            print(f"NetworkX graph not found for {graph_file}")
            continue
        
        # Add lists to store SMILES and embeddings
        data.smiles = []
        embeddings = []
        
        # Process each node
        for i, node_type in enumerate(data.node_types):
            if node_type == 'metabolite':
                node_id = data.node_ids[i]
                total_metabolites += 1
                unique_metabolites.add(node_id)
                
                # Check cache first
                if node_id in metabolite_cache:
                    smiles, embedding = metabolite_cache[node_id]
                else:
                    # Get metabolite details from API
                    metabolite_details = get_metabolite_details(node_id)
                    smiles = None
                    if metabolite_details and 'smiles' in metabolite_details:
                        smiles = metabolite_details.get('smiles')
                    
                    # Generate embedding if SMILES is available
                    embedding = None
                    if smiles:
                        embedding = generate_embedding(smiles)
                    
                    # Cache the result
                    metabolite_cache[node_id] = (smiles, embedding)
                
                # Track missing SMILES
                if not smiles:
                    metabolites_without_smiles += 1
                    unique_metabolites_without_smiles.add(node_id)
                
                # Store SMILES
                data.smiles.append(smiles if smiles else "")
                
                # Store embedding
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    # Use zero vector for missing embeddings
                    embeddings.append(np.zeros(model.config.hidden_size))
            else:
                # For non-metabolites, use empty SMILES and zero embeddings
                data.smiles.append("")
                embeddings.append(np.zeros(model.config.hidden_size))
        
        # Convert embeddings to tensor and add to data
        data.node_embeddings = torch.tensor(np.array(embeddings), dtype=torch.float)
        print(f"Node embeddings shape: {data.node_embeddings.shape}")
        
        # Save the updated graph
        output_path = os.path.join(output_dir, graph_file)
        torch.save(data, output_path)
        
        # Also save the NetworkX graph with SMILES added
        for i, node_id in enumerate(data.node_ids):
            if i < len(data.smiles) and node_id in G.nodes:
                G.nodes[node_id]['smiles'] = data.smiles[i]
        
        # Save updated NetworkX graph
        nx_output_path = os.path.join(output_dir, graph_file.replace('.pt', '_nx.pkl'))
        with open(nx_output_path, 'wb') as f:
            pickle.dump(G, f)
        
        # Save metadata in JSON format
        import json
        meta_output_path = os.path.join(output_dir, graph_file.replace('.pt', '_metadata.json'))
        meta_data = {
            'node_ids': data.node_ids,
            'node_types': data.node_types,
            'smiles': data.smiles,
            'embedding_dim': model.config.hidden_size
        }
        with open(meta_output_path, 'w') as f:
            json.dump(meta_data, f, indent=2)
    
    # Report SMILES coverage statistics
    print("\n===== SMILES Coverage Statistics =====")
    print(f"Total metabolites processed: {total_metabolites}")
    print(f"Metabolites without SMILES: {metabolites_without_smiles} ({metabolites_without_smiles/total_metabolites:.2%})")
    print(f"Unique metabolites: {len(unique_metabolites)}")
    print(f"Unique metabolites without SMILES: {len(unique_metabolites_without_smiles)} ({len(unique_metabolites_without_smiles)/len(unique_metabolites):.2%})")
    
    # Save missing SMILES list
    missing_smiles_path = os.path.join(output_dir, "metabolites_without_smiles.txt")
    with open(missing_smiles_path, 'w') as f:
        for node_id in sorted(unique_metabolites_without_smiles):
            f.write(f"{node_id}\n")
    print(f"List of metabolites without SMILES saved to: {missing_smiles_path}")

if __name__ == "__main__":
    main()
    print("Completed processing all graphs!")
