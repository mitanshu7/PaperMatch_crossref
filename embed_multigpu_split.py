# Import required libraries
import os
from multiprocess import set_start_method
from time import time
from glob import glob

from sentence_transformers import SentenceTransformer
import numpy as np
from huggingface_hub import snapshot_download, HfApi
from dotenv import load_dotenv
import torch
from datasets import load_dataset

start_time = time()
################################################################################

# Load secrets
load_dotenv('.env')

# Declare batch size
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
print(f'Using BATCH SIZE: {BATCH_SIZE}')

# Define the embedding model
embedding_model_name = "mixedbread-ai/mxbai-embed-large-v1"
print(f'Using embedding model: {embedding_model_name}')
model = SentenceTransformer(embedding_model_name, device='cpu')

# Create huggingface client to transact
hf_client = HfApi(token=os.getenv('HF_API_KEY'))
hf_username = hf_client.whoami()['name']
print(f'Huggingface username: {hf_username}')
################################################################################

# Download the split dataset
metadata_repo_id = os.getenv('HF_REPO_METADATA_SPLIT')
print(f'Download metadata from: {metadata_repo_id}')
snapshot_download(repo_id=metadata_repo_id, repo_type='dataset', local_dir=metadata_repo_id)

# Gather individual files
metadata_files = glob(f'{metadata_repo_id}/**/*.parquet', recursive=True)
metadata_files.sort()

# Repo names to upload data to. 
embedding_repo_id = os.getenv('HF_REPO_EMBEDDING_SPLIT', f'{hf_username}/crossref_metadata_embeddings_split_2025')
hf_client.create_repo(repo_id=embedding_repo_id, private=False, repo_type='dataset', exist_ok=True)
os.makedirs(embedding_repo_id, exist_ok=True)
print(f'Will upload float embeddings to {embedding_repo_id}')

embedding_repo_id_binary = os.getenv('HF_REPO_EMBEDDING_SPLIT_BINARY', f'{hf_username}/crossref_metadata_embeddings_split_2025_binary')
hf_client.create_repo(repo_id=embedding_repo_id_binary, private=False, repo_type='dataset', exist_ok=True)
os.makedirs(embedding_repo_id_binary, exist_ok=True)
print(f'Will upload binary embeddings to {embedding_repo_id_binary}.')
################################################################################

def embed_metadata(batch, rank):
    
    # Move the model on the right GPU if it's not there already
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    print(f'Loading Embedding model to GPU:{device}')
    model.to(device)

    # Calculate the embeddings for text and paragraphs
    batch["vector"] = model.encode(
        batch["abstract"],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    
    return batch

# Function to binarise float embeddings
def binarise(row):
    # Make it a numpy array, since batching sends it as list
    float_vector = np.array(row['vector'], dtype=np.float32)
    
    # Binarise
    binary_vector = np.where(float_vector >= 0, 1, 0)
    
    # Pack it to make it milvus compatible
    row['vector'] = np.packbits(binary_vector).tobytes()
    
    return row
    
################################################################################

if __name__ == "__main__":
    # Set context for GPU multiprocess
    set_start_method('spawn')
    
    # Iterate over the metadata files
    for metadata_file in metadata_files:
        
    
        # Load dataset
        print(f'Processing: {metadata_file}')
        dataset = load_dataset("parquet", data_files=metadata_file, split='train')
    
        # Embed
        print('Embedding abstracts')
        dataset = dataset.map(
            embed_metadata,
            batched=True,
            batch_size=BATCH_SIZE,
            with_rank=True,
            num_proc=torch.cuda.device_count(),  # one process per GPU
        )
        
        # Save to parquet
        float_embedding_name = f'{embedding_repo_id}/{os.path.basename(metadata_file)}'
        print(f"Saving file to: {float_embedding_name}")
        dataset.to_parquet(float_embedding_name)
        
        # Upload floats
        print('Uploading float embeddings')
        hf_client.upload_file(path_or_fileobj=float_embedding_name, path_in_repo=f'data/{os.path.basename(metadata_file)}', repo_id=embedding_repo_id, repo_type='dataset')
        
        # Cleanup floats
        print(f'Removing float file: {float_embedding_name}')
        os.remove(float_embedding_name)
        
        # Binarise embeddings
        print('Binarising vectors')
        dataset = dataset.map(binarise)
        
        # Save to parquet
        binary_embedding_name = f'{embedding_repo_id_binary}/{os.path.basename(metadata_file)}'
        print(f"Saving file to: {binary_embedding_name}")
        dataset.to_parquet(binary_embedding_name)
        
        # Upload floats
        print('Uploading binary embeddings')
        hf_client.upload_file(path_or_fileobj=binary_embedding_name, path_in_repo=f'data/{os.path.basename(metadata_file)}', repo_id=embedding_repo_id_binary, repo_type='dataset')
        
        print(f'Removing binary file: {binary_embedding_name}')
        os.remove(binary_embedding_name)
        
    
    # time
    end_time = time()
    print(f"Time taken = {end_time-start_time} seconds")