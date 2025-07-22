# Import required libraries
import os
from multiprocess import set_start_method

from sentence_transformers import SentenceTransformer
import numpy as np
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import torch
from datasets import load_dataset

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

################################################################################

# Download the split dataset
metadata_repo_id = os.getenv('HF_REPO_METADATA_SPLIT')
print(f'Download metadata from: {metadata_repo_id}')
snapshot_download(repo_id=metadata_repo_id, repo_type='dataset', local_dir=metadata_repo_id)

# Repo names to upload data to. 
# TODO create new repo here only
embedding_repo_id = os.getenv('HF_REPO_EMBEDDING_SPLIT')
embedding_repo_id_binary = os.getenv('HF_REPO_EMBEDDING_SPLIT_BINARY')
print(f'Will upload float embeddings to {embedding_repo_id} and binary to {embedding_repo_id_binary}.')
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
    
    # Load dataset
    print(f'Loading dataset')
    dataset = load_dataset("parquet", data_dir=metadata_repo_id, split='train')
    
    # try
    print(f"Slicing dataset to try it out")
    dataset = dataset.shuffle().take(BATCH_SIZE*20)

    # Embed
    print('Embedding abstracts')
    dataset = dataset.map(
        embed_metadata,
        batched=True,
        batch_size=BATCH_SIZE,
        with_rank=True,
        num_proc=torch.cuda.device_count(),  # one process per GPU
    )
    
    # Upload floats
    print('Uploading float embeddings')
    dataset.push_to_hub(embedding_repo_id)
    
    # Binarise embeddings
    print('Binarising vectors')
    dataset = dataset.map(binarise)
    
    # Upload binarised vectors
    print('Uploading binary embeddings')
    dataset.push_to_hub(embedding_repo_id_binary)