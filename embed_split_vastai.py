# Import required libraries
import os
from glob import glob
from multiprocessing import Pool, set_start_method

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from huggingface_hub import snapshot_download, HfApi
from dotenv import load_dotenv
import torch
import backoff

################################################################################

# Load secrets
load_dotenv('.env')

# Declare batch size
BATCH_SIZE = os.getenv('BATCH_SIZE')
print(BATCH_SIZE)

# Define the embedding model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Folder to save files in
embedding_folder = "crossref_embedding_split"
os.makedirs(embedding_folder, exist_ok=True)

embedding_folder_binary = "crossref_embedding_split_binary"
os.makedirs(embedding_folder_binary, exist_ok=True)

# Connect to hfapi
hf_client = HfApi(token=os.getenv('HF_ACCESS_TOKEN'))

# Get the HF repo name 
destination_repo_name = os.getenv('HF_REPO_EMBEDDING_SPLIT')

# Extract Hugging face username
username = hf_client.whoami()['name']
print(f"Uploading to {username}'s huggingface account!")

# Create repos to uplod files to
embedding_repo_id = f"{username}/{destination_repo_name}"
hf_client.create_repo(embedding_repo_id, private=False, repo_type='dataset', exist_ok=True)

binary_embedding_repo_id = f"{username}/{destination_repo_name + '_binary'}"
hf_client.create_repo(binary_embedding_repo_id, private=False, repo_type='dataset', exist_ok=True)


################################################################################

# Download the split dataset
metadata_repo_id = os.getenv('HF_REPO_METADATA_SPLIT')
snapshot_download(repo_id=metadata_repo_id, repo_type='dataset', local_dir=metadata_repo_id)

# Split metadata
split_metadata_files = glob(f"{metadata_repo_id}/*.parquet")
split_metadata_files.sort()

################################################################################

@backoff.on_exception(backoff.expo, torch.OutOfMemoryError, jitter=backoff.full_jitter)
def process_metadata(device_number, metadata_file):
    
    # Move the model on the right GPU if it's not there already
    device = f"cuda:{device_number % torch.cuda.device_count()}"
    model.to(device)
    
    # Read
    df = pd.read_parquet(metadata_file)
    
    # try
    df = df.sample(BATCH_SIZE*10)

    # Calculate the embeddings for text and paragraphs
    df["vector"] = model.encode(
        df["abstract"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()
    
    # Convert lists back to numpy arrays with the correct dtype
    df["vector"] = df["vector"].apply(
        lambda x: np.array(x, dtype=np.float32)
    )

    # Save Float
    embed_file = f"{embedding_folder}/{os.path.basename(metadata_file)}"
    df.to_parquet(embed_file, index=False)
    
    # Upload Float
    hf_client.upload_file(path_or_fileobj=embed_file, path_in_repo=embed_file, repo_id=embedding_repo_id, repo_type='dataset')
    
    # Binarise
    df['vector'] = df['vector'].apply(lambda dense_vector : np.packbits(np.where(dense_vector >= 0, 1, 0)).tobytes())
    
    # Save Binary
    binary_embed_file = f"{embedding_folder_binary}/{os.path.basename(metadata_file)}"
    df.to_parquet(binary_embed_file, index=False)
    
    # Upload Binary
    hf_client.upload_file(path_or_fileobj=binary_embed_file, path_in_repo=binary_embed_file, repo_id=binary_embedding_repo_id, repo_type='dataset')

################################################################################

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    # try
    split_metadata_files = split_metadata_files[:2]


    with Pool(torch.cuda.device_count()) as pool:
        results = pool.starmap(process_metadata, enumerate(split_metadata_files))