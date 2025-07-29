# Import required libraries
from huggingface_hub import snapshot_download # To download vector data
from glob import glob
from datasets import load_dataset
from bs4 import BeautifulSoup
import re
import os
from tqdm import tqdm
################################################################################

# Download dataset
repo_id = "bluuebunny/crossref_metadata_embeddings_split_2025_binary"
repo_type = "dataset"
local_dir = "/mnt/block_volume/volumes/milvus/embeddings_data"
allow_patterns = "*.parquet"

# Download the repo
downloaded_dir = snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=allow_patterns, cache_dir='/mnt/block_volume/hf_cache')
print(f"Downloaded '{repo_id}' at '{downloaded_dir}'")

# Gather files from the folder
embedding_files = glob(f'{downloaded_dir}/data/*.parquet')
embedding_files.sort()

# Create directory to save processed data
processed_folder = f"{downloaded_dir}/processed_data"
os.makedirs(processed_folder, exist_ok=True)

################################################################################

def prepare(row):
    # Extract abstract
    soup = BeautifulSoup(row['abstract'], 'html.parser')
    found = soup.find(re.compile(r'\w+:p|p'))
    row['abstract'] = found.get_text(strip=True) if found else soup.get_text(strip=True)
    
    # Trimming: (num of characters) =  (max milvus bytes allowed) / (max bytes per character) for utf-8.
    # Trim title
    row['title'] = str(row['title'])
    if len(row['title']) > 512: # 2048/4
        row['title'] = row['title'][:509] + '...'

    # Trim authors
    row['author'] = ", ".join(row['author'])
    if len(row['author']) > 128: # 512/4
        row['author'] = row['author'][:125] + '...'

    # Trim abstract
    row['abstract'] = str(row['abstract'])
    if len(row['abstract']) > 1024: # 4096/4 
        row['abstract'] = row['abstract'][:1149] + '...'

    return row

################################################################################
for embedding_file in tqdm(embedding_files[6:]):
    
    print(f"Processing: {embedding_file}")
    
    dataset = load_dataset("parquet", data_files=embedding_file, split='train', cache_dir='/mnt/block_volume/hf_cache')
                    
    dataset = dataset.map(prepare, num_proc=4)
    
    dataset.to_parquet(f"{processed_folder}/{os.path.basename(embedding_file)}")