# Import required libraries
from huggingface_hub import snapshot_download # To download vector data
from glob import glob
from datasets import load_dataset
from bs4 import BeautifulSoup
import re

################################################################################

repo_id = "bluuebunny/crossref_metadata_embeddings_split_2025_binary"
repo_type = "dataset"
local_dir = "volumes/milvus"
allow_patterns = "*.parquet"

# Download the repo
embeddings_folder = snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=allow_patterns)

# Gather files from the folder
embedding_files = glob(f'{embeddings_folder}/*.parquet')

def prepare(row):
    
    # # Remove newline characters from authors, title, abstract and categories columns
    # row['title'] = row['title'].astype(str).str.replace('\n', ' ', regex=False)
    
    # row['authors'] = row['authors'].astype(str).str.replace('\n', ' ', regex=False)
        
    # row['abstract'] = row['abstract'].astype(str).str.replace('\n', ' ', regex=False)

    # Extract abstract saved in html format
    soup = BeautifulSoup(row['abstract'], 'html.parser')
    found = soup.find(re.compile(r'\w+:p|p')) # i think all abstract texts have tags with a 'p' in them. 
    row['abstract'] = found.get_text() if found else row['abstract']
        
    # Trim title to 512 characters
    row['title'] = row['title'].progress_apply(lambda x: x[:508] + '...' if len(x) > 512 else x)
    
    # Create authors text from list
    row['authors'] = ", ".join(row['authors'])
    # Trim authors to 128 characters
    row['authors'] = row['authors'].progress_apply(lambda x: x[:124] + '...' if len(x) > 128 else x)

    # Trim abstract to 3072 characters
    row['abstract'] = row['abstract'].progress_apply(lambda x: x[:3068] + '...' if len(x) > 3072 else x)

for embedding_file in embedding_files:
    
    dataset = load_dataset("parquet", data_files=embedding_file, split='train')
    
    dataset = dataset.map(prepare, num_proc=4)
    
    dataset.to_parquet(embedding_file)