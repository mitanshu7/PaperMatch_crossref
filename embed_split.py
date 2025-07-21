# Import required libraries
import os
from glob import glob

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Define the embedding model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# declare batch size
BATCH_SIZE = 120

# Split metadata
split_metadata_files = glob("crossref_metadata_split/*.parquet")
split_metadata_files.sort()

# Folder to save files in
embedding_folder = "crossref_embedding_split"
os.makedirs(embedding_folder, exist_ok=True)

# Go over data files one by one
for metadata_file in tqdm(split_metadata_files):
    # Read
    df = pd.read_parquet(metadata_file)

    # Calculate the embeddings for text and paragraphs
    df["vector"] = model.encode(
        df["abstract"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()

    # Save
    df.to_parquet(f"{embedding_folder}/{os.path.basename(metadata_file)}", index=False)