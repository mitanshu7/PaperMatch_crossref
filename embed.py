# Import required libraries
from sentence_transformers import SentenceTransformer  # Embedding model
from datasets import load_dataset

# Declare batch size
BATCH_SIZE = 96

# Define the embedding model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Function to create embeddings
def embed(rows):
    embeddings = model.encode(rows["abstract"], batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    return {'vector' : embeddings}

# Create a new dataset
dataset = load_dataset("parquet", data_files="crossref_metadata.parquet", streaming=True, split='train')

# Calculate the embeddings for text and paragraphs
dataset = dataset.map(embed, batched=True, batch_size=BATCH_SIZE*10)

# Save
dataset.to_parquet("crossref_metadata_embeddings.parquet")