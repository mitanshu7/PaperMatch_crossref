# Import required libraries
from sentence_transformers import SentenceTransformer  # Embedding model
from datasets import load_dataset

# Define the embedding model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Function to create embeddings
def embed(rows):
    embeddings = model.encode(rows["abstract"], batch_size=96, show_progress_bar=True, convert_to_numpy=True).tolist()
    return {'vector' : embeddings}

# Create a new dataset
dataset = load_dataset("parquet", data_files="crossref_metadata.parquet", streaming=True)

# Calculate the embeddings for text and paragraphs
dataset = dataset.map(embed, batched=True, batch_size=960)

# Save
dataset["train"].to_parquet("crossref_metadata_embeddings.parquet")