# Import required libraries
import os

from datasets import load_dataset

# Also, whether to split the consolidated file into parts
# Set 0 for no sharding.
num_parts = 10

# Create a new dataset
dataset = load_dataset("parquet", data_dir="crossref_metadata/", num_proc=24)

# Save consolidated dataset
dataset["train"].to_parquet("crossref_metadata.parquet")

# Save consolidated dataset in parts
if num_parts:
    
    print(f"Splitting consolidated dataset in {num_parts} parts.")
    
    split_folder = 'crossref_metadata_split'
    os.makedirs(split_folder, exist_ok=True)

    for i in range(num_parts):
        dataset["train"].shard(num_parts, index=i).to_parquet(
            f"{split_folder}/part_{i + 1}.parquet"
        )
