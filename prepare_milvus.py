# Import required libraries
from pymilvus import MilvusClient, DataType 
import requests
from time import sleep
from glob import glob

################################################################################
# Gather files
# TODO: /mnt/block_volume/volumes/milvus is mapped to /var/lib/milvus/. make below better
files = glob('/mnt/block_volume/volumes/milvus/processed_data/*.parquet')
files = [ ["/var/lib/milvus/processed_data/" + i.split('/')[-1]] for i in files ]

################################################################################
# Create collection
# Define client
client = MilvusClient("http://localhost:19530")

# Drop any of the pre-existing collections
# Need to drop it because otherwise milvus does not check for (and keeps)
# duplicate records
client.drop_collection(
    collection_name="crossref"
)

# Dataset schema
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=False
)

# Add the fields to the schema
# TODO: Find optimal max length for varchar
schema.add_field(field_name="DOI", datatype=DataType.VARCHAR, max_length=256, is_primary=True)

schema.add_field(field_name="vector", datatype=DataType.BINARY_VECTOR, dim=1024)

schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=2048)
schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="abstract", datatype=DataType.VARCHAR, max_length=4096)
schema.add_field(field_name="month", datatype=DataType.VARCHAR, max_length=16)
schema.add_field(field_name="year", datatype=DataType.INT64, max_length=8, is_clustering_key=True)
schema.add_field(field_name="URL", datatype=DataType.VARCHAR, max_length=256)

print("Issues with schema: ", schema.verify())

# Create a collection
client.create_collection(
    collection_name="crossref",
    schema=schema,
    properties={ "mmap.enabled": "true" }
)

################################################################################
# Create index

# Set up the index parameters
index_params = MilvusClient.prepare_index_params()

# Add an index on the vector field.
index_params.add_index(
        field_name="vector",
        metric_type="HAMMING",
        index_type="BIN_IVF_FLAT",
        index_name="vector_index",
        params={ "nlist": 128}
    )

print("Creating Index file.")

# Create an index file
res = client.create_index(
    collection_name="crossref",
    index_params=index_params,
    sync=True # Wait for index creation to complete before returning. 
)

print(res)

print("Listing indexes.")

# List indexes
res = client.list_indexes(
    collection_name="crossref"
)

print(res)

print("Describing Index.")

# Describe index
res = client.describe_index(
    collection_name="crossref",
    index_name="vector_index"
)

print(res)

################################################################################

# Load the collection

print("Loading Collection")

client.load_collection(
    collection_name="crossref",
    replica_number=1 # Number of replicas to create on query nodes. 
)

res = client.get_load_state(
    collection_name="crossref"
)

print("Collection load state:")
print(res)

################################################################################ 
# Create import job
# https://milvus.io/docs/import-data.md

# Define the API endpoint
job_url = "http://localhost:19530/v2/vectordb/jobs/import/create"

# Define the headers
headers = {
    "Content-Type": "application/json"
}

# Define the data payload
job_url_data = {
    "files": files,
    "collectionName": "crossref"
}

# Make the POST request
job_response = requests.post(job_url, headers=headers, json=job_url_data)
job_json = job_response.json()

# Print the response
print("Job details:")
print(job_response.status_code)
print(job_json)

# Extract jobId
job_id = job_json['data']['jobId']

# Periodically check on import status
progress_url = "http://localhost:19530/v2/vectordb/jobs/import/get_progress"

progress_url_data = {
    "jobId": f"{job_id}"
}

while True:

    print('*'*80)

    # Sleep a bit
    seconds = 10
    print(f"Sleeping for {seconds} seconds")
    sleep(seconds)

    
    # Make the POST request
    progress_response = requests.post(progress_url, headers=headers, json=progress_url_data)

    progress_json = progress_response.json()
    # print(progress_json)

    progress_percent = progress_json['data']['progress']
    progress_state = progress_json['data']['state']

    if progress_state == 'Pending' or progress_state == 'Importing':

        print(f"Job: {progress_state}.")
        print(f"Finised: {progress_percent}%.")

    elif progress_state == 'Completed':

        print(f"Job: {progress_state}.")
        print(f"Imported {progress_json['data']['totalRows']} rows.")

        break

    elif progress_state == 'Failed':

        print(f"Job: {progress_state}.")
        print(progress_json)

        print("Exiting...")
        exit()

################################################################################



