from datasets import load_dataset
from time import time
from multiprocessing import Pool, cpu_count
from glob import glob
import os

################################################################################

# Gather the jsonl's
data_files = glob('/home/mitanshu/Downloads/Sample Dataset March 2025 Public Data File from Crossref/*.jsonl.gz')
data_files.sort()

# Save here
processed_folder = 'crossref_metadata'
os.makedirs(processed_folder, exist_ok=True)

# Process entries
def prepare_data(filename):
    
    # Load the file
    dataset = load_dataset('json', data_files=filename, split='train', streaming=False)
    
    # Filter needs
    dataset = dataset.filter(lambda row: (bool(row.get('DOI')) and bool(row.get('abstract')) and bool(row.get('title')) and bool(row.get('author')) and bool(row.get('URL')) and bool(row.get('created'))) )
    
    try:
        # Columns we care about
        need = ['DOI', 'abstract', 'title', 'author', 'URL', 'created']
        dataset = dataset.select_columns(need)
        return dataset
    except Exception as e:
        print(f"Error {e} for filename {filename}")
        return None
    
def prepare_metadata(row):
    # Get title text from the list
    row['title'] = row['title'][0]
    
    # Get author names
    authors = []
    for author in row['author']:
        
        # Extract information
        family = author.get('family')
        given = author.get('given')
        name = author.get('name')
        
        # Make up the name from its constituents
        full_name = str(family or "") + ' ' + str(given or "") + ' ' + str(name or "")
        
        # Add to list
        authors.append(full_name)
        
    row['author'] = authors

    # Add month and year
    timestamp = row['created']['date-time']
    
    row['year'] = int(timestamp.strftime('%Y'))
    row['month'] = timestamp.strftime('%B')
    
    return row
    
def process_file(filename):
    
    # make file usable
    dataset = prepare_data(filename)
    
    if dataset is not None:
        
        # Extract info
        dataset = dataset.map(prepare_metadata, remove_columns=['created'])
        
        # Save
        filename = f"{processed_folder}/{os.path.basename(filename).replace('.jsonl.gz', '.parquet')}"
        dataset.to_parquet(filename)
    
        return filename
    else:
        return None

################################################################################

if __name__ == '__main__':
    
    start_time = time()
    
    with Pool(cpu_count()) as pool:
        pool.map(process_file, data_files)
        
    end_time = time()
    print(f"Time taken: {(end_time - start_time)/3600} hours")

################################################################################
