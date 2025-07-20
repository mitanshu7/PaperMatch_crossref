from datasets import load_dataset
from time import time

start_time = time()
################################################################################

data_files = {'train': '/home/mitanshu/Downloads/March 2025 Public Data File from Crossref/*.jsonl.gz'}

dataset = load_dataset('json', data_files=data_files, split='train', streaming=True)

# Rows must have abstract to embed
dataset = dataset.filter(lambda row: bool(row.get('DOI')) and bool(row.get('abstract')) and bool(row.get('title')) and bool(row.get('author')) and bool(row.get('URL')) and bool(row.get('created')))

# Columns to save
need = ['DOI', 'abstract', 'title', 'author', 'URL', 'created']
dataset = dataset.select_columns(need)

# Process entries
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

################################################################################

# Map the metadata
dataset = dataset.map(prepare_metadata, remove_columns=['created'])

# dataset = dataset.rename_column("DOI", "id")

# Save it
dataset.to_parquet("crossref_metadata.parquet")
################################################################################

end_time = time()
print(f"Time taken: {(end_time - start_time)/3600} hours")