from datasets import load_dataset
from bs4 import BeautifulSoup

################################################################################

# data_files = {'train': '/home/mitanshu/Downloads/March 2025 Public Data File from Crossref/*.jsonl.gz'}
data_files = {'train': '/home/mitanshu/Downloads/Sample Dataset March 2025 Public Data File from Crossref/*.jsonl.gz'}

dataset = load_dataset('json', data_files=data_files, split='train', streaming=True)

# print(dataset)


# Columns to save
need = ['DOI', 'abstract', 'title', 'author', 'URL', 'created']
dataset = dataset.select_columns(need)

# Rows must have abstract to embed
dataset = dataset.filter(lambda row: bool(row.get('abstract')) )

# Process entries
def prepare_metadata(row):
    
    # Extract abstract save in html format
    soup = BeautifulSoup(row['abstract'], 'html.parser')
    row['abstract'] = soup.find('jats:p').get_text()
    
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
    # timestamp = datetime.strptime(row['created']['date-time'], '%Y-%m-%dT%H:%M:%SZ')
    timestamp = row['created']['date-time']
    
    row['year'] = int(timestamp.strftime('%Y'))
    row['month'] = timestamp.strftime('%B')
    
    return row

################################################################################

# Map the metadata
dataset = dataset.map(prepare_metadata, remove_columns=['created'])

# print(list(next(iter(dataset)).keys()))
# print(next(iter(dataset)))
# print(dataset)

# Save it
dataset.to_parquet("/home/mitanshu/Downloads/tmp/crossref_metadata.parquet", index=False)
