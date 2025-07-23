When logging into vast.ai,
pytorch 2.2 works with rtx 4090 on runpod. 

update uv from website, 

login to hf using cli
add token to env
Change batch size in env

AFTER testing for batch size, on sample dataset
, allow_patterns='part_1.parquet'
remove the slicing to start full script

nohup uv run embed_multigpu_split.py > output.log 2>&1 &
