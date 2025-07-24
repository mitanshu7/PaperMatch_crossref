When logging into vast.ai,
pytorch 2.2 works with rtx 4090 on runpod. 

update uv from website, 
create venv

login to hf using cli
add token to env AND batch size in env

AFTER testing for batch size, on sample dataset
, allow_patterns='part_1.parquet'
remove the slicing to start full script

```bash
nohup uv run embed_multigpu_split.py > output.log 2>&1 &
```

Log:

wasted dollars on runpod spot, got part1 and 10 done

wasted more dollars on vastai spot, got outbid so much

booked 8x5090 at $2.88/hr, 
uv torch bad, installed from https://pytorch.org/get-started/locally/ using

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install datasets pandas pyarrow python-dotenv sentence-transformers
```

and 

```bash
nohup python3 embed_multigpu_split.py > output.log 2>&1 &
```

19221