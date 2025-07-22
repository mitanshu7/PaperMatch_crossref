#!/bin/bash

# Install required packages
apt update
apt install nano htop nvtop -y

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repo
git clone https://github.com/mitanshu7/PaperMatch_crossref.git

# Install dependencies
cd PaperMatch_crossref/
uv sync

