#!/usr/bin/env bash

# check prerequisites
command -v conda >/dev/null 2>&1 || { echo >&2 "conda not found. Please refer to the README and install Miniconda."; exit 1; }
command -v git >/dev/null 2>&1 || { echo >&2 "git not found. Please refer to the README and install Git."; exit 1; }

source scripts/settings.sh

source $(conda info --base)/etc/profile.d/conda.sh
conda create -y -n $CONDA_ENV_NAME python=3.7
conda activate $CONDA_ENV_NAME

conda install -y pytorch==1.6.0 cuda100 -c pytorch
conda install -y python-blosc==1.7.0 -c conda-forge

# THAD
rm -rf thad 2> /dev/null
git clone https://github.com/pkhungurn/talking-head-anime-demo thad

pip install -r requirements.txt
