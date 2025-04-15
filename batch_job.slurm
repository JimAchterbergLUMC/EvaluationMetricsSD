#!/bin/bash
# add job parameters
#SBATCH -J evalSD
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu,highmemgpu
#SBATCH --time=100:00:00 
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=j.l.achterberg@lumc.nl

# add necessary modules
module purge 
module add tools/miniconda/python3.10/23.3.1
module add library/cuda/12.2.2/gcc.8.5.0
module add library/cudnn/12.2/cudnn

# end if anything fails
set -e

# initialize conda
# source $(conda info --base)/etc/profile.d/conda.sh

# create a conda environment
rm -rf /tmp/jlenv
mkdir /tmp/jlenv
chmod 700 /tmp/jlenv
conda create --prefix /tmp/jlenv/conda_env python=3.10 -y
conda activate /tmp/jlenv/conda_env
# install packages
pip install -r requirements.txt

# run script
python3 tabular_benchmark.py

# cleanup environment
conda deactivate
rm -rf /tmp/jlenv
module purge