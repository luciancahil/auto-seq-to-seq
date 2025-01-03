#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --job-name=GVAE
#SBATCH --account=pr-jreid03-1-gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=royhe@student.ubc.ca
#SBATCH --output=output_GVAE.txt
#SBATCH --error=output_GVAE.txt

conda activate gvae
python main.py
conda deactivate

