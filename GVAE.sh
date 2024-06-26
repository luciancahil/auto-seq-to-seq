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

cd $SLURM_SUBMIT_DIR
echo "Hello!"
module load conda
echo "Hello1!"
conda init bash
echo "Hello2!"
module load gcc
echo "Hello3!"
module load openmpi
echo "Hello4!"
module load git
echo "Hello5!"
source ~/.bashrc
echo "Hello6!"
conda activate gvae
echo "Hello6!"
python main.py
conda deactivate

