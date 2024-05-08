#!/usr/bin/bash
#SBATCH --account=pi-dfreedman
#SBATCH -p schmidt-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=schmidt
#SBATCH --time 2:00:00

module load python/miniforge-24.1.2 # python 3.10

echo "output of the visible GPU environment"
nvidia-smi

source /project/dfreedman/hackathon/hackathon-env/bin/activate
python cross_validate.py
