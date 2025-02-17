#!/bin/bash

#SBATCH --job-name=testing_slurm
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2_3/adey/w_c_21_Oct/test_output/.out/vqvae_train_%j.log
#SBATCH --error=/share/nas2_3/adey/w_c_21_Oct/test_output/.err/vqvae_train_%j.err
#SBATCH --mem=1500GB

# Display GPU status
nvidia-smi

# Activate the virtual environment
source /share/nas2_3/adey/.venv/bin/activate
echo ">> Environment activated"

# Define the Python script file
PYTHON_SCRIPT="/share/nas2_3/adey/w_c_21_Oct/test_output/check_cuda.py"

# Write a Python script that checks CUDA memory
cat <<EOL > $PYTHON_SCRIPT
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert bytes to GB
    print(f"CUDA device detected: {torch.cuda.get_device_name(device)}")
    print(f"Total available memory: {total_memory:.2f} GB")
else:
    print("No CUDA device found.")
EOL

# Run the Python script
python $PYTHON_SCRIPT
