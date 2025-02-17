#!/bin/bash

#SBATCH --job-name=testing_slurm
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=1500GB
#SBATCH --output=/share/nas2_3/adey/w_c_21_Oct/test_output/.out/vqvae_train_%j.log
#SBATCH --error=/share/nas2_3/adey/w_c_21_Oct/test_output/.err/vqvae_train_%j.err

# Display GPU status
nvidia-smi

# Increase the limit for open file descriptors
ulimit -n 65536

# Set up WandB environment variables
export WANDB_API_KEY="ae52e426af96ba6657d73e9829e28ac8891914d6"
export WANDB_PROJECT="vq_vae_sem_2"
export WANDB_ENTITY="deya-03-the-university-of-manchester"
export WANDB_CACHE_DIR="/share/nas2_3/adey/wandb_cache"
export WANDB_DATA_DIR="/share/nas2_3/adey/astro/wandb_data/"

echo ">> Starting setup"

# Activate the virtual environment
source /share/nas2_3/adey/.venv/bin/activate
echo ">> Environment activated"

# Verify Python version
python --version
/share/nas2_3/adey/.venv/bin/python --version

# Define script and configuration file paths
PYTHON_SCRIPT="/share/nas2_3/adey/astro/clean_code/Neural-Compression-for-Radio-Astronomy/vae/main_vae.py"
SWEEP_CONFIG="/share/nas2_3/adey/astro/clean_code/Neural-Compression-for-Radio-Astronomy/vae/vae_config.yaml"

# Initialize the WandB sweep and capture output
temp_file=$(mktemp)
wandb sweep "$SWEEP_CONFIG" > "$temp_file" 2>&1

# Extract the Sweep ID from the output
SWEEP_PATH=$(grep -o 'deya-03-the-university-of-manchester/vq_vae_sem_2/[^ ]*' "$temp_file" | tail -n 1)

# Remove the temporary file
rm "$temp_file"

# Check if Sweep ID was successfully retrieved
if [[ -z "$SWEEP_PATH" ]]; then
    echo "Error: Sweep ID not found. Please check your configuration."
    exit 1
fi

# Start the WandB sweep agent
echo ">> Starting sweep agent for Sweep Path: $SWEEP_PATH"
wandb agent "$SWEEP_PATH"
