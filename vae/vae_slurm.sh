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

# Set up WandB environment variables for authentication, project, and entity
export WANDB_API_KEY="ae52e426af96ba6657d73e9829e28ac8891914d6"
export WANDB_PROJECT="vq_vae_sem_2"
export WANDB_ENTITY="deya-03-the-university-of-manchester"


export WANDB_CACHE_DIR="/share/nas2_3/adey/wandb_cache"
export WANDB_DATA_DIR="/share/nas2_3/adey/astro/wandb_data/"


echo ">> Starting setup"
ulimit -n 65536

# Activate the virtual environment
source /share/nas2_3/adey/.venv/bin/activate
echo ">> Environment activated"

# Verify Python version
python --version
/share/nas2_3/adey/.venv/bin/python --version

# Define paths for the Python script and the YAML sweep configuration file
PYTHON_SCRIPT="/share/nas2_3/adey/astro/clean_code/testing_training.py"
SWEEP_CONFIG="/share/nas2_3/adey/astro/clean_code/config_vae_wandb.yaml"

# Initialize the sweep and capture the output
temp_file=$(mktemp)
wandb sweep "$SWEEP_CONFIG" > "$temp_file" 2>&1

# Extract the full sweep path from the temporary file
SWEEP_PATH=$(grep -o 'deya-03-the-university-of-manchester/vq_vae_sem_2/[^ ]*' "$temp_file" | tail -n 1)

# Remove temporary file
rm "$temp_file"

# Check if SWEEP_PATH was successfully created
if [[ -z "$SWEEP_PATH" ]]; then
  echo "Error: Sweep ID not found. Please check your configuration."
   # Show full output for debugging
  exit 1
fi

# Run the sweep agent for the specified Sweep ID
echo ">> Starting sweep agent for Sweep Path: $SWEEP_PATH"
wandb agent "$SWEEP_PATH"
