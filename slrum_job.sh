#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err
#SBATCH --partition=learnlab
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -C volta32gb

# Create logs directory if it doesn't exist

# Set environment variables
export EXP_ENV_DIR=./envs

# Log start of simulation
echo "Starting simulation at $(date)"

# Run the simulation
python3 -m simulate

# Log completion
echo "Simulation completed at $(date)"
