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

# Set environment variables
export EXP_ENV_DIR=./envs

# Log start of simulation
echo "Starting simulation at $(date)"

# Ensure git repository is up to date
echo "Updating git repository..."
git fetch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"
git pull origin $CURRENT_BRANCH
echo "Git repository updated to latest commit"

# Run the simulation
python3 -m simulate

# Log completion
echo "Simulation completed at $(date)"