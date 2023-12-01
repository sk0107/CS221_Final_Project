#!/bin/bash
#SBATCH --job-name=dqn_agent_run
#SBATCH --output=dqn_output_%j.txt
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --partition=oat

# Load Python 3.11.5 environment
module load python/3.11.5  # Adjust this line as per your system's module configuration

# Alternatively, if you have a conda environment:
# module load anaconda3/2022.05  # or appropriate module for anaconda
# conda activate myenv  # Replace 'myenv' with your conda environment name

# Install requirements (assuming requirements.txt is in the current directory)
pip install -r requirements.txt

# Run the script
python dqn_agent.py
