#!/bin/bash
#SBATCH --job-name=nfx_extract
#SBATCH --output=logs/%A_%a.out          # %A = jobID, %a = array index
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-999

# Check if BATCH_START is set, if not, set it to 0
: ${BATCH_START:=0}

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Processing batch starting at index: $BATCH_START"

VENV_HOME=$HOME/venvs/metascience_env
source "$VENV_HOME/bin/activate"

# Install packages (only if not already installed)
if [ ! -f "$VENV_DIR/.packages_installed" ]; then
    pip install --upgrade pip
    pip install numpy scipy networkx python-louvain scikit-learn pandas
    pip install tensorflow==2.13.1
    pip install keras==2.13.1
    touch "$VENV_DIR/.packages_installed"
    echo "Packages installed"
else
    echo "Packages already installed, skipping installation"
fi

# array of all pickle files
FILES=($(ls /scratch/users/averylou/Sims2500New/*.pkl))

# Calculate the actual index for this job
ACTUAL_INDEX=$((BATCH_START + SLURM_ARRAY_TASK_ID))

# file for this job array task
FILE_TO_PROCESS=${FILES[$ACTUAL_INDEX]}

echo "Processing file: $FILE_TO_PROCESS"

# Run your Python script
python $HOME/Projects/thesis/metascience/network_fx_extraction.py "$FILE_TO_PROCESS"

deactivate

echo "Job ended at $(date)"