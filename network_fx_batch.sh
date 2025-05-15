#!/bin/bash
#SBATCH --job-name=nfx_extract
#SBATCH --output=logs/%A_%a.out          # %A = jobID, %a = array index
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=830-1829

# Check if BATCH_START is set, if not, set it to 0
: ${BATCH_START:=0}

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Processing batch starting at index: $BATCH_START"

module load python/3.9.0
source $HOME/venvs/metascience_env/bin/activate

# Debug info (optional, can comment out after confirming)
python --version
which python
which pip
pip show numpy
pip show keras

# array of all pickle files
FILES=($(ls /scratch/users/averylou/Sims2500New/*.pkl))

# Calculate the actual index for this job
ACTUAL_INDEX=$((BATCH_START + SLURM_ARRAY_TASK_ID))

# file for this job array task
FILE_TO_PROCESS=${FILES[$ACTUAL_INDEX]}

echo "Processing file: $FILE_TO_PROCESS"

# Run your Python script
python $HOME/projects/metascience/network_fx_extraction.py "$FILE_TO_PROCESS"

deactivate

echo "Job ended at $(date)"