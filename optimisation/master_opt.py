import os
import time
import subprocess
import pandas as pd
import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties

# --- CONFIG ---
N_ROUNDS = 20           # How many opt batches to run
TRIALS_PER_BATCH = 40   # How many distinct param sets to try per round
SEEDS_PER_TRIAL = 10     # How many times to repeat each set (to avg noise!!)


CSV_FILE = "current_batch.csv"
JSON_STATE = "experiment_state.json"
RESULTS_DIR = "./results"

def setup_ax():
    if os.path.exists(JSON_STATE):
        print(f"Resuming experiment from {JSON_STATE}...")
        return AxClient.load_from_json_file(JSON_STATE)
    
    print("Starting new experiment...")
    client = AxClient()
    client.create_experiment(
        name="science_optimization",
        parameters=[
            {"name": "temp", "type": "range", "bounds": [0.1, 10.0], "log_scale": True}, 
            {"name": "bias", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "swaps", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "capacity", "type": "range", "bounds": [2, 10]},
        ],
        objectives={
            "subjective_loss": ObjectiveProperties(minimize=True), 
        },
    )
    return client

def submit_slurm_job(num_jobs):
    """Submits the worker.sh script as an array job."""
    # use --parsable to get just the job ID back
    cmd = ["sbatch", "--parsable", f"--array=0-{num_jobs-1}", "worker.sh"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Sbatch failed: {result.stderr}")
    
    job_id = result.stdout.strip().split(';')[0] # array id formats
    print(f"Submitted batch job {job_id} ({num_jobs} tasks)")
    return job_id

def wait_for_job(job_id):
    """Polls squeue every 180 seconds until the job is gone."""
    print(f"Waiting for job {job_id} to finish...")
    while True:
        # Check if job is still in queue
        cmd = ["squeue", "-j", job_id, "-h"] 
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if not result.stdout.strip():
            print("Batch complete.")
            break
        
        time.sleep(180)

def process_results(client, batch_mapping):
    """Reads output files, averages seeds, and updates Ax."""
    for trial_index, seeds_info in batch_mapping.items():
        # seeds_info is a list of (seed, row_id)
        scores = []
        
        for seed, row_id in seeds_info:
            result_file = os.path.join(RESULTS_DIR, f"{row_id}.txt")
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        # RESULT: <subj>, <obj>
                        line = f.read().strip()
                        if line.startswith("RESULT:"):
                            parts = line.replace("RESULT:", "").split(",")
                            scores.append(float(parts[0])) # Subjective loss
                except Exception as e:
                    print(f"Error reading {result_file}: {e}")
        
        if scores:
            mean_score = np.mean(scores)
            sem_score = np.std(scores) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
            
            try:
                client.complete_trial(
                    trial_index=trial_index, 
                    raw_data={"subjective_loss": (mean_score, sem_score)}
                )
                print(f"Trial {trial_index}: Completed with mean loss {mean_score:.4f}")
            except Exception as e:
                print(f"Could not complete trial {trial_index}: {e}")
        else:
            print(f"Trial {trial_index}: FAILED (No results found). Abandoning.")
            client.abandon_trial(trial_index)

def main():
    client = setup_ax()
    
    for round_id in range(N_ROUNDS):
        print(f"\n=== STARTING ROUND {round_id + 1}/{N_ROUNDS} ===")
        
        # 1. Generate new parameters
        # We define a "batch_mapping" to track which CSV rows belong to which Ax trial
        batch_mapping = {} # {ax_trial_index: [(seed, csv_row_id), ...]}
        csv_rows = []
        
        # Generate N distinct parameter sets
        for _ in range(TRIALS_PER_BATCH):
            params, trial_index = client.get_next_trial()
            
            # Repeat M times for robustness (seeds)
            for seed in range(SEEDS_PER_TRIAL):
                row_id = len(csv_rows) # incremental ID
                
                # Data for CSV
                row = params.copy()
                row['seed'] = seed
                row['trial_index'] = trial_index # Ax ID
                row['row_id'] = row_id           # Slurm Array ID
                csv_rows.append(row)
                
                # Track for aggregation later
                if trial_index not in batch_mapping:
                    batch_mapping[trial_index] = []
                batch_mapping[trial_index].append((seed, row_id))

        # 2. Write Batch CSV
        df = pd.DataFrame(csv_rows)
        df.to_csv(CSV_FILE, index=False)
        print(f"Generated {len(df)} tasks for this batch.")
        
        # 3. Clean previous results??
        # os.system(f"rm {RESULTS_DIR}/*.txt") 
        
        # 4. Submit & Wait
        job_id = submit_slurm_job(len(df))
        wait_for_job(job_id)
        
        # 5. Process & Update
        process_results(client, batch_mapping)
        
        # 6. Save State
        client.save_to_json_file(JSON_STATE)
        print("Experiment state saved.")

if __name__ == "__main__":
    main()