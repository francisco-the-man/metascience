# Combine into one large csv

import os
import glob
import pandas as pd

# Directory containing your processed CSVs
input_dir = os.path.expanduser('~/projects/metascience/processed_files')
pattern = os.path.join(input_dir, '*_processed_clust.csv')
files = glob.glob(pattern)

dfs = []
for file in files:
    # Extract trial number from filename
    basename = os.path.basename(file)
    # Assumes filename like simulation_trial_1728_processed_clust.csv
    try:
        trial_num = int(basename.split('_')[2])
    except (IndexError, ValueError):
        continue  # skip files that don't match pattern

    df = pd.read_csv(file)
    df.insert(0, 'trial_num', trial_num)
    dfs.append(df)

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(os.path.join(input_dir, 'all_nfx_combined.csv'), index=False)
    print(f"Combined {len(dfs)} files into all_nfx_combined.csv")
else:
    print("No files found or matched the pattern.")
