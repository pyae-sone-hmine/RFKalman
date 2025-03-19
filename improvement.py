import pandas as pd

# List your CSV file paths. Adjust these paths as needed.
csv_files = ['filtered_vehicle_states.csv', 'filtered_vehicle_states_RL.csv', 'filtered_vehicle_states_RL2.csv']

# Define the parameters you want to analyze (without the suffixes)
params = ['obs_s', 'obs_d', 'obs_mu', 'obs_speed', 'obs_kappa', 'ego_s', 'ego_d', 'ego_mu', 'ego_speed', 'ego_steering', 'ego_kappa']  # add more as needed

# Dictionary to hold the results
results = {}

for file in csv_files:
    df = pd.read_csv(file)
    file_results = {}
    
    for param in params:
        # Construct column names based on your naming scheme
        original_col = f'{param}_original'   # ground truth
        noisy_col = f'{param}_noisy'           # noisy measurement
        filtered_col = f'{param}_filtered'     # filtered measurement
        
        # Check if all expected columns are present
        if all(col in df.columns for col in [original_col, noisy_col, filtered_col]):
            # Calculate absolute error compared to ground truth for both noisy and filtered data
            df['error_noisy'] = (df[noisy_col] - df[original_col]).abs()
            df['error_filtered'] = (df[filtered_col] - df[original_col]).abs()
            
            avg_error_noisy = df['error_noisy'].mean()
            avg_error_filtered = df['error_filtered'].mean()
            
            # Calculate the improvement percentage if the noisy error is not zero
            if avg_error_noisy != 0:
                improvement = ((avg_error_noisy - avg_error_filtered) / avg_error_noisy) * 100
            else:
                improvement = 0
            
            file_results[param] = {
                'avg_error_noisy': avg_error_noisy,
                'avg_error_filtered': avg_error_filtered,
                'improvement_%': improvement
            }
        else:
            print(f"Missing expected columns for parameter {param} in file {file}.")
    
    results[file] = file_results

# Display the results for each file and parameter
for file, res in results.items():
    print(f"\nResults for {file}:")
    for param, metrics in res.items():
        print(f"  {param}: {metrics}")
