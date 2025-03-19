import pandas as pd
import numpy as np

# Load data
rl_data = pd.read_csv("filtered_vehicle_states_RL.csv")
kf_data = pd.read_csv("filtered_vehicle_states.csv")
rl2_data = pd.read_csv("filtered_vehicle_states_RL2.csv")

def compare_data(df):
    results = {}
    for col in df.columns:
        if col.endswith("_original"):
            prefix = col[:-len("_original")]
            noisy_col = f"{prefix}_noisy"
            filtered_col = f"{prefix}_filtered"
            if noisy_col in df.columns and filtered_col in df.columns:
                noisy_diff = (df[noisy_col] - df[col]).abs().mean()
                filtered_diff = (df[filtered_col] - df[col]).abs().mean()
                
                if noisy_diff != 0:
                    percent_reduction = (1 - (filtered_diff / noisy_diff)) * 100  # % reduction in noise
                else:
                    percent_reduction = None
                
                results[prefix] = {"noisy_diff": noisy_diff, "filtered_diff": filtered_diff, "percent_reduction": percent_reduction}
                
    return results

# Compute differences for all three datasets
rl_results = compare_data(rl_data)
kf_results = compare_data(kf_data)
rl2_results = compare_data(rl2_data)

# Convert to DataFrames
rl_df = pd.DataFrame.from_dict(rl_results, orient="index")
kf_df = pd.DataFrame.from_dict(kf_results, orient="index")
rl2_df = pd.DataFrame.from_dict(rl2_results, orient="index")

# Min-Max Normalization Function
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Apply normalization
rl_normalized = rl_df.apply(min_max_normalize)
kf_normalized = kf_df.apply(min_max_normalize)
rl2_normalized = rl2_df.apply(min_max_normalize)

# Compute average percentage noise reduction for all three methods
kf_avg_reduction = kf_df["percent_reduction"].mean()
rl_avg_reduction = rl_df["percent_reduction"].mean()
rl2_avg_reduction = rl2_df["percent_reduction"].mean()

# Compute improvement percentages over KF
rl_improvement = (rl_avg_reduction - kf_avg_reduction) / kf_avg_reduction * 100 if kf_avg_reduction != 0 else None
rl2_improvement = (rl2_avg_reduction - kf_avg_reduction) / kf_avg_reduction * 100 if kf_avg_reduction != 0 else None

print(rl_df)


print(f"Average Noise Reduction (KF Only): {kf_avg_reduction:.2f}%")
print(f"Average Noise Reduction (RL + KF): {rl_avg_reduction:.2f}%")
print(f"Average Noise Reduction (RL2 + KF): {rl2_avg_reduction:.2f}%")

print(f"Improvement of RL + KF over KF: {rl_improvement:.2f}%")
print(f"Improvement of RL2 + KF over KF: {rl2_improvement:.2f}%")
