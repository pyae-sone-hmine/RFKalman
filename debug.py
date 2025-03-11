import pandas as pd

rl_data = pd.read_csv("filtered_vehicle_states_RL.csv")
kf_data = pd.read_csv("filtered_vehicle_states.csv")

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
                # Avoid division by zero by checking noisy_diff
                if noisy_diff != 0:
                    percent = (filtered_diff / noisy_diff) * 100
                else:
                    percent = None
                results[prefix] = percent
                
    return results

rl_results = compare_data(rl_data)
kf_results = compare_data(kf_data)

print("RL RESULTS:", rl_results)
print("KF RESULTS:", kf_results)

def average_results(results):
    valid_results = [value for value in results.values() if value is not None]
    if valid_results:
        return sum(valid_results) / len(valid_results)
    else:
        return None

rl_average = average_results(rl_results)
kf_average = average_results(kf_results)

print("Average RL Result:", rl_average)
print("Average KF Result:", kf_average)



