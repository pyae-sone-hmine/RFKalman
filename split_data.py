import pandas as pd

# File paths
sample_headers_path = './sample_dynamics.csv'
full_data_path = './full_dataset.csv'

def process_and_split_csv(sample_headers_path, full_data_path):
    # Read the sample file to get headers
    sample_df = pd.read_csv(sample_headers_path, header=0)
    headers = list(sample_df.columns)
    
    # Read the full dataset without headers
    full_df = pd.read_csv(full_data_path, header=None)
    full_df.columns = headers  # Assign headers from sample dataset
    
    # Ensure indexing for splitting
    total_rows = len(full_df)
    train_indices = []
    test_indices = []
    
    # Splitting 80/20 in batches of 10
    for i in range(0, total_rows, 10):
        train_indices.extend(range(i, min(i+8, total_rows)))  # 8 points for training
        test_indices.extend(range(min(i+8, total_rows), min(i+10, total_rows)))  # 2 points for testing
    
    # Create train and test datasets
    train_df = full_df.iloc[train_indices]
    test_df = full_df.iloc[test_indices]
    
    # Save to new CSV files
    train_file = './train_data.csv'
    test_file = './test_data.csv'
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    return train_file, test_file

# Run the function
train_file, test_file = process_and_split_csv(sample_headers_path, full_data_path)

# Provide file paths for download
train_file, test_file
