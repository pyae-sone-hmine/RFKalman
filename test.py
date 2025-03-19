import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from kfilter import KalmanFilter

#### filepath: /home/gridsan/phmine/rf_kalman/test.py
def denoise_vehicle_states_with_rl(states_csv):

    states = pd.read_csv(states_csv)
    
    # Get original states (using the same columns as in the classic version)
    originals_obstacle = states[['Agent2 Ego Dynamics s', 'Agent2 Ego Dynamics d', 
                                 'Agent2 Ego Dynamics mu', 'Agent2 Ego Dynamics speed', 
                                 'Agent2 Ego Dynamics steering', 'Agent2 Ego Dynamics kappa']].values
    originals_ego = states[['Agent1 Ego Dynamics s', 'Agent1 Ego Dynamics d', 
                            'Agent1 Ego Dynamics mu', 'Agent1 Ego Dynamics speed', 
                            'Agent1 Ego Dynamics steering', 'Agent1 Ego Dynamics kappa']].values
    original_df = pd.DataFrame(
        np.hstack((originals_obstacle, originals_ego)),
        columns=[
            'obs_s_original', 'obs_d_original', 'obs_mu_original', 'obs_speed_original', 'obs_steering_original', 'obs_kappa_original',
            'ego_s_original', 'ego_d_original', 'ego_mu_original', 'ego_speed_original', 'ego_steering_original', 'ego_kappa_original'
        ]
    )
    
    # Add noise using identical parameters as the classic function
    noisy_obstacle_states = originals_obstacle + np.random.normal(0, np.abs(originals_obstacle/4), originals_obstacle.shape)
    noisy_ego_states = originals_ego + np.random.normal(0, np.abs(originals_ego/4), originals_ego.shape)
    noisy_df = pd.DataFrame(
        np.hstack((noisy_obstacle_states, noisy_ego_states)),
        columns=[
            'obs_s_noisy', 'obs_d_noisy', 'obs_mu_noisy', 'obs_speed_noisy', 'obs_steering_noisy', 'obs_kappa_noisy',
            'ego_s_noisy', 'ego_d_noisy', 'ego_mu_noisy', 'ego_speed_noisy', 'ego_steering_noisy', 'ego_kappa_noisy'
        ]
    )
    
    # Prepare to perform filtering with RL dynamic Q/R
    state_dim = noisy_obstacle_states.shape[1] + noisy_ego_states.shape[1]
    control_dim = 2  
    kf = KalmanFilter(dim_x=state_dim, dim_z=state_dim, dim_u=control_dim)
    
    filtered_states = []
    q_scales = []
    r_scales = []
    
    # Apply filtering per sample
    for i in range(len(noisy_obstacle_states)):
        measurement_vector = np.concatenate((noisy_obstacle_states[i], noisy_ego_states[i]))
        
        # Obtain dynamic noise scales via the RL model
        obs = np.concatenate((kf.x.flatten(), np.zeros(1)))
        q_scale, r_scale = model.predict(obs, deterministic=True)[0]
        q_scales.append(q_scale)
        r_scales.append(r_scale)
        
        kf.Q = np.eye(state_dim) * q_scale
        kf.R = np.eye(state_dim) * r_scale

        # Use control inputs from the CSV
        control_vector = np.array([
            states['Agent1 Ego Dynamics tire velocity (action)'][i],
            states['Agent1 Ego Dynamics Acceleration (action)'][i]
        ])
        kf.predict(u=control_vector)
        kf.update(measurement_vector)
        filtered_states.append(kf.x.flatten())
    
    filtered_df = pd.DataFrame(
        filtered_states, 
        columns=[
            'obs_s_filtered', 'obs_d_filtered', 'obs_mu_filtered', 'obs_speed_filtered', 'obs_steering_filtered', 'obs_kappa_filtered',
            'ego_s_filtered', 'ego_d_filtered', 'ego_mu_filtered', 'ego_speed_filtered', 'ego_steering_filtered', 'ego_kappa_filtered'
        ]
    )
    
    # Add Q and R scale values as columns
    filtered_df['Q_scale'] = q_scales
    filtered_df['R_scale'] = r_scales
    
    return pd.concat([original_df, noisy_df, filtered_df], axis=1)

def compare_errors(states_csv, filtered_states_df):
    """
    Compares overall error (Euclidean norm) of the RL-based filtered states
    and the RL-based noisy measurements with the ground truth.
    """
    states = pd.read_csv(states_csv)
    ground_truth = states.iloc[:, :12].values  # ground truth from CSV

    # Extract only the '_filtered' and '_noisy' columns from the concatenated DataFrame
    filtered = filtered_states_df.filter(like='_filtered').values
    noisy = filtered_states_df.filter(like='_noisy').values
    
    filtered_errors = np.linalg.norm(filtered - ground_truth, axis=1)
    noisy_errors = np.linalg.norm(noisy - ground_truth, axis=1)
    
    print("Average filtered error (RL):", np.mean(filtered_errors))
    print("Average noisy error (RL):", np.mean(noisy_errors))
    
    return filtered_errors, noisy_errors

def denoise_vehicle_states(states):
    # Select obstacle car states
    obstacle_states = states[['Agent2 Ego Dynamics s', 'Agent2 Ego Dynamics d', 
                              'Agent2 Ego Dynamics mu', 'Agent2 Ego Dynamics speed', 
                              'Agent2 Ego Dynamics steering', 'Agent2 Ego Dynamics kappa']].values
    
    # Select ego car states
    ego_states = states[['Agent1 Ego Dynamics s', 'Agent1 Ego Dynamics d', 
                         'Agent1 Ego Dynamics mu', 'Agent1 Ego Dynamics speed', 
                         'Agent1 Ego Dynamics steering', 'Agent1 Ego Dynamics kappa']].values
    
    # Store original values
    originals_obstacle = obstacle_states.copy()
    originals_ego = ego_states.copy()
    
    # Add noise to the obstacle and ego states
    noisy_obstacle_states = obstacle_states + np.random.normal(0, np.abs(obstacle_states / 4), obstacle_states.shape)
    noisy_ego_states = ego_states + np.random.normal(0, np.abs(ego_states / 4), ego_states.shape)

    # Define dimensions
    state_dim = obstacle_states.shape[1] + ego_states.shape[1]
    measurement_dim = state_dim
    control_dim = 2

    # Select control inputs
    ego_velocity = states['Agent1 Ego Dynamics tire velocity (action)'].values
    ego_acceleration = states['Agent1 Ego Dynamics Acceleration (action)'].values

    # Add noise to control inputs
    noise_std_velocity = 0.5
    noise_std_acceleration = 0.2
    ego_velocity_noisy = ego_velocity + np.random.normal(0, noise_std_velocity, size=ego_velocity.shape)
    ego_acceleration_noisy = ego_acceleration + np.random.normal(0, noise_std_acceleration, size=ego_acceleration.shape)

    # Initialize Kalman Filter
    kf = KalmanFilter(dim_x=state_dim, dim_z=measurement_dim, dim_u=control_dim)

    # Modify control matrix B
    kf.B[0, 0] = 1.0  
    kf.B[3, 1] = 1.0  
    kf.B[6, 0] = 1.0  
    kf.B[9, 1] = 1.0  

    # Adjust process noise covariance
    control_noise_covariance = np.diag([noise_std_velocity**2, noise_std_acceleration**2])
    kf.Q += np.dot(np.dot(kf.B, control_noise_covariance), kf.B.T)

    # Store filtered states
    filtered_states = []
    for i in range(len(obstacle_states)):
        control_vector = np.array([ego_velocity_noisy[i], ego_acceleration_noisy[i]])
        measurement_vector = np.concatenate((noisy_obstacle_states[i], noisy_ego_states[i]))
        kf.predict(u=control_vector)
        kf.update(measurement_vector)
        filtered_states.append(kf.x.flatten())

    original_df = pd.DataFrame(np.hstack((originals_obstacle, originals_ego)), 
                               columns=['obs_s_original', 'obs_d_original', 'obs_mu_original', 'obs_speed_original', 'obs_steering_original', 'obs_kappa_original',
                                        'ego_s_original', 'ego_d_original', 'ego_mu_original', 'ego_speed_original', 'ego_steering_original', 'ego_kappa_original'])
    noisy_df = pd.DataFrame(np.hstack((noisy_obstacle_states, noisy_ego_states)), 
                            columns=['obs_s_noisy', 'obs_d_noisy', 'obs_mu_noisy', 'obs_speed_noisy', 'obs_steering_noisy', 'obs_kappa_noisy',
                                     'ego_s_noisy', 'ego_d_noisy', 'ego_mu_noisy', 'ego_speed_noisy', 'ego_steering_noisy', 'ego_kappa_noisy'])
    filtered_df = pd.DataFrame(filtered_states, 
                               columns=['obs_s_filtered', 'obs_d_filtered', 'obs_mu_filtered', 'obs_speed_filtered', 'obs_steering_filtered', 'obs_kappa_filtered',
                                        'ego_s_filtered', 'ego_d_filtered', 'ego_mu_filtered', 'ego_speed_filtered', 'ego_steering_filtered', 'ego_kappa_filtered'])
    return pd.concat([original_df, noisy_df, filtered_df], axis=1)

def compare_errors_no_model(df):
    """
    Compares overall error between original, noisy, and filtered states
    using the DataFrame returned from denoise_vehicle_states.
    """
    original = df[[col for col in df.columns if col.endswith('_original')]].values
    noisy = df[[col for col in df.columns if col.endswith('_noisy')]].values
    filtered = df[[col for col in df.columns if col.endswith('_filtered')]].values
    
    noisy_errors = np.linalg.norm(noisy - original, axis=1)
    filtered_errors = np.linalg.norm(filtered - original, axis=1)
    
    print("Average noisy error (no model):", np.mean(noisy_errors))
    print("Average filtered error (no model):", np.mean(filtered_errors))
    
    return noisy_errors, filtered_errors

def compare_errors_detailed_rl(states_csv, filtered_states_df):
    """
    Computes average absolute error per state variable (s, d, mu, speed, steering, kappa)
    for both obstacle (first 6 columns) and ego (last 6 columns) using the RL-based Kalman filter.
    Ground truth is assumed to be in the first 12 columns of the CSV.
    """
    states = pd.read_csv(states_csv)
    ground_truth = states.iloc[:, :12].values  # First 6: obstacle, Last 6: ego
    noisy_obstacle = states[['Agent2 Ego Dynamics s', 'Agent2 Ego Dynamics d',
                             'Agent2 Ego Dynamics mu', 'Agent2 Ego Dynamics speed',
                             'Agent2 Ego Dynamics steering', 'Agent2 Ego Dynamics kappa']].values
    noisy_ego = states[['Agent1 Ego Dynamics s', 'Agent1 Ego Dynamics d',
                         'Agent1 Ego Dynamics mu', 'Agent1 Ego Dynamics speed',
                         'Agent1 Ego Dynamics steering', 'Agent1 Ego Dynamics kappa']].values
    noisy_states = np.concatenate((noisy_obstacle, noisy_ego), axis=1)
    filtered = filtered_states_df.values

    state_keys = ["s", "d", "mu", "speed", "steering", "kappa"]
    obstacle_noisy_errors = {}
    obstacle_filtered_errors = {}
    ego_noisy_errors = {}
    ego_filtered_errors = {}

    for idx, key in enumerate(state_keys):
         obstacle_noisy_errors[key] = np.mean(np.abs(noisy_states[:, idx] - ground_truth[:, idx]))
         obstacle_filtered_errors[key] = np.mean(np.abs(filtered[:, idx] - ground_truth[:, idx]))
         ego_noisy_errors[key] = np.mean(np.abs(noisy_states[:, idx+6] - ground_truth[:, idx+6]))
         ego_filtered_errors[key] = np.mean(np.abs(filtered[:, idx+6] - ground_truth[:, idx+6]))

    print("RL-based Kalman Filter detailed errors:")
    print("Obstacle Noisy Errors:", obstacle_noisy_errors)
    print("Obstacle Filtered Errors:", obstacle_filtered_errors)
    print("Ego Noisy Errors:", ego_noisy_errors)
    print("Ego Filtered Errors:", ego_filtered_errors)

    return obstacle_noisy_errors, obstacle_filtered_errors, ego_noisy_errors, ego_filtered_errors

def compare_errors_detailed_no_model(df):
    """
    Computes average absolute error per state variable for both obstacle and ego
    using the DataFrame returned from denoise_vehicle_states.
    """
    state_keys = ["s", "d", "mu", "speed", "steering", "kappa"]
    obstacle_noisy_errors = {}
    obstacle_filtered_errors = {}
    ego_noisy_errors = {}
    ego_filtered_errors = {}

    for key in state_keys:
         orig_obs = df["obs_" + key + "_original"].values
         noisy_obs = df["obs_" + key + "_noisy"].values
         filt_obs = df["obs_" + key + "_filtered"].values
         orig_ego = df["ego_" + key + "_original"].values
         noisy_ego = df["ego_" + key + "_noisy"].values
         filt_ego = df["ego_" + key + "_filtered"].values
         obstacle_noisy_errors[key] = np.mean(np.abs(noisy_obs - orig_obs))
         obstacle_filtered_errors[key] = np.mean(np.abs(filt_obs - orig_obs))
         ego_noisy_errors[key] = np.mean(np.abs(noisy_ego - orig_ego))
         ego_filtered_errors[key] = np.mean(np.abs(filt_ego - orig_ego))

    print("Model-Free (Classic) Kalman Filter detailed errors:")
    print("Obstacle Noisy Errors:", obstacle_noisy_errors)
    print("Obstacle Filtered Errors:", obstacle_filtered_errors)
    print("Ego Noisy Errors:", ego_noisy_errors)
    print("Ego Filtered Errors:", ego_filtered_errors)

    return obstacle_noisy_errors, obstacle_filtered_errors, ego_noisy_errors, ego_filtered_errors


### FOR V2 ###

def denoise_vehicle_states_with_rl2(states_csv):

    states = pd.read_csv(states_csv)
    
    # Get original states (using the same columns as in the classic version)
    originals_obstacle = states[['Agent2 Ego Dynamics s', 'Agent2 Ego Dynamics d', 
                                 'Agent2 Ego Dynamics mu', 'Agent2 Ego Dynamics speed', 
                                 'Agent2 Ego Dynamics steering', 'Agent2 Ego Dynamics kappa']].values
    originals_ego = states[['Agent1 Ego Dynamics s', 'Agent1 Ego Dynamics d', 
                            'Agent1 Ego Dynamics mu', 'Agent1 Ego Dynamics speed', 
                            'Agent1 Ego Dynamics steering', 'Agent1 Ego Dynamics kappa']].values
    original_df = pd.DataFrame(
        np.hstack((originals_obstacle, originals_ego)),
        columns=[
            'obs_s_original', 'obs_d_original', 'obs_mu_original', 'obs_speed_original', 'obs_steering_original', 'obs_kappa_original',
            'ego_s_original', 'ego_d_original', 'ego_mu_original', 'ego_speed_original', 'ego_steering_original', 'ego_kappa_original'
        ]
    )
    
    # Add noise using identical parameters as the classic function
    noisy_obstacle_states = originals_obstacle + np.random.normal(0, np.abs(originals_obstacle/4), originals_obstacle.shape)
    noisy_ego_states = originals_ego + np.random.normal(0, np.abs(originals_ego/4), originals_ego.shape)
    noisy_df = pd.DataFrame(
        np.hstack((noisy_obstacle_states, noisy_ego_states)),
        columns=[
            'obs_s_noisy', 'obs_d_noisy', 'obs_mu_noisy', 'obs_speed_noisy', 'obs_steering_noisy', 'obs_kappa_noisy',
            'ego_s_noisy', 'ego_d_noisy', 'ego_mu_noisy', 'ego_speed_noisy', 'ego_steering_noisy', 'ego_kappa_noisy'
        ]
    )
    
    # Prepare to perform filtering with RL dynamic Q/R from model2
    state_dim = noisy_obstacle_states.shape[1] + noisy_ego_states.shape[1]
    control_dim = 2  
    kf = KalmanFilter(dim_x=state_dim, dim_z=state_dim, dim_u=control_dim)
    
    filtered_states = []
    q_diagonals = []  # store the Q diagonal vectors
    r_diagonals = []  # store the R diagonal vectors
    
    # Apply filtering per sample
    for i in range(len(noisy_obstacle_states)):
        measurement_vector = np.concatenate((noisy_obstacle_states[i], noisy_ego_states[i]))
        
        # Prepare observation for the RL model.
        # Here, we concatenate the current state estimate and an additional zero element if required.
        obs = np.concatenate((kf.x.flatten(), np.zeros(1)))
        
        # Get the action vector from model2.
        # The action vector is assumed to have length 2*state_dim.
        action = model2.predict(obs, deterministic=True)[0]
        q_diag = action[:state_dim]
        r_diag = action[state_dim:]
        
        # Save the Q and R diagonals for logging/analysis.
        q_diagonals.append(q_diag.tolist())
        r_diagonals.append(r_diag.tolist())
        
        # Set the Q and R matrices as diagonal matrices based on the predicted values.
        kf.Q = np.diag(q_diag)
        kf.R = np.diag(r_diag)
    
        # Use control inputs from the CSV.
        control_vector = np.array([
            states['Agent1 Ego Dynamics tire velocity (action)'][i],
            states['Agent1 Ego Dynamics Acceleration (action)'][i]
        ])
        kf.predict(u=control_vector)
        kf.update(measurement_vector)
        filtered_states.append(kf.x.flatten())
    
    filtered_df = pd.DataFrame(
        filtered_states, 
        columns=[
            'obs_s_filtered', 'obs_d_filtered', 'obs_mu_filtered', 'obs_speed_filtered', 'obs_steering_filtered', 'obs_kappa_filtered',
            'ego_s_filtered', 'ego_d_filtered', 'ego_mu_filtered', 'ego_speed_filtered', 'ego_steering_filtered', 'ego_kappa_filtered'
        ]
    )
    
    # Optionally, you can add the Q and R diagonal values as new columns.
    # Here we add them as strings (or you can store them in a different format).
    filtered_df['Q_diagonal'] = [str(qd) for qd in q_diagonals]
    filtered_df['R_diagonal'] = [str(rd) for rd in r_diagonals]
    
    return pd.concat([original_df, noisy_df, filtered_df], axis=1)

def compare_errors2(states_csv, filtered_states_df):
    """
    Compares overall error (Euclidean norm) of the RL-based filtered states
    and the RL-based noisy measurements with the ground truth.
    """
    import numpy as np
    import pandas as pd
    states = pd.read_csv(states_csv)
    ground_truth = states.iloc[:, :12].values  # ground truth from CSV

    # Extract only the '_filtered' and '_noisy' columns from the concatenated DataFrame.
    filtered = filtered_states_df.filter(like='_filtered').values
    noisy = filtered_states_df.filter(like='_noisy').values
    
    filtered_errors = np.linalg.norm(filtered - ground_truth, axis=1)
    noisy_errors = np.linalg.norm(noisy - ground_truth, axis=1)
    
    print("Average filtered error (RL2):", np.mean(filtered_errors))
    print("Average noisy error (RL2):", np.mean(noisy_errors))
    
    return filtered_errors, noisy_errors

# ----- Execution for RL-based Kalman Filter -----
model = PPO.load("kalman_rl_model")
test_csv = "test_data.csv"
filtered_states_df = denoise_vehicle_states_with_rl(test_csv)
filtered_states_df.to_csv('./filtered_vehicle_states_RL.csv', index=False)
compare_errors(test_csv, filtered_states_df)
#compare_errors_detailed_rl(test_csv, filtered_states_df)

# ----- Execution for RL-based Kalman Filter -----
model2 = PPO.load("kalman_rl_model2")
test_csv = "test_data.csv"
filtered_states_df = denoise_vehicle_states_with_rl2(test_csv)
filtered_states_df.to_csv('./filtered_vehicle_states_RL2.csv', index=False)
compare_errors2(test_csv, filtered_states_df)
#compare_errors_detailed_rl(test_csv, filtered_states_df)

# ----- Execution for Model-Free (Classic) Kalman Filter -----
file_path = "test_data.csv"
data = pd.read_csv(file_path)
filtered_data = denoise_vehicle_states(data)
filtered_data.to_csv('./filtered_vehicle_states.csv', index=False)
compare_errors_no_model(filtered_data)
#compare_errors_detailed_no_model(filtered_data)