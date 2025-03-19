import time
import numpy as np
from stable_baselines3 import PPO
from kfilter import KalmanFilter

# Load the first RL model
model = PPO.load("kalman_rl_model")

# Define dimensions (assumed 12 state variables: 6 obstacle + 6 ego states)
state_dim = 12  # Adjust if needed
control_dim = 2  # For example: tire velocity and acceleration

# Initialize Kalman Filter with initial state and covariance settings
kf = KalmanFilter(dim_x=state_dim, dim_z=state_dim, dim_u=control_dim)
kf.x = np.zeros((state_dim, 1))
kf.P = np.eye(state_dim)

def get_real_system_data():
    """
    Dummy function to simulate retrieval of sensor measurements and control inputs.
    Replace this with the actual connection code to your sensors/system.
    """
    measurement = np.random.randn(state_dim)  # Simulated noisy measurement
    control = np.random.randn(control_dim)      # Simulated control input
    return measurement, control

def main():
    while True:
        # Retrieve actual measurement and control data from the real system
        measurement, control = get_real_system_data()

        # Prepare observation for the RL model (current filter state + an extra placeholder)
        obs = np.concatenate((kf.x.flatten(), [0.0]))
        
        # Get the predicted noise scales from the RL model (q_scale, r_scale)
        q_scale, r_scale = model.predict(obs, deterministic=True)[0]
        
        # Update Kalman filter noise covariance matrices dynamically
        kf.Q = np.eye(state_dim) * q_scale
        kf.R = np.eye(state_dim) * r_scale
        
        # Kalman filter prediction and update steps
        kf.predict(u=control)
        kf.update(measurement)
        
        # Get and print the filtered state estimate
        state_estimate = kf.x.flatten()
        print("Filtered State Estimate (RL Model):", state_estimate)
        
        # FILLER: Insert your system integration code to send state_estimate elsewhere
        
        # Pause to simulate a real-time loop (adjust the sleep time as needed)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
