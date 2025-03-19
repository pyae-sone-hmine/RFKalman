import time
import numpy as np
from stable_baselines3 import PPO
from kfilter import KalmanFilter

# Load the second RL model
model2 = PPO.load("kalman_rl_model2")

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
    Replace this with your actual sensor data retrieval code.
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
        
        # Get the action vector from the RL model
        # The action vector is assumed to have length 2*state_dim,
        # where the first state_dim elements are for the Q diagonal and the next for R.
        action = model2.predict(obs, deterministic=True)[0]
        q_diag = action[:state_dim]
        r_diag = action[state_dim:]
        
        # Update Kalman filter noise covariance matrices as diagonal matrices
        kf.Q = np.diag(q_diag)
        kf.R = np.diag(r_diag)
        
        # Kalman filter prediction and update steps
        kf.predict(u=control)
        kf.update(measurement)
        
        # Get and print the filtered state estimate
        state_estimate = kf.x.flatten()
        print("Filtered State Estimate (RL Model2):", state_estimate)
        
        # FILLER: Insert your system integration code to send state_estimate elsewhere
        
        # Pause to simulate a real-time loop (adjust the sleep time as needed)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
