import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from kfilter import KalmanFilter
from ppo_callback import PPOLoggingCallback

class KalmanFilterEnv(gym.Env):
    def __init__(self, csv_file):
        super(KalmanFilterEnv, self).__init__()

        self.df = pd.read_csv(csv_file)
        self.total_steps = len(self.df)

        self.state_dim = 12
        self.action_dim = 2  

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim + 1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.01, high=10, shape=(self.action_dim,), dtype=np.float32)  

        self.kf = KalmanFilter(dim_x=self.state_dim, dim_z=self.state_dim, dim_u=2)
        self.current_step = 0
        self.current_noise_type = "gaussian"  # Default

    def reset(self):
        self.current_step = 0
        self.kf.x = np.zeros((self.state_dim, 1))

        # Randomly choose a noise distribution per episode
        self.current_noise_type = np.random.choice(["gaussian", "uniform", "laplace", "white_noise"])

        obs = np.concatenate((self.kf.x.flatten(), np.zeros(1)))
        return obs

    """def add_noise(self, true_value, std):
        # Applies different noise distributions based on the selected type.
        if self.current_noise_type == "gaussian":
            return true_value + np.random.normal(0, std, true_value.shape)
        elif self.current_noise_type == "uniform":
            return true_value + np.random.uniform(-std, std, true_value.shape)
        elif self.current_noise_type == "laplace":
            return true_value + np.random.laplace(0, std, true_value.shape)
        elif self.current_noise_type == "white_noise":
            return true_value + np.random.normal(0, std * np.random.uniform(0.5, 2.0), true_value.shape)"""

    def add_noise(self, true_value, std):
        """Applies different noise distributions based on the selected type."""
        if self.current_noise_type == "gaussian":
            return true_value + np.random.normal(0, std, true_value.shape)
        elif self.current_noise_type == "uniform":
            return true_value + np.random.uniform(-std * np.sqrt(3), std * np.sqrt(3), true_value.shape)
        elif self.current_noise_type == "laplace":
            return true_value + np.random.laplace(0, std / np.sqrt(2), true_value.shape)
        elif self.current_noise_type == "white_noise":
            return true_value + np.random.normal(0, std * np.random.uniform(0.8, 1.2), true_value.shape)


    def step(self, action):
        q_scale, r_scale = action
        
        # Adjust Kalman parameters dynamically
        self.kf.Q = np.eye(self.state_dim) * q_scale
        self.kf.R = np.eye(self.state_dim) * r_scale

        # Extract ground truth state
        true_state = self.df.iloc[self.current_step].values[:self.state_dim]

        # 1. Add randomized noise to the true state (ego & obstacle)
        state_noise_std = np.abs(true_state / 4)  # Match obstacle/ego state noise scaling
        noisy_measurement = self.add_noise(true_state, state_noise_std)

        # 2. Add noise to the control inputs
        true_control_input = self.df.iloc[self.current_step][['Agent1 Ego Dynamics tire velocity (action)', 
                                                              'Agent1 Ego Dynamics Acceleration (action)']].values
        control_noise_std = np.array([0.5, 0.2])
        noisy_control_input = self.add_noise(true_control_input, control_noise_std)

        # Kalman predict and update using noisy data
        self.kf.predict(u=noisy_control_input)
        self.kf.update(noisy_measurement)

        # Compute reward (negative error)
        error = np.linalg.norm(self.kf.x.flatten() - true_state)
        reward = -error

        # Observation: KF state + innovation residual
        innovation = np.linalg.norm(noisy_measurement - self.kf.x.flatten(), ord=2)
        obs = np.concatenate((self.kf.x.flatten(), np.array([innovation])))

        self.current_step += 1
        done = self.current_step >= self.total_steps

        return obs, reward, done, {}

csv_file = "train_data.csv"  # Path to ground truth CSV
env = KalmanFilterEnv(csv_file)

model = PPO("MlpPolicy", env,tensorboard_log="./logs", verbose=1)
callback = PPOLoggingCallback()
model.learn(total_timesteps=100000, callback=callback)
model.save("kalman_rl_model")
