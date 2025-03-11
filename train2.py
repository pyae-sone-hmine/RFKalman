import gym
from gym import spaces
import numpy as np
import pandas as pd
import csv
import os
from stable_baselines3 import PPO
from kfilter import KalmanFilter

class KalmanFilterEnv(gym.Env):
    def __init__(self, csv_file):
        super(KalmanFilterEnv, self).__init__()

        self.df = pd.read_csv(csv_file)
        self.total_steps = len(self.df)

        self.state_dim = 12
        # Now action_dim is 2 * state_dim so that each non-zero (diagonal) entry of Q and R can be set individually.
        self.action_dim = 2 * self.state_dim  

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim + 1,), dtype=np.float32)
        # Each action value (individual diagonal element) is constrained between 0.01 and 10.
        self.action_space = spaces.Box(low=0.01, high=10, shape=(self.action_dim,), dtype=np.float32)

        self.kf = KalmanFilter(dim_x=self.state_dim, dim_z=self.state_dim, dim_u=2)
        self.current_step = 0
        self.current_noise_type = "gaussian"  # Default noise type

    def reset(self):
        self.current_step = 0
        self.kf.x = np.zeros((self.state_dim, 1))
        self.current_noise_type = np.random.choice(["gaussian", "uniform", "laplace", "white_noise"])
        obs = np.concatenate((self.kf.x.flatten(), np.zeros(1)))
        return obs

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
        # Split the action vector into two halves: one for Q and one for R
        q_elements = action[:self.state_dim]
        r_elements = action[self.state_dim:]
        # Set Q and R as diagonal matrices with the respective action values.
        self.kf.Q = np.diag(q_elements)
        self.kf.R = np.diag(r_elements)

        true_state = self.df.iloc[self.current_step].values[:self.state_dim]
        state_noise_std = np.abs(true_state / 4)
        noisy_measurement = self.add_noise(true_state, state_noise_std)

        true_control_input = self.df.iloc[self.current_step][['Agent1 Ego Dynamics tire velocity (action)', 
                                                              'Agent1 Ego Dynamics Acceleration (action)']].values
        control_noise_std = np.array([0.5, 0.2])
        noisy_control_input = self.add_noise(true_control_input, control_noise_std)

        self.kf.predict(u=noisy_control_input)
        self.kf.update(noisy_measurement)

        error = np.linalg.norm(self.kf.x.flatten() - true_state)
        reward = -error

        innovation = np.linalg.norm(noisy_measurement - self.kf.x.flatten(), ord=2)
        obs = np.concatenate((self.kf.x.flatten(), np.array([innovation])))

        self.current_step += 1
        done = self.current_step >= self.total_steps

        return obs, reward, done, {}

csv_file = "sample_dynamics.csv"  # Path to ground truth CSV
env = KalmanFilterEnv(csv_file)

model = PPO("MlpPolicy", env,tensorboard_log="./logs2", verbose=1)
callback = PlotProgressCallback()
model.learn(total_timesteps=1000000, callback=callback)
model.save("kalman_rl_model2")
