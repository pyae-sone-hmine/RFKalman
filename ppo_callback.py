import csv
import os
from stable_baselines3.common.callbacks import BaseCallback

class PPOLoggingCallback(BaseCallback):
    def __init__(self, log_file="ppo_kalman_logs.csv", verbose=0):
        super(PPOLoggingCallback, self).__init__(verbose)
        self.log_file = log_file
        self.episode_rewards = []  # Stores rewards for the current episode
        self.episode_length = 0    # Tracks episode length

        # Create CSV file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Total Reward", "Episode Length", "Policy Loss", "Value Loss", "Entropy"])

    def _on_step(self):
        """Called at every timestep."""
        reward = self.locals["rewards"][0]  # Get reward from the current step
        self.episode_rewards.append(reward)
        self.episode_length += 1

        # Check if episode is done
        if self.locals["dones"][0]:  # If episode ends
            total_reward = sum(self.episode_rewards)

            # Retrieve PPO Losses (if available)
            policy_loss = self.model.logger.name_to_value.get("train/policy_loss", None)
            value_loss = self.model.logger.name_to_value.get("train/value_loss", None)
            entropy = self.model.logger.name_to_value.get("train/entropy_loss", None)

            # Log to CSV
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.num_timesteps, total_reward, self.episode_length, policy_loss, value_loss, entropy])

            # Reset episode tracking
            self.episode_rewards = []
            self.episode_length = 0

        return True  # Continue training
