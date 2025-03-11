from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
import os

class PlotProgressCallback(BaseCallback):
    def __init__(self, save_freq=1000, log_dir="./plots", verbose=0):
        super(PlotProgressCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.steps = []
        self.action_means = []
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Record and log custom action statistics if available
        if "action" in self.locals:
            action = self.locals["action"]
            if isinstance(action, np.ndarray):
                action_mean = np.mean(action)
                self.logger.record("custom/action_mean", action_mean)
                self.steps.append(self.num_timesteps)
                self.action_means.append(action_mean)

        # Save plot every save_freq timesteps
        if self.num_timesteps % self.save_freq == 0 and self.steps:
            self._save_plot()
        return True

    def _save_plot(self):
        plt.figure()
        plt.plot(self.steps, self.action_means, label="Action Mean")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Action")
        plt.title("Action Mean Over Training")
        plt.legend()
        plt.tight_layout()
        filename = os.path.join(self.log_dir, f"action_mean_{self.num_timesteps}.png")
        plt.savefig(filename)
        plt.close()