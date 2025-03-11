import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("ppo_kalman_logs.csv")

# Plot Reward per Episode
plt.figure(figsize=(10, 5))
plt.plot(df["Episode"], df["Total Reward"], label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Training Performance")
plt.legend()
plt.show()

# Plot Loss Trends
plt.figure(figsize=(10, 5))
plt.plot(df["Episode"], df["Policy Loss"], label="Policy Loss")
plt.plot(df["Episode"], df["Value Loss"], label="Value Loss", linestyle="--")
plt.plot(df["Episode"], df["Entropy"], label="Entropy", linestyle=":")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("PPO Loss Trends")
plt.legend()
plt.show()
