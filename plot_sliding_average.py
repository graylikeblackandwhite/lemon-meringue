import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Load the CSV file
df = pd.read_csv('data/model_output2025-08-18 16:16:42.262302')

# Use the correct column names (handle possible unnamed index column)
if 'Episode' not in df.columns:
    df.columns = ['Index', 'Episode', 'Simulation Length (seconds)', 'Reward']

# Compute sliding average
def sliding_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

window_size = 200
episodes = df['Episode'].values
rewards = df['Reward'].values
avg_rewards = sliding_average(rewards, window_size)

# Adjust x-axis for sliding window
episodes_avg = episodes[window_size-1:]

plt.figure(figsize=(10,6))
plt.plot(list(episodes_avg), avg_rewards, label=f'{window_size}-Episode Sliding Average', color='tab:blue')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title(f'Sliding Average Reward (window={window_size})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
