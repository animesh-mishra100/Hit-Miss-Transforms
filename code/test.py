import matplotlib.pyplot as plt
import numpy as np

# Data for the table and graph
image_resolutions = ["532x412", "1024x1024"]
sequential_times = [0.1002, 1.6583]  # seconds
cuda_times = [0.0006, 0.0034]  # seconds
speedup = [178.70, 485.13]

# Bar plot
x = np.arange(len(image_resolutions))
bar_width = 0.35

plt.figure(figsize=(12, 8))  # Increased figure size

# Sequential and CUDA execution times
plt.bar(x - bar_width / 2, sequential_times, bar_width, label="Sequential Time", color="blue")
plt.bar(x + bar_width / 2, cuda_times, bar_width, label="CUDA Time", color="red")

# Annotating the bars with values
for i in range(len(sequential_times)):
    plt.text(x[i] - bar_width / 2, sequential_times[i] + 0.05, f"{sequential_times[i]:.4f}s", ha="center", va="bottom", fontsize=12, weight='bold')
    plt.text(x[i] + bar_width / 2, cuda_times[i] + 0.05, f"{cuda_times[i]:.4f}s", ha="center", va="bottom", fontsize=12, weight='bold')

# Formatting the plot
plt.yscale("log")
plt.ylabel("Execution Time (seconds)", fontsize=14, weight='bold')
plt.xlabel("Image Resolution", fontsize=14, weight='bold')
plt.title("Sequential vs CUDA Execution Time", fontsize=16, weight='bold')
plt.xticks(x, image_resolutions, fontsize=12, weight='bold')
plt.legend(fontsize=12, prop={'weight':'bold'})
plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

# Display the bar plot
plt.tight_layout()
plt.show()