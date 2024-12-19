import pandas as pd
import matplotlib.pyplot as plt

# Data
data = {
    "k": [1, 3, 5, 7, 10],  # Number of clusters (k)
    "map": [0.0884, 0.1117, 0.1150, 0.1167, 0.1171],  # MAP for each k
    "mar": [0.1270, 0.1584, 0.1633, 0.1647, 0.1650],  # MAR for each k
    "time": [
        77.93,
        234.40,
        439.85,
        651.62,
        765.76,
    ],  # Computation time for each k (in seconds)
}

df = pd.DataFrame(data)

# Calculate percentage increase in computation time from k=1 to k=10
initial_time = df["time"][0]
final_time = df["time"].iloc[-1]
percentage_increase = ((final_time - initial_time) / initial_time) * 100

# Set up the plot with subplots
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot MAP and MAR on the same plot
ax1.plot(
    df["k"], df["map"], label="MAP@K", marker="o", color="b", linestyle="-", linewidth=2
)
ax1.plot(
    df["k"], df["mar"], label="MAR@K", marker="s", color="g", linestyle="-", linewidth=2
)

# Set axis labels for MAP and MAR
ax1.set_xlabel("Number of Top Clusters (k)", fontsize=12)
ax1.set_ylabel("MAP / MAR", fontsize=12)
ax1.set_title(
    "Trade-off between MAP@K, MAR@K, and Number of Top Clusters (k)", fontsize=14
)

# Create a second y-axis to plot time on the same graph
ax2 = ax1.twinx()
ax2.plot(
    df["k"],
    df["time"],
    label="Computation Time",
    marker="^",
    color="r",
    linestyle="--",
    linewidth=2,
)

# Set the axis label for time
ax2.set_ylabel("Computation Time (seconds)", fontsize=12)

# Add grid, legend, and percentage annotation
ax1.grid(True)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Annotate the percentage increase
ax2.annotate(
    f"+{percentage_increase:.2f}% Time",
    xy=(10, df["time"].iloc[-1]),
    xytext=(7.5, df["time"].iloc[-1] + 100),
    arrowprops=dict(facecolor="black", arrowstyle="->"),
    fontsize=10,
    color="r",
)

# Show plot
plt.tight_layout()
plt.show()
