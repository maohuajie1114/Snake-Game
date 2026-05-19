import csv
import numpy as np
import matplotlib.pyplot as plt
import os


def moving_average(data: np.ndarray, window_size: int = 100) -> np.ndarray:
    """Calculate moving average to smooth the curve"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_from_csv(csv_path: str = "training_log.csv", save_path: str = "training_curve.png"):
    """Read data from a CSV file and plot the training curve"""
    if not os.path.exists(csv_path):
        print(f"Error: Log file not found {csv_path}")
        return

    steps = []
    scores = []

    # Read CSV data
    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        for row in reader:
            if not row: continue  # Skip empty rows
            steps.append(int(row[0]))
            scores.append(float(row[1]))

    if not steps:
        print("No data in the CSV file!")
        return

    # Convert to NumPy arrays for processing
    steps = np.array(steps)
    scores = np.array(scores)

    # Calculate moving average
    window_size = min(100, len(scores) // 5 + 1)  # Dynamically adjust window size
    smoothed_scores = moving_average(scores, window_size=window_size)

    # Because mode='valid', the moving average array will be shorter, so we need to align the X-axis
    smoothed_steps = steps[window_size - 1:]

    # Start plotting
    plt.figure(figsize=(12, 6))

    # Plot original score scatter/line (semi-transparent)
    plt.plot(steps, scores, alpha=0.2, color='royalblue', label='Raw Score')

    # Plot smoothed trend line (bold solid line)
    plt.plot(smoothed_steps, smoothed_scores, color='darkorange', linewidth=2.5, label=f'Smoothed (MA={window_size})')

    plt.title('DQN Snake Training Progress (Score vs Steps)', fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Score (Max 192)', fontsize=12)

    # Set Y-axis range: minimum 0, maximum can be slightly over 192 for padding
    plt.ylim(0, 200)

    # Enable grid lines for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(loc='upper left')

    # Optimize layout and save/show
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # Save as a high-resolution image
    print(f"Training curve saved to: {save_path}")

    plt.show()  # If run in an environment with a GUI, it will be displayed in a popup window


if __name__ == "__main__":
    plot_from_csv()