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

    train_steps = []
    train_scores = []

    eval_steps = []
    eval_scores = []

    # Read CSV data
    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if not row: continue
            step = int(row[0])

            # Record training score
            if row[1] != "":
                train_steps.append(step)
                train_scores.append(float(row[1]))

            # Record evaluation score (captured only when not empty)
            if len(row) > 2 and row[2] != "":
                eval_steps.append(step)
                eval_scores.append(float(row[2]))

    if not train_steps:
        print("No data in the CSV file!")
        return

    # Convert to NumPy arrays for processing
    train_steps = np.array(train_steps)
    train_scores = np.array(train_scores)
    eval_steps = np.array(eval_steps)
    eval_scores = np.array(eval_scores)

    # Calculate moving average
    window_size = min(150, len(train_scores) // 5 + 1)
    smoothed_train_scores = moving_average(train_scores, window_size=window_size)
    smoothed_train_steps = train_steps[window_size - 1:]

    # Start plotting
    plt.figure(figsize=(12, 6.5))

    plt.plot(train_steps, train_scores, alpha=0.1, color='royalblue', label='Train Score (Raw)')
    plt.plot(smoothed_train_steps, smoothed_train_scores, color='royalblue', linewidth=1.5,
             label=f'Train Score (MA={window_size})')

    if len(eval_steps) > 0:
        plt.plot(eval_steps, eval_scores, color='darkorange', linewidth=2.5, label='Evaluation Score (Epsilon=0)')

        max_annotations = 10
        if len(eval_steps) > max_annotations:
            indices = np.linspace(0, len(eval_steps) - 1, max_annotations).astype(int)
        else:
            indices = np.arange(len(eval_steps))

        sampled_steps = eval_steps[indices]
        sampled_scores = eval_scores[indices]

        plt.plot(sampled_steps, sampled_scores, color='darkorange', marker='o', markersize=6, linestyle='None')

        for x, y in zip(sampled_steps, sampled_scores):
            plt.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                         ha='center', fontsize=9, fontweight='bold', color='darkred')

    # chart details
    plt.title('DQN Snake: Training Exploration vs Policy Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Score (Max 192)', fontsize=12)

    plt.ylim(-5, 205)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Training and evaluation curve plot saved successfully to: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_from_csv()