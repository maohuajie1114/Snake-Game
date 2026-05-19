import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import csv

from snake_feature import SnakeEnv
from dqn_agent import DQNAgent


def moving_average(data: list, window_size: int = 50) -> list:
    """Calculates smoothed data for cleaner plotting."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_training_curve(scores: list, save_path: str = "training_curve.png") -> None:
    """Plots and saves the training progress."""
    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Raw Score', alpha=0.3, color='blue')

    smoothed = moving_average(scores, window_size=50)
    # Offset the smoothed curve to align with the raw data
    plt.plot(range(len(scores) - len(smoothed), len(scores)), smoothed,
             label='Smoothed Score (MA=50)', color='red', linewidth=2)

    plt.title('DQN Snake Training Progress')
    plt.xlabel('Episodes')
    plt.ylabel('Score (Max 192)')
    plt.ylim([0, 200])  # Cap y-axis slightly above max score for readability
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training curve saved to {save_path}")


def evaluate_and_record(agent: DQNAgent, step: int, num_episodes: int = 5, save_dir: str = "eval_videos") -> float:
    """
    Performs noise-free, purely greedy evaluation.
    The first episode is silently recorded and saved as an mp4 video; subsequent episodes are run only to calculate the average score.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n--- Triggering Evaluation at Step {step} ---")

    eval_scores = []

    for ep in range(num_episodes):
        # Only record the 0-th episode, the rest are just for calculating the score to speed things up
        if ep == 0:
            base_env = SnakeEnv(render_mode="rgb_array")
            env = RecordVideo(
                base_env,
                video_folder=save_dir,
                name_prefix=f"step_{step}",
                episode_trigger=lambda x: True,
                disable_logger=True
            )
        else:
            env = SnakeEnv(render_mode=None)

        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state, greedy=True)  # Strictly noise-free evaluation
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        eval_scores.append(info['score'])
        env.close()

    avg_eval_score = np.mean(eval_scores)
    print(f"Evaluation finished over {num_episodes} games. Scores: {eval_scores} | Avg: {avg_eval_score:.2f}")
    return avg_eval_score


def train():
    # --- Configuration & Argument Parsing ---
    parser = argparse.ArgumentParser(description="DQN Snake Training Script")
    parser.add_argument('--resume_ckpt', type=str, default=None, help="Path to checkpoint .pth file to resume from")
    parser.add_argument('--resume_buffer', type=str, default=None,
                        help="Path to replay buffer .pkl file to resume from")
    args = parser.parse_args()

    # Constraints parameters
    MAX_TIME_SECONDS = 12 * 3600  # Training duration (Hours)
    CHECKPOINT_INTERVAL = 100_000  # Save & Record every 100k steps
    TARGET_UPDATE_FREQ = 10_000  # Hard update target network every 10k steps

    # Setup directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("buffers", exist_ok=True)

    env = SnakeEnv()
    agent = DQNAgent(state_dim=17, action_dim=3)

    global_step = 0
    scores_history = []

    # --- Fault Tolerance: Resume Loading ---
    if args.resume_ckpt:
        global_step, scores_history = agent.load_checkpoint(args.resume_ckpt)
    if args.resume_buffer:
        agent.memory.load(args.resume_buffer)
        print(f"Replay buffer loaded with {len(agent.memory)} experiences.")

    start_time = time.time()
    next_checkpoint_step = ((global_step // CHECKPOINT_INTERVAL) + 1) * CHECKPOINT_INTERVAL

    # ================= Initialize CSV file =================
    csv_file_path = "training_log.csv"
    # If resuming from a checkpoint, select append mode 'a'; otherwise, select overwrite mode 'w'
    file_mode = 'a' if args.resume_ckpt else 'w'
    csv_file = open(csv_file_path, mode=file_mode, newline='')
    csv_writer = csv.writer(csv_file)

    if file_mode == 'w':
        # Only write the header for a new training session
        csv_writer.writerow(["Step", "Train_Score", "Eval_Score"])
    # ===================================================

    print("=======================================")
    print(f"Starting Training at step: {global_step}")
    print("=======================================")

    try:
        while True:  # Episode Loop
            state, _ = env.reset()
            episode_score = 0
            done = False
            trajectory = []

            while not done:  # Step Loop
                # Time limit enforcement
                elapsed_time = time.time() - start_time
                if elapsed_time > MAX_TIME_SECONDS:
                    print("\n[Time Limit Reached] Commencing safe shutdown...")
                    raise KeyboardInterrupt

                # 1. Step in Environment
                action = agent.act(state, greedy=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Format transition to match the skeleton's expected signature
                trajectory.append((state, action, reward, next_state))
                state = next_state

                global_step += 1

                # 2. Update target network periodically
                if global_step % TARGET_UPDATE_FREQ == 0:
                    agent.update_target_network()

                # 3. Checkpoint & Record Video
                if global_step >= next_checkpoint_step:
                    ckpt_path = f"checkpoints/dqn_step_{global_step}.pth"
                    buf_path = f"buffers/buffer_step_{global_step}.pkl"

                    agent.save_checkpoint(ckpt_path, global_step, scores_history)
                    agent.memory.save(buf_path)
                    print(f"Checkpoint saved to {ckpt_path}")

                    # Get the average score from the pure greedy evaluation
                    avg_score = evaluate_and_record(agent, global_step)
                    csv_writer.writerow([global_step, episode_score, avg_score])
                    csv_file.flush()
                    next_checkpoint_step += CHECKPOINT_INTERVAL

                    print(
                        f"Step: {global_step} | Episodes: {len(scores_history)} | Epsilon: {agent.epsilon:.3f} | Last Eval Avg Score: {avg_score:.2f}")

            # End of Episode Processing
            episode_score = info['score']
            scores_history.append(episode_score)

            # ================= Write current step and score to CSV =================
            csv_writer.writerow([global_step, episode_score, ""]) # When a normal training episode ends, Eval_Score is left empty ("") (indicating no evaluation was performed at this step)
            csv_file.flush()  # Force flush to disk to prevent data loss on crash
            # =====================================================================

            # Pass the single episode trajectory to the agent
            agent.learn([trajectory])
            agent.decay_epsilon()

            # Logging
            if len(scores_history) % 100 == 0:
                recent_avg = np.mean(scores_history[-100:])

                elapsed_time = time.time() - start_time

                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{int(hours):02d}h:{int(minutes):02d}m:{int(seconds):02d}s"

                print(
                    f"[{time_str}] Step: {global_step} | Episodes: {len(scores_history)} | Epsilon: {agent.epsilon:.3f} | Last 100 Avg Score: {recent_avg:.2f}")

    except KeyboardInterrupt:
        # This block triggers either by manual Ctrl+C or the 12-hour time limit raise.
        print("\nSaving final model and plotting training curves before exit...")

        final_ckpt = f"checkpoints/dqn_final_step_{global_step}.pth"
        final_buf = f"buffers/buffer_final_step_{global_step}.pkl"

        agent.save_checkpoint(final_ckpt, global_step, scores_history)
        agent.memory.save(final_buf)

        # Close the CSV file
        csv_file.close()

        # You can also call the plotting function directly here, or leave it to a separate script
        plot_training_curve(scores_history)
        env.close()
        print("Safe shutdown complete.")


if __name__ == "__main__":
    train()