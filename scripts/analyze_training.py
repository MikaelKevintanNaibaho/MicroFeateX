
import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def get_latest_log_dir(log_root):
    dirs = [d for d in glob.glob(os.path.join(log_root, "*")) if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def analyze_logs(log_dir):
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return

    # Use the largest event file (likely the main one)
    event_file = max(event_files, key=os.path.getsize)
    print(f"Analyzing {event_file}...")

    # Load the event accumulator
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    tags = ea.Tags()['scalars']

    metrics_of_interest = [
        "loss/total", "loss/coarse", "loss/fine", "loss/heatmap",
        "acc/coarse", "acc/fine", "acc/heatmap", "debug/grad_norm"
    ]

    print("\n--- Current Training State ---")

    # Store data for trend analysis
    data = {}

    for tag in metrics_of_interest:
        if tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = (steps, values)

            if len(values) > 0:
                current = values[-1]
                avg_last_50 = np.mean(values[-50:]) if len(values) >= 50 else np.mean(values)
                print(f"{tag:<20}: Current={current:.4f} | Avg(50)={avg_last_50:.4f}")
        else:
            print(f"{tag:<20}: Not found")

    # Simple trend analysis
    if "loss/total" in data:
        steps, values = data["loss/total"]
        if len(values) > 100:
            first_50 = np.mean(values[-100:-50])
            last_50 = np.mean(values[-50:])
            diff = last_50 - first_50
            trend = "Decreasing" if diff < 0 else "Increasing"
            print(f"\nLoss Trend (last 100 steps): {trend} ({diff:.4f})")

    if "acc/coarse" in data:
        steps, values = data["acc/coarse"]
        print(f"Steps analysed: {steps[-1] if steps else 0}")

    print("\n------------------------------")

if __name__ == "__main__":
    log_root = "logs"
    if not os.path.exists(log_root):
        # try parent directory just in case
        log_root = "../logs"

    if os.path.exists(log_root):
        latest_dir = get_latest_log_dir(log_root)
        if latest_dir:
            print(f"Latest log directory: {latest_dir}")
            analyze_logs(latest_dir)
        else:
            print("No log directories found.")
    else:
        print(f"Log root '{log_root}' does not exist.")
