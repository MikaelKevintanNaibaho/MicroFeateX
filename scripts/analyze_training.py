#!/usr/bin/env python3
"""
Ultra-fast training log analyzer with incremental caching.
First run: Parse all records and cache.
Subsequent runs: Only parse NEW records since last run.
"""

import os
import glob
import struct
import pickle
from collections import defaultdict
import numpy as np

from tensorboard.compat.proto import event_pb2


CACHE_DIR = ".cache"

# ANSI color codes
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_RESET = "\033[0m"
COLOR_DIM = "\033[2m"


def sparkline(values, width=12):
    """
    Generate ASCII sparkline from values.
    Uses Unicode block characters: ▁▂▃▄▅▆▇█
    """
    if len(values) < 2:
        return ""

    # Take last N values for sparkline
    vals = values[-width:] if len(values) > width else values

    # Normalize to 0-7 range for block characters
    min_v, max_v = min(vals), max(vals)
    if max_v - min_v < 1e-8:
        return "▄" * len(vals)  # Flat line

    blocks = "▁▂▃▄▅▆▇█"
    normalized = [(v - min_v) / (max_v - min_v) for v in vals]
    spark = "".join(blocks[min(7, int(n * 7.99))] for n in normalized)

    return spark


def trend_arrow(values, n=50):
    """
    Get colored trend arrow based on recent values.
    For loss: green if decreasing, red if increasing
    For acc: green if increasing, red if decreasing
    """
    if len(values) < n:
        return ""

    recent = values[-n:]
    mid = len(recent) // 2
    diff = recent[mid:].mean() - recent[:mid].mean()

    return diff


def get_latest_log_dir(log_root):
    dirs = [d for d in glob.glob(os.path.join(log_root, "*")) if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def scan_tfrecord_offsets(filepath, start_offset=0):
    """
    Scan TFRecord file structure from a given offset.
    Returns list of (offset, length) for all records.

    Safety: Handles truncated records at EOF during live training.
    """
    offsets = []
    file_size = os.path.getsize(filepath)

    with open(filepath, "rb") as f:
        f.seek(start_offset)

        while True:
            offset = f.tell()

            # Check if enough bytes remain for a complete header
            if offset + 16 > file_size:
                break

            length_bytes = f.read(8)
            if len(length_bytes) < 8:
                break

            length = struct.unpack("<Q", length_bytes)[0]

            # Safety: Check if full record exists before adding
            record_end = offset + 8 + 4 + length + 4
            if record_end > file_size:
                # Truncated record - skip it (will be parsed next run)
                break

            skip_bytes = 4 + length + 4
            f.seek(skip_bytes, 1)
            offsets.append((offset, length))

    return offsets


def parse_records_at_offsets(filepath, offsets):
    """
    Parse TFRecord entries at specific offsets.

    Safety: Skips truncated or corrupted records gracefully.
    """
    events = []
    skipped = 0

    with open(filepath, "rb") as f:
        for offset, expected_length in offsets:
            try:
                f.seek(offset)
                _ = f.read(8)  # length
                _ = f.read(4)  # CRC
                data = f.read(expected_length)
                _ = f.read(4)  # CRC

                # Verify we got the expected amount of data
                if len(data) != expected_length:
                    skipped += 1
                    continue

                event = event_pb2.Event()
                event.ParseFromString(data)
                events.append(event)

            except Exception:
                # Skip corrupted/truncated records
                skipped += 1
                continue

    if skipped > 0:
        print(f"  (skipped {skipped} truncated/corrupted records)")

    return events


def extract_scalars(events):
    """Extract scalar values from events."""
    data = defaultdict(list)

    for event in events:
        if event.HasField("summary"):
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    data[value.tag].append((event.step, value.simple_value))

    return dict(data)


def merge_data(cached_data, new_data):
    """Merge new scalar data into cached data."""
    for tag, values in new_data.items():
        if tag in cached_data:
            cached_data[tag].extend(values)
        else:
            cached_data[tag] = values
    return cached_data


def get_cache_path(event_file):
    """Get cache file path for an event file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    basename = os.path.basename(event_file)
    return os.path.join(CACHE_DIR, f"{basename}.cache.pkl")


def fast_analyze_logs(log_dir):
    """
    Fast log analysis with incremental caching.
    Analyzes ALL event files in the directory (training creates new files on restart).
    """
    event_files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return

    # Process all event files and merge data
    all_data = {}
    total_new_records = 0

    for event_file in event_files:
        cache_path = get_cache_path(event_file)

        # Load cache if exists
        cached_data = {}
        cached_offset = 0

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                    cached_data = cache.get("data", {})
                    cached_offset = cache.get("offset", 0)
            except Exception:
                cached_data = {}
                cached_offset = 0

        # Scan for new records
        new_offsets = scan_tfrecord_offsets(event_file, start_offset=cached_offset)

        if new_offsets:
            new_events = parse_records_at_offsets(event_file, new_offsets)
            new_data = extract_scalars(new_events)
            cached_data = merge_data(cached_data, new_data)
            total_new_records += len(new_offsets)

            # Update cache
            new_end_offset = new_offsets[-1][0] + new_offsets[-1][1] + 16
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump({"data": cached_data, "offset": new_end_offset}, f)
            except Exception:
                pass

        # Merge this file's data into all_data
        all_data = merge_data(all_data, cached_data)

    if total_new_records > 0:
        print(f"Parsed {total_new_records} new records from {len(event_files)} files")
    else:
        print(f"Loaded cached data from {len(event_files)} files")

    # Now analyze using ALL merged data
    data = all_data

    metrics_of_interest = [
        "loss/total",
        "loss/coarse",
        "loss/fine",
        "loss/heatmap",
        "acc/coarse",
        "acc/fine",
        "acc/heatmap",
        "debug/grad_norm",
    ]

    # Find last step
    last_step = 0
    total_records = 0
    for tag, values in data.items():
        if values:
            last_step = max(last_step, max(v[0] for v in values))
            total_records += len(values)

    print(
        f"\n--- Training State (Step {last_step:,}, {total_records:,} total records) ---"
    )

    for tag in metrics_of_interest:
        if tag in data:
            values = np.array([v[1] for v in data[tag]])
            if len(values) > 0:
                current = values[-1]

                # Generate sparkline
                spark = sparkline(values[-50:])

                # Determine trend and color
                diff = trend_arrow(values)
                is_loss = "loss" in tag or "grad" in tag

                if diff != "":
                    if is_loss:  # For loss, decreasing is good
                        color = COLOR_GREEN if diff < 0 else COLOR_RED
                        arrow = "↓" if diff < 0 else "↑"
                    else:  # For accuracy, increasing is good
                        color = COLOR_GREEN if diff > 0 else COLOR_RED
                        arrow = "↑" if diff > 0 else "↓"
                    trend_str = f"{color}{arrow}{COLOR_RESET}"
                else:
                    trend_str = " "

                print(
                    f"{tag:<20}: {current:.4f} {COLOR_DIM}{spark}{COLOR_RESET} {trend_str}"
                )
        else:
            print(f"{tag:<20}: Not found")

    # Full trend analysis using ALL data
    if "loss/total" in data:
        values = np.array([v[1] for v in data["loss/total"]])
        if len(values) > 100:
            # Compare first 25% vs last 25%
            n = len(values)
            first_quarter = values[: n // 4].mean()
            last_quarter = values[-n // 4 :].mean()
            diff = last_quarter - first_quarter
            trend = "Decreasing ↓" if diff < 0 else "Increasing ↑"
            pct_change = (diff / first_quarter) * 100
            print(f"\nOverall Trend: {trend} ({pct_change:+.1f}% from start)")

            # Recent trend (last 1000 steps)
            recent = values[-1000:] if len(values) > 1000 else values
            mid = len(recent) // 2
            recent_diff = recent[mid:].mean() - recent[:mid].mean()
            recent_trend = "Decreasing ↓" if recent_diff < 0 else "Increasing ↑"
            print(
                f"Recent Trend:  {recent_trend} ({recent_diff:+.4f} last {len(recent)} steps)"
            )

    print("\n------------------------------")


if __name__ == "__main__":
    log_root = "logs"
    if not os.path.exists(log_root):
        log_root = "../logs"

    if os.path.exists(log_root):
        latest_dir = get_latest_log_dir(log_root)
        if latest_dir:
            print(f"Latest log directory: {latest_dir}")
            fast_analyze_logs(latest_dir)
        else:
            print("No log directories found.")
    else:
        print(f"Log root '{log_root}' does not exist.")
