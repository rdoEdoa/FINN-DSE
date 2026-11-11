import csv
import fcntl
import logging
import os

from collections import Counter
from typing import Any, Dict, List, Mapping

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe counters
# ─────────────────────────────────────────────────────────────────────────────


def _dedup_model_keys(models: List[str]) -> List[str]:
    counts, seen, keys = Counter(models), Counter(), []
    for m in models:
        if counts[m] > 1:
            keys.append(f"{m}_{seen[m]}")
            seen[m] += 1
        else:
            keys.append(m)
    return keys

def _next_trial_number(config_dir: str) -> int:
    os.makedirs(config_dir, exist_ok=True)
    counter_path = os.path.join(config_dir, ".trial_counter")
    lock_path    = counter_path + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            n = int(open(counter_path).read().strip()) if os.path.exists(counter_path) else 0
            with open(counter_path, "w") as f:
                f.write(str(n + 1))
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
    return n


def _increment_counter(counter_dir: str, name: str) -> int:
    os.makedirs(counter_dir, exist_ok=True)
    counter_path = os.path.join(counter_dir, f".counter_{name}")
    lock_path    = counter_path + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            n = int(open(counter_path).read().strip()) if os.path.exists(counter_path) else 0
            n += 1
            with open(counter_path, "w") as f:
                f.write(str(n))
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
    return n

def _append_to_csv(
    csv_path: str,
    trial_name: str,
    trial_params: Dict[str, Any],
    metrics: Dict[str, float],
    obj_value: float,
    success: bool,
    *,
    device: Mapping[str, float],
) -> None:
    """Append one trial result row to the running CSV."""

    # Compute percentage columns if raw counts are available
    lut  = metrics.get("lut",  0)
    dsp  = metrics.get("dsp",  0)
    bram = metrics.get("bram", 0)
    ff   = metrics.get("ff",   0)

    row = {
        "trial_name":    trial_name,
        "build_success": int(success),
        "objective":     round(obj_value, 8),
        "throughput_fps": metrics.get("throughput_fps", ""),
        "lut":      lut,
        "lut_pct":  round(lut  / device["LUT"]  * 100, 2) if lut  else "",
        "bram":     bram,
        "bram_pct": round(bram / device["BRAM"] * 100, 2) if bram else "",
        "dsp":      dsp,
        "dsp_pct":  round(dsp  / device["DSP"]  * 100, 2) if dsp  else "",
        "ff":       ff,
        "ff_pct":   round(ff   / device["FF"]   * 100, 2) if ff   else "",
        "fmax_mhz": metrics.get("fmax_mhz", ""),
    }
    # Add all trial_params as columns
    row.update({k: v for k, v in trial_params.items()})

    lock_path = csv_path + ".lock"

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            # Reuse the existing header so columns stay aligned even when the
            # row's key order differs from the first row written.
            fieldnames = None
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                with open(csv_path, newline="") as rf:
                    fieldnames = next(csv.reader(rf), None)
            write_header = not fieldnames
            if write_header:
                fieldnames = list(row.keys())
            else:
                dropped = [k for k in row if k not in fieldnames]
                if dropped:
                    log.warning("CSV %s has no columns for %s — values dropped.",
                                csv_path, dropped)

            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                        restval="", extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)