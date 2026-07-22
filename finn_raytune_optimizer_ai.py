"""
finn_raytune_optimizer_ai.py
─────────────────────────────────────────────────────────────────────────────
FINN folding-config tuner with adaptive LLM-based design space pruning.

Extends finn_raytune_optimizer.py with:
  - A running CSV of all trial results (trial_params + metrics).
  - Periodic calls to space_pruner.py every --prune_every trials.
  - The search algorithm (Optuna/Nevergrad/Random) is NOT interrupted:
    Optuna resumes from its persistent study; Nevergrad/Random restart
    their sampling from the new restricted space.

Workflow
────────
  Round 0 (optional): call space_pruner once before any builds.
  Round k (every N trials):
    1. Run N trials with the current search space.
    2. Append all N trial results to trial_results.csv.
    3. Call space_pruner — it reads the CSV and proposes a new space.
    4. Reload the space and run the next round.

Usage
─────
  python finn_raytune_optimizer_ai.py \
      --baseline_cfg        dataset/config_files/resnet50/folding_config_baseline.json \
      --build_cfg           build_configs/resnet50_build_config.json \
      --build_script        custom_builds/resnet50_build.py \
      --onnx_path           dataset/resnet50/resnet50.onnx \
      --epoch_schedule      "50,50,100,200,250,350" \
      --prune_on_start \
      --objective           throughput \
      --constraints         "Maximize the throughput while keeping the resource usage under the physical board limits. The search needs to converge rapidly to the maximum feasible throughput, so prune aggressively towards values that improve the throughput." \
      --search_strategy     nevergrad \
      --max_concurrent      10 \
      --extra_build_args    --estimate-only
"""

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

# Import the custom modules
from space_pruner import run_pruner
from database_integrator import scan_source, integrate

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEVICE_LUT  = 1727040
DEVICE_DSP  = 12288
DEVICE_BRAM = 2688
DEVICE_FF   = 3454080

MAX_UTIL    = 0.85
PENALTY      = 10
PENALTY_HARD = 10000

REPORT_FILES = {
    "status" : "status.json",
    "synth"  : "ooc_synth_and_timing.json",
    "perf"   : "rtlsim_performance.json",
}

# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe counters  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _next_trial_number(config_dir: str) -> int:
    import fcntl
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
    import fcntl
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


def _read_counter(counter_dir: str, name: str) -> int:
    path = os.path.join(counter_dir, f".counter_{name}")
    try:
        return int(open(path).read().strip())
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Folding config helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def load_folding_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


FINN_CONFIG_KEYS = {
    "PE", "SIMD", "ram_style", "mem_mode", "resType",
    "preferred_impl_style", "runtime_writeable_weights",
}


def _sanitize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for node_name, node_val in cfg.items():
        if not isinstance(node_val, dict):
            result[node_name] = node_val
            continue
        clean = {}
        for k, v in node_val.items():
            if node_name == "Defaults":
                continue
            if k not in FINN_CONFIG_KEYS:
                continue
            if isinstance(v, float) and v.is_integer():
                clean[k] = int(v)
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        result[node_name] = clean
    return result


def save_folding_config(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_sanitize_config(cfg), f, indent=2)


def patch_folding_config(
    baseline: Dict[str, Any],
    trial_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Takes a clean baseline (MVAU_0) and explodes it into 
    explicit HLS and RTL variants (MVAU_hls_0, MVAU_rtl_0) 
    for the final build config.
    """
    # Start with a clean config containing only Defaults
    patched = {"Defaults": deepcopy(baseline.get("Defaults", {}))}

    # First, transfer any non-tuning info (like ram_style) from baseline
    # to the new explicit format
    for node_name, node_cfg in baseline.items():
        if node_name == "Defaults":
            continue
        
        # Parse "MVAU_0" -> "MVAU", "0"
        m = re.match(r"^(.+)_(\d+)$", node_name)
        if m:
            prefix, idx = m.group(1), m.group(2)
            for variant in ("hls", "rtl"):
                target_name = f"{prefix}_{variant}_{idx}"
                patched[target_name] = {"ram_style": "auto"}
                # Copy existing baseline values as starting points
                if "PE" in node_cfg: patched[target_name]["PE"] = node_cfg["PE"]
                if "SIMD" in node_cfg: patched[target_name]["SIMD"] = node_cfg["SIMD"]

    # Second, apply the sampled parameters from Ray Tune
    for key, value in trial_params.items():
        if "__" not in key:
            continue
        
        # e.g., "MVAU_0__PE" -> logical_name="MVAU_0", param="PE"
        logical_name, param = key.rsplit("__", 1)
        val_int = int(value)

        # Parse "MVAU_0" -> "MVAU", "0"
        m = re.match(r"^(.+)_(\d+)$", logical_name)
        if m:
            prefix, idx = m.group(1), m.group(2)
            for variant in ("hls", "rtl"):
                target_name = f"{prefix}_{variant}_{idx}"
                
                # Create entry if it doesn't exist, then set param
                if target_name not in patched:
                    patched[target_name] = {"ram_style": "auto"}
                
                patched[target_name][param] = val_int

    return patched


def total_parallelism(trial_params: Dict[str, Any]) -> int:
    total = 0
    for v in trial_params.values():
        try:
            total += int(v)
        except (TypeError, ValueError):
            pass
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Search space loader
# ─────────────────────────────────────────────────────────────────────────────

def load_search_space(filepath: str) -> Dict[str, Any]:
    """Load a pruned_space.json and wrap lists in tune.choice()."""
    with open(filepath) as f:
        raw = json.load(f)
    space = {}
    for param_name, choices in raw.items():
        if not choices:
            log.warning("Parameter %s has empty choices — skipping.", param_name)
            continue
        space[param_name] = tune.choice(choices)
    return space


# ─────────────────────────────────────────────────────────────────────────────
# Database lookup  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _configs_match(trial_params: Dict[str, Any], db_params_path: str) -> bool:
    try:
        with open(db_params_path) as f:
            db_params = json.load(f)
        return trial_params == db_params
    except Exception:
        return False


def lookup_database(
    trial_params: Dict[str, Any],
    database_dir: str,
    finn_output_dir: str,
) -> bool:
    import shutil
    if not os.path.isdir(database_dir):
        return False
    for entry in sorted(Path(database_dir).iterdir()):
        if not entry.is_dir():
            continue
        db_params_path = entry / "trial_params.json"
        if not db_params_path.exists():
            continue
        if _configs_match(trial_params, str(db_params_path)):
            os.makedirs(finn_output_dir, exist_ok=True)
            for src_file in entry.iterdir():
                if src_file.is_file():
                    import shutil as sh
                    sh.copy(src_file, os.path.join(finn_output_dir, src_file.name))
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Report parsing  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _read_json(directory: str, filename: str) -> Optional[dict]:
    path = Path(directory) / filename
    if not path.exists():
        return None
    try:
        content = path.read_text().strip()
        return json.loads(content) if content else None
    except json.JSONDecodeError:
        return None


def collect_metrics(finn_output_dir: str) -> Dict[str, float]:
    """
    Reads JSON reports. Automatically handles both full Vivado builds
    and FINN analytical estimation builds.
    """
    metrics: Dict[str, float] = {}
    
    # ── 1. Status Check ──────────────────────────────────────────────────────
    status = _read_json(finn_output_dir, "status.json")
    if status:
        metrics["finn_success"] = float(status.get("success", 0))

    # ── 2. Resource Metrics (Vivado vs Estimates) ────────────────────────────
    synth   = _read_json(finn_output_dir, "ooc_synth_and_timing.json")
    est_res = _read_json(finn_output_dir, "estimate_layer_resources.json")
    est_hls = _read_json(finn_output_dir, "estimate_layer_resources_hls.json")

    if synth:
        # Parse flat Vivado report
        for src, dst in [("LUT", "lut"), ("BRAM", "bram"), ("DSP", "dsp"),
                         ("FF", "ff"), ("WNS", "wns"), ("fmax_mhz", "fmax_mhz")]:
            val = synth.get(src)
            if val is not None:
                try: metrics[dst] = float(val)
                except (TypeError, ValueError): pass
    elif est_res or est_hls:
        # Step A: Merge the reports into a single dictionary to PREVENT double-counting.
        # HLS estimates will safely overwrite analytical estimates for the same layer.
        combined_estimates = {}
        
        if est_res:
            for layer_name, counts in est_res.items():
                if layer_name != "total" and isinstance(counts, dict):
                    combined_estimates[layer_name] = counts
                    
        if est_hls:
            for layer_name, counts in est_hls.items():
                if isinstance(counts, dict):
                    combined_estimates[layer_name] = counts  # True overwrite!

        # Step B: Tally up the merged unique layers
        lut, dsp, bram, ff = 0.0, 0.0, 0.0, 0.0
        
        for layer_name, counts in combined_estimates.items():
            lut += float(counts.get("LUT", 0))
            ff  += float(counts.get("FF", 0))
            dsp += float(counts.get("DSP", 0)) + float(counts.get("DSP48E", 0))
            
            # Safely request exact BRAM keys instead of wildcard matching
            bram_18k = float(counts.get("BRAM_18K", 0))
            bram_36k = float(counts.get("BRAM_36K", 0)) 
            bram += (bram_18k / 2.0) + bram_36k

        metrics.update({
            "lut": float(lut),
            "dsp": float(dsp),
            "bram": float(bram),
            "ff": float(ff),
            "wns": 0.0,         # Estimates assume perfect timing
            "fmax_mhz": 100.0   # Assumes 10ns target clock
        })

    # ── 3. Throughput Metrics (Vivado vs Estimates) ──────────────────────────
    perf     = _read_json(finn_output_dir, "rtlsim_performance.json")
    est_perf = _read_json(finn_output_dir, "estimate_network_performance.json")

    if perf:
        val = perf.get("throughput[images/s]")
        if val is not None:
            metrics["throughput_fps"] = float(val)
    elif est_perf:
        val = est_perf.get("estimated_throughput_fps")
        if val is not None:
            metrics["throughput_fps"] = float(val)

    return metrics


def get_warm_start_points(csv_path: str, new_space_json: str, max_points: int = 5) -> List[Dict[str, Any]]:
    """
    Reads past trials, filters for successful builds, verifies they fit the new space,
    and returns ONLY the top N performing configurations to warm-start Optuna.
    """
    if not os.path.exists(csv_path) or not os.path.exists(new_space_json):
        return []
    
    with open(new_space_json, 'r') as f:
        new_space = json.load(f)
        
    valid_trials = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 1. Only keep successful builds
                if row.get('build_success') != '1':
                    continue
                
                trial_params = {}
                is_legal = True
                
                # 2. Check if it's still legal
                for param_name, valid_choices in new_space.items():
                    if param_name not in row:
                        is_legal = False
                        break
                    
                    past_val = int(row[param_name])
                    if past_val in valid_choices:
                        trial_params[param_name] = past_val
                    else:
                        is_legal = False
                        break
                
                # 3. Save both the params and the objective score
                if is_legal:
                    # Look for objective, default to a massive penalty if missing
                    obj_val = float(row.get('objective', 1e9))
                    valid_trials.append({'params': trial_params, 'objective': obj_val})
                    
    except Exception as e:
        log.warning("Could not load warm start points: %s", e)
        
    # Sort the trials by best objective (lowest is better)
    valid_trials.sort(key=lambda x: x['objective'])
    
    # Extract just the parameter dictionaries for the top N trials
    warm_start_points = [t['params'] for t in valid_trials[:max_points]]
    
    return warm_start_points

# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────

def compute_objective(metrics, objective):
    if not metrics:
        return PENALTY_HARD
    
    lut  = metrics.get("lut",  PENALTY)
    dsp  = metrics.get("dsp",  0.0)
    bram = metrics.get("bram", 0.0)
    ff   = metrics.get("ff",   0.0)
    wns  = metrics.get("wns",  None)
    tp   = metrics.get("throughput_fps", 0.0)

    # Normalised utilisation fractions
    lut_u  = lut  / DEVICE_LUT
    dsp_u  = dsp  / DEVICE_DSP  if dsp  > 0 else 0.0
    bram_u = bram / DEVICE_BRAM if bram > 0 else 0.0
    ff_u   = ff   / DEVICE_FF   if ff   > 0 else 0.0
    
    # Double check for utilization over 100%
    if lut_u > 1 or dsp_u > 1 or bram_u > 1 or ff_u > 1:
        return PENALTY_HARD
    else:
        fail_penalty = 0.0

    # Over-utilisation penalty (hard wall at MAX_UTIL)
    over_util = max(0.0, lut_u - MAX_UTIL,
                         dsp_u - MAX_UTIL,
                         bram_u - MAX_UTIL)
    util_penalty = over_util * PENALTY

    timing_ok = (wns is None) or (wns >= 0.0)
    timing_penalty = 0.0 if timing_ok else abs(wns) * 1000

    # Bottleneck utilisation = the most-used resource
    bottleneck = max(lut_u, dsp_u, bram_u, ff_u)

    if objective == "lut":
        return lut_u + util_penalty

    elif objective == "lut_slack":
        return lut_u + timing_penalty + util_penalty

    elif objective == "throughput":
        if tp <= 0:
            return PENALTY_HARD
        return -tp + fail_penalty
    
    elif objective == "resource_avg":
        avg =  (lut_u + dsp_u + bram_u + ff_u) / 4 
        return avg + util_penalty + fail_penalty

    elif objective == "balanced":
        if tp <= 0:
            return PENALTY_HARD
        return bottleneck / tp + util_penalty

    else:
        raise ValueError(f"Unknown objective: '{objective}'")


# ─────────────────────────────────────────────────────────────────────────────
# FINN build runner  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def run_finn_build(
    build_script: str,
    onnx_dir: str,
    model_name: str,
    folding_config_name: str,
    cwd: str,
    timeout: int,
    extra_args: list,
) -> Tuple[bool, str]:
    cmd = [
        sys.executable, build_script,
        os.path.basename(onnx_dir),
        "--model-name",     model_name,
        "--folding-config", folding_config_name,
    ] + extra_args
    try:
        result = subprocess.run(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, text=True,
        )
        success = result.returncode == 0
        if not success:
            log.warning("Build exited %d. Last 4 kB:\n%s",
                        result.returncode, result.stdout[-4000:])
        return success, result.stdout
    except subprocess.TimeoutExpired:
        log.error("Build timed out after %d s.", timeout)
        return False, "TIMEOUT"
    except Exception as exc:
        log.error("Build subprocess error: %s", exc)
        return False, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# CSV results logger
# ─────────────────────────────────────────────────────────────────────────────

# CSV columns: trial_name, build_success, objective, throughput_fps,
#              lut, lut_pct, bram, bram_pct, dsp, dsp_pct, ff, ff_pct,
#              wns, fmax_mhz, + all trial_params keys

def _append_to_csv(
    csv_path: str,
    trial_name: str,
    trial_params: Dict[str, Any],
    metrics: Dict[str, float],
    obj_value: float,
    success: bool,
) -> None:
    """Append one trial result row to the running CSV."""
    import fcntl

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
        "lut_pct":  round(lut  / DEVICE_LUT  * 100, 2) if lut  else "",
        "bram":     bram,
        "bram_pct": round(bram / DEVICE_BRAM * 100, 2) if bram else "",
        "dsp":      dsp,
        "dsp_pct":  round(dsp  / DEVICE_DSP  * 100, 2) if dsp  else "",
        "ff":       ff,
        "ff_pct":   round(ff   / DEVICE_FF   * 100, 2) if ff   else "",
        "wns":      metrics.get("wns",      ""),
        "fmax_mhz": metrics.get("fmax_mhz", ""),
    }
    # Add all trial_params as columns
    row.update({k: v for k, v in trial_params.items()})

    file_exists = os.path.exists(csv_path)
    lock_path   = csv_path + ".lock"

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()),
                                        extrasaction="ignore")
                if not file_exists or os.path.getsize(csv_path) == 0:
                    writer.writeheader()
                writer.writerow(row)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def create_all_ones_params(baseline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scans the baseline config and generates a parameter dict setting all PE and SIMD to 1.
    Universal version: supports both 'MVAU_0' and 'MVAU_hls_0' formats.
    """
    params = {}
    for node_name, attrs in baseline_cfg.items():
        if not isinstance(attrs, dict) or node_name == "Defaults":
            continue
            
        # Normalize the name to its logical format (e.g., MVAU_hls_0 -> MVAU_0)
        logical_name = node_name.replace("_hls_", "_").replace("_rtl_", "_")
        
        # We only want to record each logical layer once
        if "PE" in attrs and f"{logical_name}__PE" not in params:
            params[f"{logical_name}__PE"] = 1
            
        if "SIMD" in attrs and f"{logical_name}__SIMD" not in params:
            params[f"{logical_name}__SIMD"] = 1
            
    return params

def run_feasibility_check(
    baseline_cfg: Dict[str, Any],
    algo_config_root: str,
    dataset_root: str,
    model_stem: str,
    model_name: str,
    onnx_dir: str,
    build_script: str,
    build_timeout: int,
    extra_build_args: list,
    objective: str,
    results_csv: str,
) -> None:
    """Runs a minimal hardware build. Aborts the entire script if it fails."""
    log.info("═══ Running Minimum Feasibility Check (All PE/SIMD = 1) ═══")
    trial_name = "feasibility_check"
    trial_config_dir = os.path.join(algo_config_root, trial_name)
    os.makedirs(trial_config_dir, exist_ok=True)

    trial_params = create_all_ones_params(baseline_cfg)
    patched_cfg = patch_folding_config(baseline_cfg, trial_params)

    cfg_filename = f"{trial_name}.json"
    cfg_finn_path = os.path.join(dataset_root, "dataset", "config_files", model_stem, cfg_filename)
    save_folding_config(patched_cfg, cfg_finn_path)

    with open(os.path.join(trial_config_dir, "trial_params.json"), "w") as f:
        json.dump(trial_params, f, indent=2)

    finn_output_dir = os.path.join(dataset_root, "dataset", "results_synth", model_stem, trial_name)

    success, build_log = run_finn_build(
        build_script=build_script,
        onnx_dir=onnx_dir,
        model_name=model_name,
        folding_config_name=cfg_filename,
        cwd=dataset_root,
        timeout=build_timeout,
        extra_args=extra_build_args,
    )

    with open(os.path.join(trial_config_dir, "build.log"), "w") as f:
        f.write(build_log)

    metrics = {}
    obj_value = PENALTY_HARD
    if success:
        metrics = collect_metrics(finn_output_dir)
        if metrics.get("finn_success", 1) == 0:
            success = False
        else:
            obj_value = compute_objective(metrics, objective)
            # Check if it failed due to over-utilization
            if obj_value >= PENALTY_HARD:
                success = False

    # Write this result to the CSV so the LLM and Optuna can see it!
    _append_to_csv(results_csv, trial_name, trial_params, metrics, obj_value, success)

    if not success:
        log.error("🚨 FEASIBILITY CHECK FAILED!")
        log.error("The model is too large for this board even at PE=1, SIMD=1.")
        log.error("Halting execution. Please check the model architecture or select a larger FPGA.")
        sys.exit(1)
    else:
        log.info("✅ Feasibility check passed! The 'all-ones' build is recorded in the CSV.")
        log.info("   LUT: %s | BRAM: %s | DSP: %s", 
                 metrics.get('lut', 0), metrics.get('bram', 0), metrics.get('dsp', 0))

# ─────────────────────────────────────────────────────────────────────────────
# Trainable
# ─────────────────────────────────────────────────────────────────────────────

def make_trainable(
    baseline_cfg:          Dict[str, Any],
    build_script:          str,
    onnx_path:             str,
    dataset_root:          str,
    objective:             str,
    search_strategy:       str,
    build_timeout:         int,
    extra_build_args:      list,
    parallelism_threshold: int,
    database_dir:          Optional[str],
    results_csv:           str,
):
    model_stem = Path(onnx_path).stem
    onnx_dir   = str(Path(onnx_path).parent)
    model_name = Path(onnx_path).name
    algo_config_root = os.path.join(
        dataset_root, "dataset", "config_files", model_stem, search_strategy
    )

    def trainable(trial_config: Dict[str, Any]):
        import os, json, shutil, logging
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)

        patched_cfg = patch_folding_config(baseline_cfg, trial_config)
        trial_num   = _next_trial_number(algo_config_root)
        trial_name  = f"config_{trial_num:04d}"

        trial_config_dir = os.path.join(algo_config_root, trial_name)
        os.makedirs(trial_config_dir, exist_ok=True)

        cfg_filename  = f"{search_strategy}_{trial_name}.json"
        cfg_finn_path = os.path.join(
            dataset_root, "dataset", "config_files", model_stem, cfg_filename
        )
        save_folding_config(patched_cfg, cfg_finn_path)
        shutil.copy(cfg_finn_path, os.path.join(trial_config_dir, "folding_config.json"))
        with open(os.path.join(trial_config_dir, "trial_params.json"), "w") as f:
            json.dump(trial_config, f, indent=2)

        cfg_stem        = os.path.splitext(cfg_filename)[0]
        finn_output_dir = os.path.join(
            dataset_root, "dataset", "results_synth", model_stem, cfg_stem
        )

        total   = total_parallelism(trial_config)
        success = False
        metrics = {}

        # ── Threshold check ────────────────────────────────────────────────
        if total > parallelism_threshold:
            n_skip = _increment_counter(algo_config_root, "skipped")
            log.info("[%s] SKIPPED (parallelism %d > %d) | skipped=%d",
                     trial_name, total, parallelism_threshold, n_skip)
            os.makedirs(finn_output_dir, exist_ok=True)
            with open(os.path.join(finn_output_dir, "status.json"), "w") as f:
                json.dump({"success": 0,
                           "error_message": f"skipped: parallelism {total} > threshold"}, f)
            obj_value = PENALTY

        # ── Database lookup ────────────────────────────────────────────────
        elif database_dir and lookup_database(trial_config, database_dir, finn_output_dir):
            n_hit = _increment_counter(algo_config_root, "cache_hit")
            log.info("[%s] CACHE HIT (parallelism %d) | cache=%d",
                     trial_name, total, n_hit)
            metrics = collect_metrics(finn_output_dir)
            success = metrics.get("finn_success", 0) == 1
            obj_value = compute_objective(metrics, objective) if success else PENALTY

        # ── Full build ─────────────────────────────────────────────────────
        else:
            n_built = _increment_counter(algo_config_root, "built")
            log.info("[%s] BUILDING (parallelism %d) | built=%d",
                     trial_name, total, n_built)
            success, build_log = run_finn_build(
                build_script        = build_script,
                onnx_dir            = onnx_dir,
                model_name          = model_name,
                folding_config_name = cfg_filename,
                cwd                 = dataset_root,
                timeout             = build_timeout,
                extra_args          = extra_build_args,
            )
            with open(os.path.join(trial_config_dir, "build.log"), "w") as f:
                f.write(build_log)
            # ── 6. Parse reports ───────────────────────────────────────────────
        if not success:
            obj_value = PENALTY
            metrics   = {}
        else:
            metrics = collect_metrics(finn_output_dir)
            if metrics.get("finn_success", 1) == 0:
                log.warning("status.json reports FINN compiler failure (%s).", trial_name)
                obj_value = PENALTY
                success = False
            else:
                obj_value = compute_objective(metrics, objective)
                
                # OVERRIDE: If the hardware limits are exceeded, mark it as a build failure
                if obj_value >= PENALTY_HARD:
                    log.warning("[%s] Estimation exceeds physical board limits. Marking as failed.", trial_name)
                    success = False
                    
                    # Physically overwrite status.json on disk so downstream scripts know it failed
                    status_file = os.path.join(finn_output_dir, "status.json")
                    if os.path.exists(status_file):
                        with open(status_file, "r") as f:
                            st_data = json.load(f)
                        st_data["success"] = 0
                        st_data["error_message"] = "Estimation exceeds physical board limits (Over-utilization)."
                        with open(status_file, "w") as f:
                            json.dump(st_data, f, indent=4)
                else:
                    success = True
                    _increment_counter(algo_config_root, "built_success")

        # ── Append to CSV ──────────────────────────────────────────────────
        _append_to_csv(results_csv, trial_name, trial_config,
                       metrics, obj_value, success)

        # Read the current state of all thread-safe counters
        n_built   = _read_counter(algo_config_root, "built")
        n_success = _read_counter(algo_config_root, "built_success")
        n_hit     = _read_counter(algo_config_root, "cache_hit")
        n_skip    = _read_counter(algo_config_root, "skipped")

        # Print the live summary
        log.info(
            "[%s] obj=%.4g | STATS -> Built: %d (Success: %d) | Cache Hits: %d | Skipped: %d",
            trial_name, obj_value, n_built, n_success, n_hit, n_skip
        )

        report_dict = {
            "objective":     obj_value,
            "build_success": int(success),
            "trial_name":    trial_name,
        }
        report_dict.update({k: v for k, v in metrics.items() if k != "finn_success"})
        tune.report(report_dict)

    return trainable


# ─────────────────────────────────────────────────────────────────────────────
# Search algorithm factory
# ─────────────────────────────────────────────────────────────────────────────

def build_search_alg(
    strategy:          str,
    seed:              int,
    max_concurrent:    int,
    study_db:          Optional[str] = None,
    study_name:        Optional[str] = None,
    warm_start_points: Optional[List[Dict[str, Any]]] = None,
):
    """
    Create the search algorithm object with warm-starting capabilities.
    """
    if strategy == "optuna":
        import optuna
        search_alg = OptunaSearch(
            storage            = optuna.storages.RDBStorage(url=study_db) if study_db else None,
            study_name         = study_name,
            metric             = "objective",
            mode               = "min",
            points_to_evaluate = warm_start_points, 
        )
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent)
        log.info("Search: Optuna (isolated study) — max_concurrent=%d", max_concurrent)

    elif strategy == "nevergrad":
        import nevergrad
        from ray.tune.search.nevergrad import NevergradSearch
        search_alg = NevergradSearch(
            optimizer = nevergrad.optimizers.OnePlusOne,
            metric    = "objective",
            mode      = "min",
            points_to_evaluate = warm_start_points,
        )
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent)
        log.info("Search: Nevergrad OnePlusOne")

    elif strategy == "nevergrad_de":
        import nevergrad
        from ray.tune.search.nevergrad import NevergradSearch
        search_alg = NevergradSearch(
            optimizer = nevergrad.optimizers.TwoPointsDE,
            metric    = "objective",
            mode      = "min",
            points_to_evaluate = warm_start_points,
        )
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent)
        log.info("Search: Nevergrad TwoPointsDE")

    elif strategy == "hyperopt":
        from ray.tune.search.hyperopt import HyperOptSearch
        search_alg = HyperOptSearch(
            metric            = "objective",
            mode              = "min",
            random_state_seed = seed,
            points_to_evaluate= warm_start_points,
        )
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent)
        log.info("Search: HyperOpt TPE")

    else:  # random
        from ray.tune.search.basic_variant import BasicVariantGenerator
        search_alg = BasicVariantGenerator(
            random_state   = seed,
            max_concurrent = max_concurrent,
        )
        log.info("Search: Random — seed=%d", seed)

    return search_alg


# ─────────────────────────────────────────────────────────────────────────────
# Post-run utilities  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def save_best_config(
    analysis,
    baseline: Dict[str, Any],
    output_path: str,
) -> None:
    try:
        best_params = analysis.get_best_config(metric="objective", mode="min")
    except Exception as exc:
        log.warning("Could not retrieve best config: %s", exc)
        return
    if best_params is None:
        log.warning("No best config found.")
        return
    best_cfg = patch_folding_config(baseline, best_params)
    save_folding_config(best_cfg, output_path)
    log.info("Best config saved to: %s", output_path)
    for k, v in sorted(best_params.items()):
        log.info("  %-45s = %s", k, v)


def print_trial_summary(analysis) -> None:
    try:
        df = analysis.results_df
    except Exception:
        return
    if df is None or df.empty:
        return
    cols = [c for c in ["objective", "lut", "bram", "dsp", "ff",
                         "wns", "fmax_mhz", "throughput_fps", "build_success"]
            if c in df.columns]
    if not cols:
        return
    sort_col = "objective" if "objective" in df.columns else cols[0]
    print("\n═══════════════ Round Summary ═══════════════")
    print(df[cols].sort_values(sort_col).to_string(index=True))
    print("═════════════════════════════════════════════\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FINN folding tuner with adaptive LLM space pruning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline_cfg",   required=True)
    p.add_argument("--build_cfg",      required=True,
                   help="build_config.json passed to space_pruner")
    p.add_argument("--build_script",   required=True)
    p.add_argument("--onnx_path",      required=True)
    p.add_argument("--best_cfg_out",   default=None)
    p.add_argument("--raytune_dir",    default="./raytune_logs")
    p.add_argument("--epoch_schedule", type=str, default="100,100,100,100,100",
                   help="Comma-separated list of trials per LLM pruning round.")
    p.add_argument("--max_concurrent", type=int, default=1)
    p.add_argument("--objective",
                   choices=["lut", "lut_slack", "throughput",
                             "resource_avg", "balanced"],
                   default="balanced")
    p.add_argument("--build_timeout",  type=int, default=7200)
    p.add_argument("--search_strategy",
                   choices=["optuna", "random", "nevergrad",
                             "nevergrad_de", "hyperopt"],
                   default="optuna")
    p.add_argument("--seed",           type=int, default=20)
    p.add_argument("--constraints",    type=str,
                   default="Maximise throughput while minimising resource usage.",
                   help="Natural language constraints passed to space_pruner")
    p.add_argument("--parallelism_threshold", type=int, default=1000000)
    p.add_argument("--database_dir",   default=None)
    p.add_argument("--groq_api_key",   default=None,
                   help="API key")
    p.add_argument("--prune_on_start", action="store_true",
                   help="Call space_pruner once before the first round")
    p.add_argument("--extra_build_args", nargs=argparse.REMAINDER, default=[])
    p.add_argument("--ray_address",    default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main — round-based loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.raytune_dir = os.path.abspath(args.raytune_dir)

    for label, path in [("baseline_cfg", args.baseline_cfg),
                         ("build_script",  args.build_script),
                         ("build_cfg",     args.build_cfg),
                         ("onnx_path",     args.onnx_path)]:
        if not os.path.exists(path):
            log.error("%s not found: %s", label, path)
            sys.exit(1)

    onnx_abs     = os.path.abspath(args.onnx_path)
    model_stem   = Path(onnx_abs).stem
    dataset_root = str(Path(onnx_abs).parent.parent.parent)

    model_config_root = os.path.join(
        dataset_root, "dataset", "config_files", model_stem
    )
    algo_config_root = os.path.join(model_config_root, args.search_strategy)
    os.makedirs(algo_config_root, exist_ok=True)

    best_cfg_out = args.best_cfg_out or os.path.join(
        model_config_root, f"folding_config_best_{args.search_strategy}.json"
    )

    # Shared pruned space JSON — overwritten by pruner each round
    pruned_space_path = os.path.join(
        algo_config_root, "pruned_space.json"
    )

    # Running CSV — appended to by every trial
    results_csv = os.path.join(algo_config_root, "trial_results.csv")

    experiment_name = f"finn_tune_{args.search_strategy}_{args.objective}"

    baseline = load_folding_config(args.baseline_cfg)
    
    # ── Fail-Fast Feasibility Check ─────────────────────────────────────
    # Only run if this is a completely new experiment (no CSV exists yet)
    if not os.path.exists(results_csv):
        run_feasibility_check(
            baseline_cfg     = baseline,
            algo_config_root = algo_config_root,
            dataset_root     = dataset_root,
            model_stem       = model_stem,
            model_name       = Path(onnx_abs).name,
            onnx_dir         = str(Path(onnx_abs).parent),
            build_script     = os.path.abspath(args.build_script),
            build_timeout    = args.build_timeout,
            extra_build_args = args.extra_build_args,
            objective        = args.objective,
            results_csv      = results_csv,
        )

    # ── Optuna Database setup ──────────────────────────────────────────────
    study_db = None
    if args.search_strategy == "optuna":
        study_db = f"sqlite:///{algo_config_root}/optuna_study.db"

    ray.init(address=args.ray_address, ignore_reinit_error=True)

    # ── Determine pruning schedule ─────────────────────────────────────────
    try:
        epoch_schedule = [int(x.strip()) for x in args.epoch_schedule.split(',')]
    except ValueError:
        log.error("Invalid --epoch_schedule. Must be comma-separated integers (e.g., '50,100,250,600').")
        sys.exit(1)

    total_samples = sum(epoch_schedule)
    pruning_round = 0

    # ── Optional initial pruning before any builds ─────────────────────────
    if args.prune_on_start or not Path(pruned_space_path).exists():
        log.info("═══ Initial pruning (round 0) ═══")
        run_pruner(
            build_cfg_path    = args.build_cfg,
            baseline_cfg_path = args.baseline_cfg,
            out_json_path     = pruned_space_path,
            constraints       = args.constraints,
            results_csv       = None,  # no history yet
            pruning_round     = 0,
            api_key           = args.groq_api_key,
        )
        pruning_round += 1
    
    # ── Round-based execution loop ─────────────────────────────────────────
    trials_done = 0
    best_analysis = None

    for round_samples in epoch_schedule:
        log.info("═══ Pruning round %d | trials %d–%d of %d ═══",
                 pruning_round, trials_done + 1,
                 trials_done + round_samples, total_samples)

        # Load current search space
        if not Path(pruned_space_path).exists():
            log.error("pruned_space.json not found at %s.", pruned_space_path)
            sys.exit(1)

        search_space = load_search_space(pruned_space_path)
        log.info("Search space: %d hyperparameters.", len(search_space))

        # Recreate Optuna study with a unique name per round to prevent category crashes
        current_study_name = None
        if args.search_strategy == "optuna":
            current_study_name = f"finn_{model_stem}_round{pruning_round}"

        # ── KNOWLEDGE TRANSFER (WARM START) ──
        warm_start_points = []
        if Path(results_csv).exists():
            warm_start_points = get_warm_start_points(results_csv, pruned_space_path)
            if warm_start_points:
                log.info("Injecting %d past successful trials to warm-start the optimizer.", len(warm_start_points))

        # Recreate search algorithm with warm start points
        search_alg = build_search_alg(
            strategy          = args.search_strategy,
            seed              = args.seed + pruning_round,
            max_concurrent    = args.max_concurrent,
            study_db          = study_db,
            study_name        = current_study_name,
            warm_start_points = warm_start_points if warm_start_points else None
        )

        trainable = make_trainable(
            baseline_cfg          = baseline,
            build_script          = os.path.abspath(args.build_script),
            onnx_path             = onnx_abs,
            dataset_root          = dataset_root,
            objective             = args.objective,
            search_strategy       = args.search_strategy,
            build_timeout         = args.build_timeout,
            extra_build_args      = args.extra_build_args,
            parallelism_threshold = args.parallelism_threshold,
            database_dir          = args.database_dir,
            results_csv           = results_csv,
        )

        round_name = f"{experiment_name}_round{pruning_round:02d}"

        reporter = CLIReporter(
            metric_columns       = ["objective", "lut", "dsp", "throughput_fps", "build_success"],
            max_report_frequency = 60,
        )

        analysis = tune.run(
            trainable,
            config                = search_space,
            metric                = "objective",
            mode                  = "min",
            num_samples           = round_samples,
            search_alg            = search_alg,
            progress_reporter     = reporter,
            storage_path          = args.raytune_dir,
            name                  = round_name,
            verbose               = 1,
            raise_on_failed_trial = False,
            resources_per_trial   = {"cpu": 2, "gpu": 0},
        )

        print_trial_summary(analysis)
        best_analysis = analysis
        trials_done  += round_samples
        
        # ── Auto-Integrate into Database ───────────────────────────────────────
        if args.database_dir:
            if "--estimate-only" in args.extra_build_args:
                log.info("═══ Skipping Database Integration (Estimation-Only Mode) ═══")
            else:
                log.info("═══ Integrating Round %d builds into Database ═══", pruning_round)
                try:
                    source_dir = os.path.join(dataset_root, "dataset")
                    trials_to_integrate = scan_source(source_dir, model_stem)
                    integrate(trials_to_integrate, args.database_dir, dry_run=False)
                except Exception as e:
                    log.error("Database auto-integration failed: %s", e)

        pruning_round += 1

        # ── Call space_pruner for next round (unless we are done) ──────────────
        if trials_done < total_samples:
            log.info("═══ Calling space_pruner (round %d, %d trials recorded) ═══",
                     pruning_round, trials_done)
            run_pruner(
                build_cfg_path    = args.build_cfg,
                baseline_cfg_path = args.baseline_cfg,
                out_json_path     = pruned_space_path,
                constraints       = args.constraints,
                results_csv       = results_csv,
                pruning_round     = pruning_round,
                api_key           = args.groq_api_key,
            )

            # ── DETERMINISTIC LLM GUARDRAIL ──
            # Check the actual failure rate of the last round
            last_round_successes = 0
            trials_checked = 0
            if os.path.exists(results_csv):
                with open(results_csv, 'r') as f:
                    # Read the last N trials (where N is round_samples)
                    rows = list(csv.DictReader(f))[-round_samples:]
                    trials_checked = len(rows)
                    last_round_successes = sum(1 for r in rows if r.get('build_success') == '1')
            
            # If 100% of recent builds failed, DO NOT trust the LLM. Clamp the JSON physically.
            if trials_checked > 0 and last_round_successes <= 1:
                log.warning("🚨 EMERGENCY GUARDRAIL TRIGGERED: Clamping JSON to max value of 8.")
                with open(pruned_space_path, 'r') as f:
                    unsafe_space = json.load(f)
                
                safe_space = {}
                for key, arr in unsafe_space.items():
                    # Keep only values <= 8. If none exist (e.g., [5, 10]), keep just the absolute minimum.
                    clamped_arr = [x for x in arr if x <= 8]
                    safe_space[key] = clamped_arr if clamped_arr else [min(arr)]
                
                with open(pruned_space_path, 'w') as f:
                    json.dump(safe_space, f, indent=2)

    # ── Save final best config ─────────────────────────────────────────────
    if best_analysis:
        save_best_config(best_analysis, baseline, best_cfg_out)

    ray.shutdown()
    log.info("Done. Total trials: %d", trials_done)


if __name__ == "__main__":
    main()