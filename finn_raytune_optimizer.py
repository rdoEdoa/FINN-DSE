"""
finn_raytune_optimizer.py
─────────────────────────────────────────────────────────────────────────────
Automatic FINN folding-config tuner using Ray Tune.

Workflow
────────
1.  Load a baseline folding config (JSON) produced by generate_folding_config.py.
2.  Build a search space: one PE and one SIMD hyperparameter per logical layer.
    Each logical layer covers both its _hls_ and _rtl_ config variants.
    Valid choices are constrained to divisors of MH / MW so FINN never
    receives an illegal value.
3.  For each trial Ray Tune samples PE/SIMD values, writes a patched config,
    calls full_build.py as a subprocess, then reads the JSON reports it
    produces to compute a scalar objective.
4.  After all trials the best config is saved.

Usage
─────
    python finn_raytune_optimizer.py \
        --baseline_cfg      dataset/config_files/resnet50/folding_config_baseline.json \
        --build_script      custom_builds/resnet50_build.py \
        --onnx_path         dataset/resnet50/resnet50.onnx \
        --database_dir      database/resnet50 \
        --num_samples       1000 \
        --objective         throughput \
        --search_strategy   nevergrad \
        --max_concurrent    10 \
        --extra_build_args  --estimate-only
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

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

# Candidate PE/SIMD values. Only divisors of MH/MW are actually offered to
# Ray Tune, so invalid values are never sampled.
VALID_PE_SIMD = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Device constants 
DEVICE_LUT  = 1727040
DEVICE_DSP  = 12288
DEVICE_BRAM = 2688
DEVICE_FF   = 3454080

# Maximum allowed utilisation fraction before penalising
MAX_UTIL = 0.85

# Sentinel returned when a build fails or reports are missing
PENALTY = 10
PENALTY_HARD = 10000

# JSON report filenames written by full_build.py
# (confirmed from the results-parsing script)
REPORT_FILES = {
    "status" : "status.json",                # success flag + error message
    "synth"  : "ooc_synth_and_timing.json",  # LUT, BRAM, DSP, FF, WNS, fmax_mhz
    "perf"   : "rtlsim_performance.json",    # throughput[images/s]
}

# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe trial counter
# ─────────────────────────────────────────────────────────────────────────────
# Ray workers run in separate processes, so we use a file-based counter stored
# in the config directory. A lock file serialises access so concurrent workers
# never get the same number.

def _next_trial_number(config_dir: str) -> int:
    """
    Return the next 0-based trial number and atomically increment the counter.
    The counter is stored in <config_dir>/.trial_counter so it persists across
    restarts and is shared by all Ray worker processes.
    """
    import fcntl
    os.makedirs(config_dir, exist_ok=True)
    counter_path = os.path.join(config_dir, ".trial_counter")
    lock_path    = counter_path + ".lock"

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            if os.path.exists(counter_path):
                with open(counter_path) as f:
                    n = int(f.read().strip())
            else:
                n = 0
            with open(counter_path, "w") as f:
                f.write(str(n + 1))
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Trial counters (skipped / cache-hit / built)
# ─────────────────────────────────────────────────────────────────────────────

def _increment_counter(counter_dir: str, name: str) -> int:
    """
    Atomically increment a named counter stored in counter_dir.
    Returns the new value. Uses the same file-lock pattern as _next_trial_number.
    """
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
# Parallelism threshold
# ─────────────────────────────────────────────────────────────────────────────

def total_parallelism(trial_params: Dict[str, Any]) -> int:
    """Sum all PE and SIMD values in a trial config dict."""
    total = 0
    for v in trial_params.values():
        try:
            total += int(v)
        except (TypeError, ValueError):
            pass
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Database lookup
# ─────────────────────────────────────────────────────────────────────────────

def _configs_match(trial_params: Dict[str, Any], db_params_path: str) -> bool:
    """
    Compare a trial_params dict with a database trial_params.json file.
    Returns True if every key-value pair is identical.
    """
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
    """
    Search database_dir for a config matching trial_params.
    If found, copy all files from the database entry into finn_output_dir.
    Returns True if a match was found and copied, False otherwise.

    Database layout expected:
        database/config_<xxxxx>/
            trial_params.json   ← hyperparameter values to match against
            status.json         ← build result files to copy
            ooc_synth_and_timing.json
            rtlsim_performance.json
            ...
    """
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
            # Match found — copy all result files to finn_output_dir
            os.makedirs(finn_output_dir, exist_ok=True)
            for src_file in entry.iterdir():
                if src_file.is_file():
                    shutil.copy(src_file, os.path.join(finn_output_dir, src_file.name))
            return True
    return False



def load_folding_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


# Keys that FINN's step_apply_folding_config actually recognises.
# Any extra keys we add for bookkeeping (MH, MW, op_type) must be stripped
# before writing the config that gets passed to full_build.py.
FINN_CONFIG_KEYS = {
    "PE", "SIMD", "ram_style", "mem_mode", "resType",
    "preferred_impl_style", "runtime_writeable_weights",
}


def _sanitize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively walk the config and:
      1. Convert whole-number floats and numpy scalars to Python int.
      2. Strip any key that is not in FINN_CONFIG_KEYS from non-Defaults entries
         so FINN's step_apply_folding_config never tries to subscript a scalar.
    The "Defaults" entry is passed through unchanged.
    """
    result = {}
    for node_name, node_val in cfg.items():
        if not isinstance(node_val, dict):
            result[node_name] = node_val
            continue

        clean = {}
        for k, v in node_val.items():
            # Defaults must be empty — FINN applies every key in it to all nodes
            if node_name == "Defaults":
                continue
            # For layer entries keep only FINN-recognised keys
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


def _split_config_key(node_name: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse a config key such as "MVAU_hls_0" or "ConvolutionInputGenerator_rtl_2"
    into (layer_type, variant, layer_index).

    The regex anchors on the LAST occurrence of _(hls|rtl)_<digits> so that
    layer type names containing underscores (e.g. StreamingDataWidthConverter)
    are handled correctly.

    Returns None for keys that do not match (e.g. "Defaults").
    """
    m = re.match(r"^(.+)_(hls|rtl)_(\d+)$", node_name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


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


# ─────────────────────────────────────────────────────────────────────────────
# Search-space builder
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict, Any, Tuple
from ray import tune
import logging

log = logging.getLogger(__name__)

def build_search_space(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Universal hyperparameter search space builder.
    Works for BOTH old PyTorch ONNX exports (with _hls_ / _rtl_ variants)
    and new hardware-extracted netlists (clean topological names).
    """
    logical_configs = {}

    # ─── STEP 1: Normalize and Deduplicate ──────────────────────────────────
    for node_name, node_cfg in baseline.items():
        if node_name == "Defaults":
            continue
            
        # Strip the _hls_ or _rtl_ tag if it exists (Backward Compatibility)
        # e.g., "MVAU_hls_14" -> "MVAU_14". If it's already "MVAU_14", nothing changes.
        logical_name = node_name.replace("_hls_", "_").replace("_rtl_", "_")
        
        # If it's the first time we see this logical layer, OR if it's the 'hls' variant, save it.
        # (This perfectly deduplicates the old format while safely passing the new format).
        if logical_name not in logical_configs or "_hls_" in node_name:
            logical_configs[logical_name] = node_cfg

    space: Dict[str, Any] = {}

    # ─── STEP 2: Dynamically calculate mathematical divisors ────────────────
    for logical_name, node_cfg in sorted(logical_configs.items()):
        mh = node_cfg.get("MH", None)
        
        # IFMChannels limits SIMD for InputGenerators; MW limits it for MVAUs
        limit_dim = node_cfg.get("IFMChannels", node_cfg.get("MW", None))

        # Build PE Search Space
        # We tune PE if MH is present (meaning it's an MVAU compute layer)
        if mh is not None:
            valid_pe = [v for v in range(1, mh + 1) if mh % v == 0]
            space[f"{logical_name}__PE"] = tune.choice(valid_pe)
        elif "PE" in node_cfg: 
            # Catch case for legacy configs that had PE but missing MH
            log.warning("Layer %s: MH is missing — skipping PE.", logical_name)

        # Build SIMD Search Space
        # We tune SIMD if a limiting dimension exists
        if limit_dim is not None:
            valid_simd = [v for v in range(1, limit_dim + 1) if limit_dim % v == 0]
            if valid_simd:
                space[f"{logical_name}__SIMD"] = tune.choice(valid_simd)
            else:
                log.warning("Layer %s: no valid SIMD found.", logical_name)
        elif "SIMD" in node_cfg:
             log.warning("Layer %s: Both IFMChannels and MW are missing — skipping SIMD.", logical_name)

    return space


# ─────────────────────────────────────────────────────────────────────────────
# Report parsing
# ─────────────────────────────────────────────────────────────────────────────

def _read_json(directory: str, filename: str) -> Optional[dict]:
    """Read a JSON file from directory. Returns None if missing or corrupt."""
    path = Path(directory) / filename
    if not path.exists():
        return None
    try:
        content = path.read_text().strip()
        if not content:
            return None
        return json.loads(content)
    except json.JSONDecodeError as exc:
        log.warning("Could not parse %s: %s", path, exc)
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


# ─────────────────────────────────────────────────────────────────────────────
# Objective function
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
# FINN build runner
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
    """
    Run full_build.py as a subprocess from `cwd` so that its relative
    path assumptions (dataset/ root) are always satisfied regardless of
    where the optimizer itself was launched from.

    CLI:  python full_build.py <onnx_dir_basename>
              --model-name     <model_name>
              --folding-config <config_name>
    """
    cmd = [
        sys.executable, build_script,
        os.path.basename(onnx_dir),   # positional: basename only
        "--model-name",     model_name,
        "--folding-config", folding_config_name,
    ] + extra_args

    log.info("Build command : %s", " ".join(cmd))
    log.info("Working dir   : %s", cwd)

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
        )
        success = result.returncode == 0
        if not success:
            log.warning(
                "Build exited with code %d. Last 4 kB of output:\n%s",
                result.returncode, result.stdout[-4000:],
            )
        return success, result.stdout

    except subprocess.TimeoutExpired:
        log.error("Build timed out after %d s.", timeout)
        return False, "TIMEOUT"
    except Exception as exc:
        log.error("Build subprocess error: %s", exc)
        return False, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Ray Tune trainable
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
):
    """
    Return a Ray Tune trainable closure.

    For each trial:
      1. Check total PE+SIMD against parallelism_threshold — skip if over.
      2. Look up the config in database_dir — copy results if found.
      3. Otherwise run the full FINN build.

    Directory layout
    ────────────────
    dataset/config_files/<model>/<search_strategy>/
        config_0000/
            folding_config.json
            trial_params.json
            build.log
        config_0000.json
    dataset/results_synth/<model>/<search_strategy>_config_0000/
    """
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

        # ── 1. Patch config ────────────────────────────────────────────────
        patched_cfg = patch_folding_config(baseline_cfg, trial_config)

        # ── 2. Assign sequential trial number ─────────────────────────────
        trial_num  = _next_trial_number(algo_config_root)
        trial_name = f"config_{trial_num:04d}"

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

        # ── 3. Threshold check ─────────────────────────────────────────────
        total = total_parallelism(trial_config)

        if total > parallelism_threshold:
            n_skip  = _increment_counter(algo_config_root, "skipped")
            n_hit   = _read_counter(algo_config_root, "cache_hit")
            n_built = _read_counter(algo_config_root, "built")
            n_succ  = _read_counter(algo_config_root, "built_success")
            log.info(
                "[%s] SKIPPED (parallelism %d > %d) | retrieved: %d, skipped: %d, executed: %d (successes: %d)",
                trial_name, total, parallelism_threshold, n_hit, n_skip, n_built, n_succ
            )
            # Write a synthetic failed status so results parsers work normally
            os.makedirs(finn_output_dir, exist_ok=True)
            with open(os.path.join(finn_output_dir, "status.json"), "w") as f:
                json.dump({
                    "success": 0,
                    "error_message": f"skipped: parallelism {total} > threshold {parallelism_threshold}",
                }, f)
            tune.report({"objective": PENALTY, "build_success": 0, "trial_name": trial_name})
            return

        # ── 4. Database lookup ─────────────────────────────────────────────
        if database_dir and lookup_database(trial_config, database_dir, finn_output_dir):
            n_hit   = _increment_counter(algo_config_root, "cache_hit")
            n_skip  = _read_counter(algo_config_root, "skipped")
            n_built = _read_counter(algo_config_root, "built")
            n_succ  = _read_counter(algo_config_root, "built_success")
            log.info(
                "[%s] CACHE HIT (parallelism %d) | retrieved: %d, skipped: %d, executed: %d (successes: %d)",
                trial_name, total, n_hit, n_skip, n_built, n_succ
            )
            # Results already copied — parse them directly
            metrics   = collect_metrics(finn_output_dir)
            obj_value = compute_objective(metrics, objective) \
                        if metrics.get("finn_success", 0) == 1 else PENALTY
            report_dict = {
                "objective":     obj_value,
                "build_success": int(metrics.get("finn_success", 0)),
                "trial_name":    trial_name,
            }
            report_dict.update({k: v for k, v in metrics.items() if k != "finn_success"})
            tune.report(report_dict)
            return

        # ── 5. Run FINN build ──────────────────────────────────────────────
        n_built = _increment_counter(algo_config_root, "built")
        log.info("[%s] BUILDING  (parallelism %d)", trial_name, total)

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

        n_skip  = _read_counter(algo_config_root, "skipped")
        n_hit   = _read_counter(algo_config_root, "cache_hit")
        n_built = _read_counter(algo_config_root, "built")
        n_succ  = _read_counter(algo_config_root, "built_success")
        
        log.info(
            "[%s] finished — objective: %.4g | retrieved: %d, skipped: %d, executed: %d (successes: %d)", 
            trial_name, obj_value, n_hit, n_skip, n_built, n_succ
        )

        # ── 7. Report ──────────────────────────────────────────────────────
        report_dict = {
            "objective":     obj_value,
            "build_success": int(success),
            "trial_name":    trial_name,
        }
        report_dict.update({k: v for k, v in metrics.items() if k != "finn_success"})
        tune.report(report_dict)

    return trainable


# ─────────────────────────────────────────────────────────────────────────────
# Post-run utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_best_config(analysis, baseline: Dict[str, Any], output_path: str) -> None:
    try:
        best_params = analysis.get_best_config(metric="objective", mode="min")
    except Exception as exc:
        log.warning("Could not retrieve best config: %s", exc)
        best_params = None

    if best_params is None:
        log.warning("No best config found — nothing saved.")
        return

    best_cfg = patch_folding_config(baseline, best_params)
    save_folding_config(best_cfg, output_path)
    log.info("Best folding config saved to: %s", output_path)

    log.info("Best hyperparameters:")
    for k, v in sorted(best_params.items()):
        log.info("  %-45s = %s", k, v)

    try:
        best_result = analysis.best_result
        log.info("Best trial metrics:")
        for k, v in sorted(best_result.items()):
            if not k.startswith("config/"):
                log.info("  %-45s = %s", k, v)
    except Exception:
        pass


def print_trial_summary(analysis) -> None:
    try:
        df = analysis.results_df
    except Exception:
        log.info("No trial results available.")
        return
    if df is None or df.empty:
        log.info("No trial results available.")
        return
    cols = [c for c in ["objective", "lut", "bram", "dsp", "ff",
                         "wns", "fmax_mhz", "throughput_fps", "build_success"]
            if c in df.columns]
    if not cols:
        log.info("No recognised metric columns found in results.")
        return
    sort_col = "objective" if "objective" in df.columns else cols[0]
    print("\n═══════════════ Trial Summary ═══════════════")
    print(df[cols].sort_values(sort_col).to_string(index=True))
    print("═════════════════════════════════════════════\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Tune FINN folding config (PE/SIMD) with Ray Tune.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example\n"
            "───────\n"
            "  python finn_raytune_optimizer.py \\\n"
            "      --baseline_cfg  dataset/config_files/model/folding_config_baseline.json \\\n"
            "      --build_script  full_build.py \\\n"
            "      --onnx_path     dataset/resnet50/model.onnx \\\n"
            "      --best_cfg_out  dataset/config_files/model/folding_config_best.json \\\n"
            "      --num_samples   40 --objective lut_slack\n"
        ),
    )
    p.add_argument("--baseline_cfg",  required=True,
                   help="Baseline folding config JSON (from generate_folding_config.py)")
    p.add_argument("--build_script",  required=True,
                   help="Path to full_build.py")
    p.add_argument("--onnx_path",     required=True,
                   help="Path to the exported ONNX, e.g. dataset/resnet50/model.onnx")
    p.add_argument("--best_cfg_out",  default=None,
                   help="Output path for the best config. "
                        "Defaults to dataset/config_files/<model_stem>/folding_config_best.json")
    p.add_argument("--raytune_dir",   default="./raytune_logs",
                   help="Directory for Ray Tune logs (separate from FINN outputs)")
    p.add_argument("--num_samples",   type=int,   default=100)
    p.add_argument("--max_concurrent",type=int,   default=1,
                   help="Concurrent FINN builds (1 unless you have multiple Vivado licences)")
    p.add_argument("--objective",
                   choices=["lut", "lut_slack", "throughput", "balanced"],
                   default="lut_slack")
    p.add_argument("--build_timeout", type=int,   default=7200,
                   help="Per-trial timeout in seconds (default: 2 h)")
    p.add_argument("--search_strategy",
                   choices=["optuna", "random", "nevergrad", "nevergrad_de", "hyperopt"], default="optuna",
                   help="Search algorithm: optuna (Bayesian), random, or nevergrad")
    p.add_argument("--seed", type=int, default=357569,
                   help="Random seed for reproducibility (default: 357569)")
    p.add_argument("--extra_build_args", nargs=argparse.REMAINDER, default=[],
                   help="Extra arguments forwarded verbatim to full_build.py")
    p.add_argument("--ray_address",   default=None,
                   help="Ray cluster address (omit for local mode)")
    p.add_argument(
        "--parallelism_threshold", type=int, default=1000000000000,
        help="Skip builds where total PE+SIMD exceeds this value (default: 1000000000000)",
    )
    p.add_argument(
        "--database_dir", default=None,
        help="Path to database folder of pre-built configs to reuse "
             "(e.g. database/resnet50). Skipped if not provided.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    
    # Convert relative path to absolute path to prevent pyarrow URI errors
    args.raytune_dir = os.path.abspath(args.raytune_dir)

    # ── Validate inputs ────────────────────────────────────────────────────
    for label, path in [("baseline_cfg", args.baseline_cfg),
                         ("build_script",  args.build_script),
                         ("onnx_path",     args.onnx_path)]:
        if not os.path.exists(path):
            log.error("%s not found: %s", label, path)
            sys.exit(1)

    onnx_abs     = os.path.abspath(args.onnx_path)
    model_stem   = Path(onnx_abs).stem
    dataset_root = str(Path(onnx_abs).parent.parent.parent)
    log.info("dataset_root (full_build.py working dir): %s", dataset_root)

    # Default best config output: dataset/config_files/<model>/folding_config_best_<alg>.json
    model_config_root = os.path.join(dataset_root, "dataset", "config_files", model_stem)
    best_cfg_out = args.best_cfg_out or os.path.join(
        model_config_root, f"folding_config_best_{args.search_strategy}.json"
    )

    # Ray Tune experiment name encodes algorithm + objective for clear separation
    # of logs when running multiple strategies:
    #   raytune_logs/finn_tune_optuna_lut_slack/
    #   raytune_logs/finn_tune_random_lut_slack/
    experiment_name = f"finn_tune_{args.search_strategy}_{args.objective}"

    # ── Load baseline + build search space ────────────────────────────────
    baseline = load_folding_config(args.baseline_cfg)
    log.info("Loaded baseline: %d entries from %s", len(baseline), args.baseline_cfg)

    search_space = build_search_space(baseline)
    if not search_space:
        log.error(
            "Search space is empty. Check that the baseline config contains "
            "PE/SIMD entries with valid MH/MW values."
        )
        sys.exit(1)
    log.info("Search space: %d hyperparameters across %d logical layers.",
             len(search_space), len(search_space) // 2 + len(search_space) % 2)

    # Print the search space so the user can verify it before long runs
    for hp_name, hp_dist in sorted(search_space.items()):
        log.info("  %-50s %s", hp_name, hp_dist)

    # ── Ray initialisation ─────────────────────────────────────────────────
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    # ── Search algorithm ───────────────────────────────────────────────────
    # In Ray 2.x, metric/mode are set on tune.run(), not on the search alg.
    # Both strategies receive the same seed so runs are reproducible and
    # directly comparable against each other.
    if args.search_strategy == "optuna":
        import optuna
        # Persist the Optuna study to a database file inside the algo folder
        # so that restarting the script resumes from where it left off —
        # Optuna will remember all previous trials and keep improving.
        algo_config_root = os.path.join(
            dataset_root, "dataset", "config_files",
            Path(onnx_abs).stem, args.search_strategy
        )
        os.makedirs(algo_config_root, exist_ok=True)
        study_db   = f"sqlite:///{algo_config_root}/optuna_study.db"
        study_name = f"finn_{Path(onnx_abs).stem}"
        study = optuna.create_study(
            study_name   = study_name,
            storage      = study_db,
            direction    = "minimize",
            load_if_exists = True,   # resumes existing study if found
        )
        n_existing = len(study.trials)
        if n_existing > 0:
            log.info(
                "Resuming Optuna study '%s' — %d previous trials loaded from %s",
                study_name, n_existing, study_db,
            )
        else:
            log.info("New Optuna study '%s' created at %s", study_name, study_db)

        storage = optuna.storages.RDBStorage(url=study_db)
        search_alg = OptunaSearch(
            storage    = storage,
            study_name = study_name,
            metric     = "objective",
            mode       = "min",
        )
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=args.max_concurrent)
        log.info("Search: Optuna (Bayesian) — seed=%d", args.seed)
        
    elif args.search_strategy == "nevergrad":
        # OnePlusOne is a good default for discrete/categorical spaces
        # Other options: ng.optimizers.DE, ng.optimizers.PSO, ng.optimizers.CMA
        import nevergrad
        from ray.tune.search.nevergrad import NevergradSearch
        search_alg = NevergradSearch(
            optimizer  = nevergrad.optimizers.OnePlusOne,
            metric     = "objective",
            mode       = "min",
        )
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=args.max_concurrent)
        log.info("Search: Nevergrad (OnePlusOne) — install: pip install nevergrad")
        
    elif args.search_strategy == "nevergrad_de":
        import nevergrad
        from ray.tune.search.nevergrad import NevergradSearch
        search_alg = NevergradSearch(
            optimizer = nevergrad.optimizers.TwoPointsDE,
            metric    = "objective",
            mode      = "min",
        )
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=args.max_concurrent)
        log.info("Search: Nevergrad (TwoPointsDE)")
        
    elif args.search_strategy == "hyperopt":
        from ray.tune.search.hyperopt import HyperOptSearch
        search_alg = HyperOptSearch(
            metric = "objective",
            mode   = "min",
            random_state_seed = args.seed,
        )
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=args.max_concurrent)
        log.info("Search: HyperOpt (TPE) — install: pip install hyperopt")

    else:  # random
        from ray.tune.search.basic_variant import BasicVariantGenerator
        search_alg = BasicVariantGenerator(
            random_state   = args.seed,
            max_concurrent = args.max_concurrent,
        )
        log.info("Search: Random — seed=%d", args.seed)
    scheduler = None

    reporter = CLIReporter(
        metric_columns=["objective", "lut", "dsp", "wns",
                        "fmax_mhz", "throughput_fps", "build_success"],
        max_report_frequency=60,
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
    )

    log.info("Starting: %d samples, max %d concurrent, objective='%s'",
             args.num_samples, args.max_concurrent, args.objective)
    log.info("Parallelism threshold : %d (builds above this are skipped)",
             args.parallelism_threshold)
    if args.database_dir:
        log.info("Database dir          : %s", args.database_dir)
    else:
        log.info("Database dir          : not set (all configs will be built)")

    analysis = tune.run(
        trainable,
        config                = search_space,
        metric                = "objective",
        mode                  = "min",
        num_samples           = args.num_samples,
        # max_concurrent_trials = args.max_concurrent,
        search_alg            = search_alg,
        scheduler             = scheduler,
        progress_reporter     = reporter,
        storage_path          = args.raytune_dir,
        name                  = experiment_name,
        verbose               = 1,
        raise_on_failed_trial = False,
        resources_per_trial   = {"cpu": 2, "gpu": 0},
    )

    print_trial_summary(analysis)
    save_best_config(analysis, baseline, best_cfg_out)
    ray.shutdown()
    log.info("Done.")


if __name__ == "__main__":
    main()