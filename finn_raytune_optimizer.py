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
        --baseline_cfg  dataset/config_files/model/folding_config_baseline.json \
        --build_script  full_build.py \
        --onnx_path     dataset/model/model.onnx \
        --best_cfg_out  dataset/config_files/model/folding_config_best.json \
        --num_samples   40 \
        --objective     lut_slack
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
from ray.tune.search.nevergrad import NevergradSearch

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

# Device constants for xc7z020clg400-1, default board
DEVICE_LUT  = 53200
DEVICE_DSP  = 220
DEVICE_BRAM = 140
DEVICE_FF   = 106400

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
# Folding-config helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    Apply Ray Tune trial parameters to a baseline folding config.

    trial_params keys use LOGICAL layer names (no hls/rtl suffix):
        "<layer_type>_<idx>__PE"    e.g. "MVAU_0__PE"
        "<layer_type>_<idx>__SIMD"  e.g. "ConvolutionInputGenerator_0__SIMD"

    The same sampled value is written to BOTH the _hls_ and _rtl_ variants
    of each logical layer so they always stay in sync.
    """
    patched = deepcopy(baseline)

    for key, value in trial_params.items():
        if "__" not in key:
            continue
        logical_name, param = key.rsplit("__", 1)

        # logical_name is e.g. "MVAU_0" — split at the last underscore
        m = re.match(r"^(.+)_(\d+)$", logical_name)
        if not m:
            log.warning("Cannot parse logical layer '%s' — skipping.", logical_name)
            continue
        layer_type = m.group(1)
        layer_idx  = m.group(2)

        applied = 0
        for variant in ("hls", "rtl"):
            node_name = f"{layer_type}_{variant}_{layer_idx}"
            if node_name in patched and param in patched[node_name]:
                # Ray Tune may return floats even for integer choices — cast explicitly
                patched[node_name][param] = int(value)
                applied += 1

        if applied == 0:
            log.warning(
                "No config entry found for logical layer '%s' param '%s' — skipping.",
                logical_name, param,
            )

    return patched


# ─────────────────────────────────────────────────────────────────────────────
# Search-space builder
# ─────────────────────────────────────────────────────────────────────────────

def build_search_space(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build one hyperparameter per logical layer (layer_type + index).

    Deduplicates _hls_ / _rtl_ variants so each physical layer contributes
    exactly one PE and one SIMD hyperparameter. Valid choices are computed
    from MH and MW stored in the baseline config so only legal values are
    ever offered to the search algorithm.
    """
    # Collect one representative config per logical layer
    # (prefer _hls_ if both are present)
    logical: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for node_name, node_cfg in baseline.items():
        if node_name == "Defaults":
            continue
        parsed = _split_config_key(node_name)
        if parsed is None:
            continue
        layer_type, variant, layer_idx = parsed
        lkey = (layer_type, layer_idx)
        if lkey not in logical or variant == "hls":
            logical[lkey] = node_cfg

    space: Dict[str, Any] = {}
    for (layer_type, layer_idx), node_cfg in sorted(logical.items()):
        logical_name = f"{layer_type}_{layer_idx}"
        mh = node_cfg.get("MH", None)
        mw = node_cfg.get("MW", None)

        if "PE" in node_cfg:
            valid_pe = [v for v in VALID_PE_SIMD if mh is None or mh % v == 0]
            if valid_pe:
                space[f"{logical_name}__PE"] = tune.choice(valid_pe)
            else:
                log.warning(
                    "Layer %s: no valid PE found for MH=%s — skipping PE.",
                    logical_name, mh,
                )

        if "SIMD" in node_cfg:
            valid_simd = [v for v in VALID_PE_SIMD if mw is None or mw % v == 0]
            if valid_simd:
                space[f"{logical_name}__SIMD"] = tune.choice(valid_simd)
            else:
                log.warning(
                    "Layer %s: no valid SIMD found for MW=%s — skipping SIMD.",
                    logical_name, mw,
                )

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
    Read the JSON reports produced by full_build.py from finn_output_dir.
    All files are flat in that directory after collect_reports_and_cleanup().

    Returns a dict of float-valued metrics. Missing or corrupt files are
    skipped silently — callers check for missing keys.
    """
    metrics: Dict[str, float] = {}

    # status.json
    status = _read_json(finn_output_dir, REPORT_FILES["status"])
    if status:
        metrics["finn_success"] = float(status.get("success", 0))

    # ooc_synth_and_timing.json
    synth = _read_json(finn_output_dir, REPORT_FILES["synth"])
    if synth:
        for src, dst in [("LUT", "lut"), ("BRAM", "bram"), ("DSP", "dsp"),
                         ("FF", "ff"), ("WNS", "wns"), ("fmax_mhz", "fmax_mhz")]:
            val = synth.get(src)
            if val is not None:
                try:
                    metrics[dst] = float(val)
                except (TypeError, ValueError):
                    pass

    # rtlsim_performance.json
    perf = _read_json(finn_output_dir, REPORT_FILES["perf"])
    if perf:
        val = perf.get("throughput[images/s]")
        if val is not None:
            try:
                metrics["throughput_fps"] = float(val)
            except (TypeError, ValueError):
                pass

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
        fail_penalty = PENALTY_HARD
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
    
    if objective == "throughput":
        if tp <= 0:
            return PENALTY_HARD
        return -tp + fail_penalty
    
    elif objective == "resource_avg":
        avg =  (lut_u + dsp_u + bram_u + ff_u) / 4 
        return avg + util_penalty + fail_penalty

    elif objective == "balanced":
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
    baseline_cfg:     Dict[str, Any],
    build_script:     str,
    onnx_path:        str,
    dataset_root:     str,
    objective:        str,
    search_strategy:  str,
    build_timeout:    int,
    extra_build_args: list,
):
    """
    Return a Ray Tune trainable closure.

    Directory layout
    ────────────────
    dataset/config_files/<model>/<search_strategy>_config_0000/
            folding_config.json   ← config used for this trial
            trial_params.json     ← raw hyperparameter values
            build.log             ← full stdout from full_build.py
            config_0000.json          ← copy at folder level for full_build.py
        ...
        folding_config_best_<strategy>.json   ← written after all trials

    dataset/results_synth/<model>/<search_strategy>_config_0000/
        ...                       ← FINN synthesis outputs
    """
    model_stem = Path(onnx_path).stem        # "model"
    onnx_dir   = str(Path(onnx_path).parent) # ".../dataset/model"
    model_name = Path(onnx_path).name        # "model.onnx"

    # All configs and results for this algorithm live under their own subfolder
    algo_config_root  = os.path.join(
        dataset_root, "dataset", "config_files", model_stem, search_strategy
    )
    algo_results_root = os.path.join(
        dataset_root, "dataset", "results_synth", model_stem, search_strategy
    )

    def trainable(trial_config: Dict[str, Any]):
        import os, json, shutil, logging
        log = logging.getLogger(__name__)

        # ── 1. Patch config ────────────────────────────────────────────────
        patched_cfg = patch_folding_config(baseline_cfg, trial_config)

        # ── 2. Assign a sequential trial number ───────────────────────────
        trial_num  = _next_trial_number(algo_config_root)
        trial_name = f"config_{trial_num:04d}"

        # Per-trial folder:  dataset/config_files/<model>/<algo>/config_xxxx/
        trial_config_dir = os.path.join(algo_config_root, trial_name)
        os.makedirs(trial_config_dir, exist_ok=True)

        # full_build.py resolves the folding config as:
        #   dataset/config_files/<model_stem>/<cfg_name>
        # So the JSON must sit directly under the model config root (one level
        # above our algo subfolder). We write it there, then copy it into the
        # trial folder for traceability.
        cfg_filename  = f"{search_strategy}_{trial_name}.json"
        cfg_finn_path = os.path.join(
            dataset_root, "dataset", "config_files", model_stem, cfg_filename
        )
        save_folding_config(patched_cfg, cfg_finn_path)
        shutil.copy(cfg_finn_path, os.path.join(trial_config_dir, "folding_config.json"))

        # Save trial hyperparameters inside the trial folder
        with open(os.path.join(trial_config_dir, "trial_params.json"), "w") as f:
            json.dump(trial_config, f, indent=2)

        # FINN output dir is determined by full_build.py as:
        #   dataset/results_synth/<model_stem>/<config_file_stem>/
        # The config filename is e.g. "optuna_config_0000.json"
        # so the stem is "optuna_config_0000"
        cfg_stem        = os.path.splitext(cfg_filename)[0]
        finn_output_dir = os.path.join(
            dataset_root, "dataset", "results_synth", model_stem, cfg_stem
        )

        # ── 3. Run build ───────────────────────────────────────────────────
        log.info("Trial %s/%s starting", search_strategy, trial_name)
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

        # ── 4. Parse reports ───────────────────────────────────────────────
        if not success:
            obj_value = PENALTY
            metrics   = {}
        else:
            metrics = collect_metrics(finn_output_dir)
            if metrics.get("finn_success", 1) == 0:
                log.warning(
                    "status.json reports failure despite zero exit code (%s).",
                    trial_name,
                )
                obj_value = PENALTY
            else:
                obj_value = compute_objective(metrics, objective)

        log.info("Trial %s finished — objective: %.4g", trial_name, obj_value)

        # ── 5. Report ──────────────────────────────────────────────────────
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
            "      --onnx_path     dataset/model/model.onnx \\\n"
            "      --best_cfg_out  dataset/config_files/model/folding_config_best.json \\\n"
            "      --num_samples   40 --objective lut_slack\n"
        ),
    )
    p.add_argument("--baseline_cfg",  required=True,
                   help="Baseline folding config JSON (from generate_folding_config.py)")
    p.add_argument("--build_script",  required=True,
                   help="Path to full_build.py")
    p.add_argument("--onnx_path",     required=True,
                   help="Path to the exported ONNX, e.g. dataset/model/model.onnx")
    p.add_argument("--best_cfg_out",  default=None,
                   help="Output path for the best config. "
                        "Defaults to dataset/config_files/<model_stem>/folding_config_best.json")
    p.add_argument("--raytune_dir",   default="./raytune_logs",
                   help="Directory for Ray Tune logs (separate from FINN outputs)")
    p.add_argument("--max_concurrent",type=int,   default=1,
                   help="Concurrent FINN builds (1 unless you have multiple Vivado licences)")
    p.add_argument("--num_samples",   type=int,   default=20)
    p.add_argument("--objective",
                   choices=["lut", "lut_slack", "throughput", "resource_avg", "balanced"],
                   default="lut_slack")
    p.add_argument("--build_timeout", type=int,   default=7200,
                   help="Per-trial timeout in seconds (default: 2 h)")
    p.add_argument("--search_strategy",
                   choices=["optuna", "nevergrad", "random"], default="optuna",
                   help="Search algorithm: optuna (Bayesian) or random")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--extra_build_args", nargs=argparse.REMAINDER, default=[],
                   help="Extra arguments forwarded verbatim to full_build.py")
    p.add_argument("--ray_address",   default=None,
                   help="Ray cluster address (omit for local mode)")
    return p.parse_args()


def main():
    args = parse_args()

    # Normalize the raytune_dir to an absolute path right away
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
        log.info("Search: Optuna (Bayesian) — seed=%d", args.seed)
        
    elif args.search_strategy == "nevergrad":
        import nevergrad
        # OnePlusOne is a good default for discrete/categorical spaces
        # Other options: ng.optimizers.DE, ng.optimizers.PSO, ng.optimizers.CMA
        search_alg = NevergradSearch(
            optimizer  = nevergrad.optimizers.OnePlusOne,
            metric     = "objective",
            mode       = "min",
        )
        log.info("Search: Nevergrad (OnePlusOne) — install: pip install nevergrad")

    else:  # random
        from ray.tune.search.basic_variant import BasicVariantGenerator
        search_alg = BasicVariantGenerator(
            random_state   = args.seed,
            max_concurrent = args.max_concurrent,
        )
        log.info("Search: Random — seed=%d", args.seed)
    scheduler = None

    # ConcurrencyLimiter only works with Searcher subclasses (e.g. OptunaSearch).
    # BasicVariantGenerator handles concurrency via its own max_concurrent param.
    if args.search_strategy == "optuna":
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=args.max_concurrent)
    log.info("Concurrency limit: %d simultaneous build(s)", args.max_concurrent)

    reporter = CLIReporter(
        metric_columns=["objective", "lut", "dsp", "wns",
                        "fmax_mhz", "throughput_fps", "build_success"],
        max_report_frequency=60,
    )

    trainable = make_trainable(
        baseline_cfg     = baseline,
        build_script     = os.path.abspath(args.build_script),
        onnx_path        = onnx_abs,
        dataset_root     = dataset_root,
        objective        = args.objective,
        search_strategy  = args.search_strategy,
        build_timeout    = args.build_timeout,
        extra_build_args = args.extra_build_args,
    )

    log.info("Starting: %d samples, objective='%s'",
             args.num_samples, args.objective)

    analysis = tune.run(
        trainable,
        config                = search_space,
        metric                = "objective",
        mode                  = "min",
        max_concurrent_trials = args.max_concurrent,
        num_samples           = args.num_samples,
        search_alg            = search_alg,
        scheduler             = scheduler,
        progress_reporter     = reporter,
        storage_path          = args.raytune_dir,
        name                  = experiment_name,
        verbose               = 1,
        raise_on_failed_trial = False,
        resources_per_trial   = {"cpu": 4, "gpu": 0},
    )

    print_trial_summary(analysis)
    save_best_config(analysis, baseline, best_cfg_out)
    ray.shutdown()
    log.info("Done.")


if __name__ == "__main__":
    main()