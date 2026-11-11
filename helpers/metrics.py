import csv
import json
import logging
import os
import sys

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

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
                         ("FF", "ff"), ("fmax_mhz", "fmax_mhz")]:
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
                if layer_name != "total" and isinstance(counts, dict):
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

_RESOURCE_COLS = {"lut": "LUT", "dsp": "DSP", "bram": "BRAM", "ff": "FF"}


def get_warm_start_points(csv_path: str, new_space_json: str, max_points: int = 5,
                          return_resources: bool = False) -> List[Any]:
    """
    Reads past trials, filters for successful builds, verifies they fit the new space,
    and returns ONLY the top N performing configurations to warm-start Optuna.
    With return_resources=True, returns (params, {LUT, DSP, BRAM, FF}) tuples.
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
                
                # 2. Check if it's still legal (a malformed cell skips only this row)
                for param_name, valid_choices in new_space.items():
                    try:
                        past_val = int(row[param_name])
                    except (KeyError, TypeError, ValueError):
                        is_legal = False
                        break
                    if past_val in valid_choices:
                        trial_params[param_name] = past_val
                    else:
                        is_legal = False
                        break
                
                # 3. Save both the params and the objective score
                if is_legal:
                    # Look for objective, default to a massive penalty if missing
                    try:
                        obj_val = float(row.get('objective') or 1e9)
                    except ValueError:
                        obj_val = 1e9
                    resources = {}
                    for col, dev_key in _RESOURCE_COLS.items():
                        try:
                            resources[dev_key] = float(row.get(col) or 0)
                        except ValueError:
                            resources[dev_key] = 0.0
                    valid_trials.append({'params': trial_params, 'objective': obj_val,
                                         'resources': resources})

    except Exception as e:
        log.warning("Could not load warm start points: %s", e)

    # Sort the trials by best objective (lowest is better)
    valid_trials.sort(key=lambda x: x['objective'])
    top = valid_trials[:max_points]

    if return_resources:
        return [(t['params'], t['resources']) for t in top]
    # Extract just the parameter dictionaries for the top N trials
    return [t['params'] for t in top]

def get_joint_warm_start_points(
    models: Dict[str, Dict[str, Any]],
    max_points: int = 5,
    device: Optional[Mapping[str, float]] = None,
    instance_counts: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Build warm-start points for the JOINT space by combining each model's own
    best-so-far legal points, paired by rank (best-with-best, 2nd-best-with-
    2nd-best, etc.). Optuna requires every warm-start point to specify EVERY
    dimension of the combined search space, so a combo is only emitted when
    ALL models have a point at that rank — models with fewer warm-start
    points simply limit how many joint combos can be formed, rather than
    producing partial (invalid) combos.

    If `device` is given, combos whose SUMMED resources (each model weighted
    by its instance count) exceed the board are dropped, since each model
    fitting on its own does not mean they fit together.
    """
    instance_counts = instance_counts or {}
    per_model_points: Dict[str, List[Any]] = {}
    for ms, m in models.items():
        pts = get_warm_start_points(m["results_csv"], m["pruned_space_path"],
                                    max_points=max_points, return_resources=True)
        per_model_points[ms] = pts

    if not per_model_points or any(len(pts) == 0 for pts in per_model_points.values()):
        log.info("Skipping joint warm-start: at least one model has no valid "
                 "warm-start points yet (%s).",
                 {ms: len(pts) for ms, pts in per_model_points.items()})
        return []

    # Only as many joint combos as the SHORTEST per-model list allows —
    # every combo must have a contribution from every model.
    n_combos = min(len(pts) for pts in per_model_points.values())

    joint_points = []
    for i in range(n_combos):
        combo = {}
        totals = {dev_key: 0.0 for dev_key in _RESOURCE_COLS.values()}
        for ms, pts in per_model_points.items():
            params, resources = pts[i]
            for k, v in params.items():
                combo[f"{ms}::{k}"] = v
            for dev_key, val in resources.items():
                totals[dev_key] += val * instance_counts.get(ms, 1)

        if device is not None:
            over = [k for k, v in totals.items() if device.get(k) and v > device[k]]
            if over:
                log.info("Dropping joint warm-start combo #%d: combined %s exceed the board.",
                         i, over)
                continue
        joint_points.append(combo)

    return joint_points