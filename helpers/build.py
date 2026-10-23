import json
import logging
import os
import sys
import subprocess
from subprocess import PIPE
from typing import Any, Dict, Mapping, Optional, Tuple

from helpers.metrics import collect_metrics
from helpers.objectives import compute_objective
from helpers.trials import _append_to_csv
from helpers.folding import patch_folding_config, save_folding_config

from space_pruner import run_pruner_multi


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PENALTY_HARD = 10000

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

def call_pruner(models: Dict[str, Dict[str, Any]], constraints: str,
                 pruning_round: int, api_key: Optional[str], include_results: bool) -> None:
    payload = {
        key: dict(
            build_cfg_path    = m["build_cfg_path"],
            baseline_cfg_path = m["baseline_cfg_path"],
            out_json_path     = m["pruned_space_path"],
            results_csv       = m["results_csv"] if include_results else None,
        )
        for key, m in models.items()
    }
    run_pruner_multi(models=payload, constraints=constraints,
                      pruning_round=pruning_round, api_key=api_key)

# ─────────────────────────────────────────────────────────────────────────────
# Feasibility check (all PE/SIMD = 1)
# ─────────────────────────────────────────────────────────────────────────────

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
    onnx_stem: str,
    model_name: str,
    onnx_dir: str,
    build_script: str,
    build_timeout: int,
    extra_build_args: list,
    objective: str,
    results_csv: str,
    device: Mapping[str, float],
) -> Tuple[Dict[str, Any], Dict[str, float], float, bool]:
    """Runs a minimal hardware build. Aborts the entire script if it fails."""
    log.info("═══ Running Minimum Feasibility Check (All PE/SIMD = 1) ═══")
    trial_name = f"{model_stem}_feasibility_check"
    trial_config_dir = os.path.join(algo_config_root, trial_name)
    os.makedirs(trial_config_dir, exist_ok=True)

    trial_params = create_all_ones_params(baseline_cfg)
    patched_cfg = patch_folding_config(baseline_cfg, trial_params)

    cfg_filename = f"{trial_name}.json"
    cfg_finn_path = os.path.join(dataset_root, "dataset", "config_files", onnx_stem, cfg_filename)
    save_folding_config(patched_cfg, cfg_finn_path)

    with open(os.path.join(trial_config_dir, "trial_params.json"), "w") as f:
        json.dump(trial_params, f, indent=2)

    finn_output_dir = os.path.join(dataset_root, "dataset", "results_synth", onnx_stem, trial_name)

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
            obj_value = compute_objective(metrics, objective, device)
            # Check if it failed due to over-utilization
            if obj_value >= PENALTY_HARD:
                success = False

    # Write this result to the CSV so the LLM and Optuna can see it!
    _append_to_csv(results_csv, trial_name, trial_params, metrics, obj_value, success, device=device)

    if not success:
        log.error("🚨 FEASIBILITY CHECK FAILED!")
        log.error("The model is too large for this board even at PE=1, SIMD=1.")
        log.error("Halting execution. Please check the model architecture or select a larger FPGA.")
        sys.exit(1)
    else:
        log.info("✅ Feasibility check passed! The 'all-ones' build is recorded in the CSV.")
        log.info("   LUT: %s | BRAM: %s | DSP: %s", 
                 metrics.get('lut', 0), metrics.get('bram', 0), metrics.get('dsp', 0))
        
    return trial_params, metrics, obj_value, success 