"""
finn_raytune_optimizer_ai.py
─────────────────────────────────────────────────────────────────────────────
FINN folding-config tuner with adaptive LLM-based design space pruning.

Extends finn_raytune_optimizer.py with:
  - A running CSV of all trial results (trial_params + metrics).
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
      --board               Zynq-7015 \
      --model               cnv_w1a1 \
      --epoch_schedule      "50,50,50,100,150,250,350" \
      --prune_on_start \
      --objective           throughput \
      --constraints         "You are working with multiple models that have to share the same target board. You MUST MAXIMIZE their throughputs while ensuring that they all fit on the board." \
      --search_strategy     optuna \
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
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

# Import the custom helpers
from helpers.boards import BOARDS, DEFAULT_BOARD, board_arg
from helpers.metrics import collect_metrics, get_joint_warm_start_points
from helpers.objectives import compute_objective
from helpers.trials import _append_to_csv, _dedup_model_keys
from helpers.trainable import make_joint_trainable
from helpers.folding import patch_folding_config, save_folding_config
from helpers.build import run_finn_build, run_feasibility_check, call_pruner

import threading
from ray.tune import register_trainable

_registry_lock = threading.Lock()

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

# Active board resources. Filled in main() from --board.
DEVICE: Dict[str, int] = dict(BOARDS[DEFAULT_BOARD])

PENALTY_HARD = 10000

# ─────────────────────────────────────────────────────────────────────────────
# Folding config helpers  
# ─────────────────────────────────────────────────────────────────────────────

def load_folding_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


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
# Post-run utilities 
# ─────────────────────────────────────────────────────────────────────────────

def print_trial_summary(analysis) -> None:
    try:
        df = analysis.results_df
    except Exception:
        return
    if df is None or df.empty:
        return
    cols = [c for c in ["objective", "lut", "bram", "dsp", "ff",
                        "fmax_mhz", "throughput_fps", "build_success"]
            if c in df.columns]
    if not cols:
        return
    sort_col = "objective" if "objective" in df.columns else cols[0]
    print("\n═══════════════ Round Summary ═══════════════")
    print(df[cols].sort_values(sort_col).to_string(index=True))
    print("═════════════════════════════════════════════\n")


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER 
# ─────────────────────────────────────────────────────────────────────────────

def build_joint_search_space(models: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Merge each model's pruned_space.json into one flat space, prefixed by model key."""
    combined = {}
    for ms, m in models.items():
        if not Path(m["pruned_space_path"]).exists():
            log.error("[%s] pruned_space.json not found at %s.", ms, m["pruned_space_path"])
            sys.exit(1)
        space_ms = load_search_space(m["pruned_space_path"])
        for k, v in space_ms.items():
            combined[f"{ms}::{k}"] = v
        log.info("[%s] %d hyperparameters.", ms, len(space_ms))
    return combined

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FINN folding tuner with adaptive LLM space pruning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, nargs="+",
               help="One or more model names sharing the same board, "
                    "e.g. --model cnv_w2a2 cnv_w4a4")
    p.add_argument("--raytune_dir",    default="./raytune_logs")
    p.add_argument("--epoch_schedule", type=str, default="100,100,100,100,100",
                   help="Comma-separated list of trials per LLM pruning round.")
    p.add_argument("--max_concurrent", type=int, default=1)
    p.add_argument("--objective",
                   choices=["throughput",
                             "resource_avg", "balanced"],
                   default="balanced")
    p.add_argument("--build_timeout",  type=int, default=7200)
    p.add_argument("--board", type=board_arg, default=DEFAULT_BOARD,
                   help=f"Target board (resource budget). One of: {', '.join(BOARDS)}.")
    p.add_argument("--search_strategy",
                   choices=["optuna", "random", "nevergrad",
                             "nevergrad_de", "hyperopt"],
                   default="optuna")
    p.add_argument("--seed",           type=int, default=20)
    p.add_argument("--constraints",    type=str,
                   default="Maximise throughput while minimising resource usage.",
                   help="Natural language constraints passed to space_pruner")
    p.add_argument("--api_key",   default=None,
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
    args.run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    log.info("Run ID: %s", args.run_id)
    args.raytune_dir = os.path.abspath(args.raytune_dir)

    DEVICE.update(BOARDS[args.board])
    log.info("Target board: %s  %s", args.board, DEVICE)

    base_model_names = args.model
    model_keys  = _dedup_model_keys(base_model_names)
    
    models: Dict[str, Dict[str, Any]] = {}
    for name, key in zip(base_model_names, model_keys):
        baseline_cfg_path = os.path.join("dataset", "config_files", name, "folding_config_baseline.json")
        build_cfg_path    = os.path.join("build_configs", f"{name}_build_config.json")
        onnx_path         = os.path.join("dataset", name, f"{name}.onnx")
        build_script      = os.path.join("custom_builds", f"{name}_build.py")

        for label, path in [(f"{key}:baseline_cfg", baseline_cfg_path),
                            (f"{key}:build_cfg", build_cfg_path),
                            (f"{key}:onnx_path", onnx_path),
                            (f"{key}:build_script", build_script)]:
            if not os.path.exists(path):
                log.error("%s not found: %s", label, path)
                sys.exit(1)

        onnx_abs     = os.path.abspath(onnx_path)
        onnx_stem  = Path(onnx_abs).parent.name 
        model_stem   = key
        dataset_root = str(Path(onnx_abs).parent.parent.parent)

        model_config_root = os.path.join(dataset_root, "dataset", "config_files", model_stem)
        algo_config_root  = os.path.join(model_config_root, args.search_strategy)
        os.makedirs(algo_config_root, exist_ok=True)

        models[key] = dict(
            name              = name,
            baseline_cfg_path = baseline_cfg_path,
            build_cfg_path    = build_cfg_path,
            onnx_abs          = onnx_abs,
            onnx_stem         = onnx_stem,
            model_stem        = model_stem,
            dataset_root      = dataset_root,
            model_config_root = model_config_root,
            algo_config_root  = algo_config_root,
            pruned_space_path = os.path.join(algo_config_root, "pruned_space.json"),
            results_csv       = os.path.join(algo_config_root, "trial_results.csv"),
            best_cfg_out      = os.path.join(model_config_root, f"folding_config_best_{args.search_strategy}.json"),
            baseline          = load_folding_config(baseline_cfg_path),
            build_script      = os.path.abspath(build_script),
        )
        
    # ── Group duplicate instances of the same physical model ──────────────
    name_to_keys: Dict[str, List[str]] = {}
    for key, m in models.items():
        name_to_keys.setdefault(m["name"], []).append(key)

    representative_keys: List[str] = [keys[0] for keys in name_to_keys.values()]
    instance_counts: Dict[str, int] = {keys[0]: len(keys) for keys in name_to_keys.values()}
    duplicate_siblings: Dict[str, List[str]] = {keys[0]: keys[1:] for keys in name_to_keys.values()}
    rep_models: Dict[str, Dict[str, Any]] = {k: models[k] for k in representative_keys}

    for name, keys in name_to_keys.items():
        if len(keys) > 1:
            log.info("Model '%s' deployed %d times (%s) — built ONCE as "
                     "representative '%s'; the LLM pruner still sees all %d instances.",
                     name, len(keys), keys, keys[0], len(keys))

    dataset_roots = {m["dataset_root"] for m in models.values()}
    if len(dataset_roots) > 1:
        log.warning("Models resolve to different dataset roots: %s — "
                    "is this intentional?", dataset_roots)
    shared_dataset_root = sorted(dataset_roots)[0]

    combo_id       = "+".join(model_keys)
    single_network = len(model_keys) == 1
    if single_network:
        # One network: use its own folder, dataset/config_files/<model>/<strategy>/
        joint_root = models[model_keys[0]]["algo_config_root"]
    else:
        joint_root = os.path.join(
            shared_dataset_root, "dataset", "config_files", f"JOINT__{combo_id}", args.search_strategy
        )
    os.makedirs(joint_root, exist_ok=True)
    joint_study_db = (f"sqlite:///{joint_root}/optuna_study.db"
                    if args.search_strategy == "optuna" else None)
    
    # ── Fail-Fast Feasibility Check ─────────────────────────────────────
    # Only run if this is a completely new experiment (no CSV exists yet)
    for rep_key in representative_keys:
        m = models[rep_key]
        if not os.path.exists(m["results_csv"]):
            trial_params, metrics, obj_value, success = run_feasibility_check(
                baseline_cfg     = m["baseline"],
                algo_config_root = m["algo_config_root"],
                dataset_root     = m["dataset_root"],
                model_stem       = m["model_stem"],
                onnx_stem        = m["onnx_stem"],
                model_name       = Path(m["onnx_abs"]).name,
                onnx_dir         = str(Path(m["onnx_abs"]).parent),
                build_script     = m["build_script"],
                build_timeout    = args.build_timeout,
                extra_build_args = args.extra_build_args,
                objective        = args.objective,
                results_csv      = m["results_csv"],
                device           = DEVICE,
            )
            for sib_key in duplicate_siblings.get(rep_key, []):
                sib = models[sib_key]
                if not os.path.exists(sib["results_csv"]):
                    trial_name_sib = f"{sib['model_stem']}_feasibility_check"
                    _append_to_csv(sib["results_csv"], trial_name_sib,
                                    trial_params, metrics, obj_value, success, device=DEVICE)
    
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    # ── Determine pruning schedule ─────────────────────────────────────────
    try:
        epoch_schedule = [int(x.strip()) for x in args.epoch_schedule.split(',')]
    except ValueError:
        log.error("Invalid --epoch_schedule. Must be comma-separated integers (e.g., '50,100,250,600').")
        sys.exit(1)

    total_samples = sum(epoch_schedule)
    pruning_round = 0
    trials_done = 0

    # ── Optional initial pruning before any builds ─────────────────────────
    if args.prune_on_start or any(not Path(m["pruned_space_path"]).exists() for m in models.values()):
        log.info("═══ Initial pruning (round 0) ═══")
        call_pruner(models, args.constraints, pruning_round=0,
                    api_key=args.api_key, include_results=False)
        pruning_round += 1
    
    # ── Round-based execution loop ─────────────────────────────────────────
    experiment_name = f"finn_tune_{args.search_strategy}_{args.objective}"

    for round_samples in epoch_schedule:
        log.info("═══ Round %d | JOINT trial across %d models × %d samples ═══",
                pruning_round, len(models), round_samples)

        combined_space = build_joint_search_space(rep_models)
        log.info("Combined search space: %d hyperparameters.", len(combined_space))

        current_study_name = None
        if args.search_strategy == "optuna":
            current_study_name = f"finn_joint_{combo_id}_run{args.run_id}_round{pruning_round}"

        warm_start_points = get_joint_warm_start_points(rep_models)
        if warm_start_points:
            log.info("Warm-starting joint search with %d combined points.", len(warm_start_points))

        search_alg = build_search_alg(
            strategy=args.search_strategy, seed=args.seed + pruning_round,
            max_concurrent=args.max_concurrent, study_db=joint_study_db,
            study_name=current_study_name,
            warm_start_points=warm_start_points or None,
        )

        trainable = make_joint_trainable(
            models=rep_models,              
            all_models=models,              
            duplicate_siblings=duplicate_siblings,
            instance_counts=instance_counts,
            objective=args.objective, search_strategy=args.search_strategy,
            build_timeout=args.build_timeout, extra_build_args=args.extra_build_args,
            joint_root=joint_root,
            single_network=single_network,
            device=DEVICE,
        )

        round_name = f"{experiment_name}_JOINT_round{pruning_round:02d}"
        trainable_name = f"trainable_{round_name}"
        with _registry_lock:
            register_trainable(trainable_name, trainable)

        reporter = CLIReporter(
            metric_columns=["objective", "build_success", "combined_throughput_fps",
                            "combined_lut_pct", "combined_bram_pct"],
            max_report_frequency=60,
        )

        analysis = tune.run(
            trainable_name, config=combined_space, metric="objective", mode="min",
            num_samples=round_samples, search_alg=search_alg,
            progress_reporter=reporter, storage_path=args.raytune_dir,
            name=round_name, verbose=1, raise_on_failed_trial=False,
            resources_per_trial={"cpu": 2 * len(rep_models), "gpu": 0},
        )

        print_trial_summary(analysis)
        joint_last_analysis = analysis   # single shared analysis, not per-model

        trials_done   += round_samples
        pruning_round += 1

        if trials_done < total_samples:
            log.info("═══ Calling pruner (round %d, %d trials done) ═══", pruning_round, trials_done)
            call_pruner(models, args.constraints, pruning_round, args.api_key, include_results=True)

            for key, m in models.items():
                rows = []
                if os.path.exists(m["results_csv"]):
                    with open(m["results_csv"]) as f:
                        rows = list(csv.DictReader(f))[-round_samples:]
                successes = sum(1 for r in rows if r.get("build_success") == "1")
                if rows and successes <= 1:
                    log.warning("[%s] 🚨 GUARDRAIL: clamping to max value 8.", key)
                    with open(m["pruned_space_path"]) as f:
                        unsafe_space = json.load(f)
                    safe_space = {pkey: ([x for x in arr if x <= 8] or [min(arr)])
                                for pkey, arr in unsafe_space.items()}
                    with open(m["pruned_space_path"], "w") as f:
                        json.dump(safe_space, f, indent=2)

    # ── Save final best config ─────────────────────────────────────────────
    if 'joint_last_analysis' in dir() and joint_last_analysis:
        try:
            best_params = joint_last_analysis.get_best_config(metric="objective", mode="min")
        except Exception as exc:
            log.warning("Could not retrieve joint best config: %s", exc)
            best_params = None

        if best_params:
            per_model_best: Dict[str, Dict[str, Any]] = {ms: {} for ms in representative_keys}   
            for key, value in best_params.items():
                ms, param = key.split("::", 1)
                per_model_best[ms][param] = value

            for rep_key, sib_keys in duplicate_siblings.items():
                for sib_key in sib_keys:  
                    per_model_best[sib_key] = per_model_best[rep_key] 

            for ms, m in models.items():
                best_cfg = patch_folding_config(m["baseline"], per_model_best[ms])
                save_folding_config(best_cfg, m["best_cfg_out"])
                log.info("[%s] Best joint config saved to: %s", ms, m["best_cfg_out"])
                for k, v in sorted(per_model_best[ms].items()):
                    log.info("  [%s] %-45s = %s", ms, k, v)

    ray.shutdown()
    log.info("Done. Total trials: %d", trials_done)


if __name__ == "__main__":
    main()