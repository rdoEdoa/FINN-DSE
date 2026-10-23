import json
import logging
import os
import shutil
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping

from ray import tune

from helpers.metrics import collect_metrics
from helpers.objectives import (
    PENALTY,
    PENALTY_HARD,
    compute_objective,
    compute_joint_objective,
)
from helpers.trials import (
    _append_to_csv,
    _increment_counter,
    _next_trial_number,
)

from helpers.folding import patch_folding_config, save_folding_config
from helpers.build import run_finn_build

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

def make_joint_trainable(
    models: Dict[str, Dict[str, Any]],
    all_models: Dict[str, Dict[str, Any]],
    duplicate_siblings: Dict[str, List[str]],
    instance_counts: Dict[str, int],
    objective: str,
    search_strategy: str,
    build_timeout: int,
    extra_build_args: list,
    joint_root: str,
    device: Mapping[str, float],
    single_network: bool = False,
):
    def trainable(trial_config: Dict[str, Any]):
        import os, json, shutil, logging
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)

        # ── Split merged config back into per-model param dicts ────────────
        per_model_params: Dict[str, Dict[str, Any]] = {ms: {} for ms in models}
        for key, value in trial_config.items():
            ms, param = key.split("::", 1)
            per_model_params[ms][param] = value

        trial_num  = _next_trial_number(joint_root)
        trial_name = f"config_{trial_num:04d}"

        def build_one(ms: str):
            m = models[ms]
            patched_cfg = patch_folding_config(m["baseline"], per_model_params[ms])

            cfg_filename  = f"{search_strategy}_{ms}_{trial_name}.json"
            cfg_finn_path = os.path.join(
                m["dataset_root"], "dataset", "config_files", m["onnx_stem"], cfg_filename
            )
            save_folding_config(patched_cfg, cfg_finn_path)

            trial_config_dir = (os.path.join(joint_root, trial_name) if single_network
                                else os.path.join(joint_root, trial_name, ms))
            os.makedirs(trial_config_dir, exist_ok=True)
            shutil.copy(cfg_finn_path, os.path.join(trial_config_dir, "folding_config.json"))
            with open(os.path.join(trial_config_dir, "trial_params.json"), "w") as f:
                json.dump(per_model_params[ms], f, indent=2)

            cfg_stem        = os.path.splitext(cfg_filename)[0]
            finn_output_dir = os.path.join(
                m["dataset_root"], "dataset", "results_synth", m["onnx_stem"], cfg_stem
            )

            _increment_counter(m["algo_config_root"], "built")
            success, build_log = run_finn_build(
                build_script        = m["build_script"],
                onnx_dir            = str(Path(m["onnx_abs"]).parent),
                model_name          = Path(m["onnx_abs"]).name,
                folding_config_name = cfg_filename,
                cwd                 = m["dataset_root"],
                timeout             = build_timeout,
                extra_args          = extra_build_args,
            )
            with open(os.path.join(trial_config_dir, "build.log"), "w") as f:
                f.write(build_log)

            metrics = {}
            if success:
                metrics = collect_metrics(finn_output_dir)
                if metrics.get("finn_success", 1) == 0:
                    success = False
            if success:
                _increment_counter(m["algo_config_root"], "built_success")

            return ms, success, metrics

        results: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = [executor.submit(build_one, ms) for ms in models]
            for fut in as_completed(futures):
                ms, success, metrics = fut.result()
                results[ms] = {"success": success, "metrics": metrics}

        # ── Per-model CSV logging (so space_pruner's per-model view still works) ──
        for ms, r in results.items():
            m_obj = (
                compute_objective(r["metrics"], objective, device)
                if r["success"]
                else PENALTY
            )
            _append_to_csv(models[ms]["results_csv"], trial_name, per_model_params[ms],
                           r["metrics"], m_obj, r["success"], device=device)
            for sib_key in duplicate_siblings.get(ms, []):       
                _append_to_csv(all_models[sib_key]["results_csv"], trial_name,
                               per_model_params[ms], r["metrics"], m_obj, r["success"])

        # ── Joint objective across all models, summing resources ──────────
        obj_value, breakdown = compute_joint_objective(
            results,
            objective,
            instance_counts,
            device=device,
        )
        overall_success = (obj_value < PENALTY_HARD) and all(r["success"] for r in results.values())

        report_dict: Dict[str, Any] = {
            "objective":     obj_value,
            "build_success": int(overall_success),
            "trial_name":    trial_name,
        }
        for ms, r in results.items():
            report_dict[f"{ms}__success"] = int(r["success"])
            for k, v in r["metrics"].items():
                if k != "finn_success":
                    report_dict[f"{ms}__{k}"] = v
        if breakdown:
            report_dict.update({
                "combined_lut_pct":  round(sum(b["lut_contrib"]  for b in breakdown.values()) * 100, 2),
                "combined_dsp_pct":  round(sum(b["dsp_contrib"]  for b in breakdown.values()) * 100, 2),
                "combined_bram_pct": round(sum(b["bram_contrib"] for b in breakdown.values()) * 100, 2),
                "combined_ff_pct":   round(sum(b["ff_contrib"]   for b in breakdown.values()) * 100, 2),
                "combined_throughput_fps": round(sum(b["throughput_contrib"] for b in breakdown.values()), 3),
            })

        log.info("[%s] objective=%.4g | success=%s", trial_name, obj_value, overall_success)
        tune.report(report_dict)

    return trainable

