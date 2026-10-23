import json
import os
import re
from copy import deepcopy
from typing import Any, Dict

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

def save_folding_config(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_sanitize_config(cfg), f, indent=2)