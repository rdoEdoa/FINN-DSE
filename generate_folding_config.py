"""
generate_folding_config.py
─────────────────────────────────────────────────────────────────────────────
Given a Brevitas/PyTorch model file and a JSON config, this script:

  1. Instantiates the model and exports it to ONNX via export_qonnx.
  2. Reads the ONNX graph directly to find Conv/MatMul/Gemm nodes.
  3. Extracts MH and MW from the weight tensors (same approach as your
     config-generation script), then builds a folding config with PE=1
     and SIMD=1 everywhere as the baseline for Ray Tune.

No FINN transformations are run here. The ONNX is the raw Brevitas export
(QONNX format), so node names will differ from the HLS node names FINN uses
after step_convert_to_hw. The config is therefore structured as:

    { "Defaults": { ... },
      "node_<i>": { "PE": 1, "SIMD": 1, "MH": ..., "MW": ..., "op_type": ... } }

When full_build.py runs, FINN renames nodes to MVAU_hls_0, MVAU_rtl_0, etc.
The config must use those names. So we follow the same naming convention
as your existing config-generation script:

    MVAU_hls_<i>, MVAU_rtl_<i>, ConvolutionInputGenerator_hls_<i>,
    ConvolutionInputGenerator_rtl_<i>

This matches exactly what full_build.py expects.

Usage
─────
    python generate_folding_config.py \
        --model_file  lenet5_quantized.py \
        --model_class LeNet5Quantized     \
        --config      build_config.json
"""

import argparse
import importlib.util
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import onnx
from onnx import numpy_helper

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config defaults
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_DEFAULTS: Dict[str, Any] = {
    "output_onnx"  : "dataset/lenet5/model.onnx",
    "output_cfg"   : "dataset/config_files/model/folding_config_baseline.json",
    "opset"        : 11,
    "input_shape"  : [1, 1, 32, 32],
    "board"        : "Pynq-Z2",
    "fpga_part"    : "xc7z020clg400-1",
    "clock_ns"     : 10.0,
    "target_fps"   : None,
    "model_kwargs" : {},
}

TEMPLATE_CONFIG: Dict[str, Any] = {
    "_comment"    : "Build configuration for generate_folding_config.py",
    "output_onnx" : "dataset/lenet5/model.onnx",
    "output_cfg"  : "dataset/config_files/model/folding_config_baseline.json",
    "opset"       : 11,
    "input_shape" : [1, 1, 32, 32],
    "board"       : "Pynq-Z2",
    "fpga_part"   : "xc7z020clg400-1",
    "clock_ns"    : 10.0,
    "target_fps"  : None,
    "model_kwargs": {
        "num_classes"      : 10,
        "weight_bit_width" : 4,
        "act_bit_width"    : 4,
        "inp_bit_width"    : 8,
        "in_channels"      : 1,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Config file helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_build_config(config_path: Optional[str]) -> Dict[str, Any]:
    cfg = dict(CONFIG_DEFAULTS)
    if config_path is None:
        log.info("No --config provided — using defaults.")
        with open("build_config.template.json", "w") as f:
            json.dump(TEMPLATE_CONFIG, f, indent=2)
        log.info("Template written to build_config.template.json")
        return cfg
    with open(config_path) as f:
        user_cfg = json.load(f)
    cfg.update({k: v for k, v in user_cfg.items() if k != "_comment"})
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_file(
    model_file: str,
    model_class: Optional[str],
    model_kwargs: Dict[str, Any],
) -> torch.nn.Module:
    model_path = Path(model_file).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    spec = importlib.util.spec_from_file_location("_user_model", model_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if model_class:
        target = getattr(mod, model_class, None)
        if target is None:
            raise AttributeError(f"Class '{model_class}' not found in {model_file}.")
    else:
        for fname in ("get_model", "build_model", "create_model"):
            if hasattr(mod, fname) and callable(getattr(mod, fname)):
                return getattr(mod, fname)(**model_kwargs)
        target = None
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
                target = obj
                break
        if target is None:
            raise RuntimeError(f"No nn.Module subclass found in {model_file}.")

    model = target(**model_kwargs)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def export_to_onnx(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
    opset: int = 11,
) -> str:
    """
    Export model to ONNX for FINN consumption.

    Priority order
    ──────────────
    1. FINNManager  — exports directly to FINN's expected format.
                      Handles residual connections (ResNet, etc.) correctly.
                      Required for any network with AddStreams / skip connections.
    2. export_qonnx — QONNX format. Works for simple feed-forward networks
                      (LeNet-5, etc.) but inserts Transpose nodes that cause
                      shape inference errors in FINN for residual networks.
    3. torch.onnx   — plain float fallback, quantisation annotations lost.
    """
    output_path = str(Path(output_path).resolve())
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dummy = torch.zeros(input_shape)

    # ── Attempt 1: FINNManager (preferred for FINN compatibility) ───────────
    try:
        from brevitas.export import FINNManager
        log.info("Exporting with FINNManager …")
        FINNManager.export(model, input_shape, output_path, opset_version=opset)
        log.info("FINNManager OK: %s", output_path)
        return output_path
    except ImportError:
        pass
    except Exception as exc:
        log.warning("FINNManager failed (%s), trying export_qonnx.", exc)

    # ── Attempt 2: export_qonnx (fallback for simple networks) ─────────────
    try:
        from brevitas.export import export_qonnx
        log.info("Exporting with export_qonnx …")
        export_qonnx(model, args=dummy, export_path=output_path, opset_version=opset)
        log.info("export_qonnx OK: %s", output_path)
        return output_path
    except ImportError:
        pass
    except Exception as exc:
        log.warning("export_qonnx failed (%s), falling back to torch.onnx.", exc)

    # ── Attempt 3: plain torch.onnx (quantisation lost) ────────────────────
    log.warning("Falling back to torch.onnx.export — quantisation annotations lost.")
    torch.onnx.export(
        model, dummy, output_path,
        opset_version=opset,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    log.info("torch.onnx OK: %s", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# ONNX graph parsing  (same logic as your config-generation script)
# ─────────────────────────────────────────────────────────────────────────────

def _find_weights(start_tensor_name, initializer_map, producer_map, max_depth=6):
    """Walk backwards from a tensor name to find the weight initializer."""
    queue   = [(start_tensor_name, 0)]
    visited = set()
    while queue:
        curr_name, depth = queue.pop(0)
        if depth > max_depth or curr_name in visited:
            continue
        visited.add(curr_name)
        if curr_name in initializer_map:
            w = numpy_helper.to_array(initializer_map[curr_name])
            if len(w.shape) == 4 or (len(w.shape) == 2 and w.shape[0] > 1):
                return w
        if curr_name in producer_map:
            for inp in producer_map[curr_name].input:
                queue.append((inp, depth + 1))
    return None


def _get_div(n: int):
    """Power-of-two divisors of n."""
    n = max(1, int(n))
    return [i for i in range(1, n + 1) if n % i == 0 and (i & (i - 1)) == 0]


def _get_mh_mw_ifm(node, initializer_map, producer_map):
    """Return (MH, MW, IFMChannels) for a Conv/MatMul/Gemm node from its weight tensor."""
    mh, mw, ifm = 1, 1, 1
    if len(node.input) > 1:
        w = _find_weights(node.input[1], initializer_map, producer_map)
        if w is not None:
            if len(w.shape) == 4:           # Conv: [OC, IC, KH, KW]
                mh = w.shape[0]
                mw = w.shape[1] * w.shape[2] * w.shape[3]
                ifm = w.shape[1]            # <--- This is the IFMChannels
            elif len(w.shape) == 2:         # MatMul/Gemm: [out, in]
                mh = w.shape[0]
                mw = w.shape[1]
                ifm = w.shape[1]            # For linear layers, IFM == MW
    return max(1, mh), max(1, mw), max(1, ifm)


def extract_folding_config_from_onnx(
    onnx_path: str,
    board: str,
    fpga_part: str,
    clock_ns: float,
    target_fps: Optional[float],
) -> Dict[str, Any]:
    """
    Parse the exported ONNX and build a baseline folding config with PE=1 / SIMD=1.

    Node naming follows the convention used by FINN after step_convert_to_hw
    and by your existing config-generation script:

        MVAU_hls_<i>  / MVAU_rtl_<i>
        ConvolutionInputGenerator_hls_<i> / ConvolutionInputGenerator_rtl_<i>

    Both _hls and _rtl variants are written with the same PE/SIMD so FINN
    can choose whichever implementation it prefers.
    """
    log.info("Parsing ONNX graph: %s", onnx_path)
    model = onnx.load(onnx_path)
    graph = model.graph

    initializer_map = {init.name: init for init in graph.initializer}
    producer_map    = {
        out: node
        for node in graph.node
        for out in node.output
    }

    cfg: Dict[str, Any] = {
        "Defaults": {}
    }

    mva_counter = 0
    swg_counter = 0

    for node in graph.node:
        if node.op_type not in ("Conv", "MatMul", "Gemm"):
            continue

        # Use the updated function to get IFM
        mh, mw, ifm = _get_mh_mw_ifm(node, initializer_map, producer_map)

        log.debug(
            "  %s node — op=%s  MH=%d  MW=%d  IFM=%d",
            node.name or node.op_type, node.op_type, mh, mw, ifm,
        )

        # MVAU (matrix-vector activation unit) — both HLS and RTL variants
        for suffix in ("hls", "rtl"):
            key = f"MVAU_{suffix}_{mva_counter}"
            cfg[key] = {
                "PE":        1,
                "SIMD":      1,
                "MH":        int(mh),
                "MW":        int(mw),
                "ram_style": "auto",
                "op_type":   node.op_type,
            }

        # ConvolutionInputGenerator — only for Conv nodes
        if node.op_type == "Conv":
            for suffix in ("hls", "rtl"):
                key = f"ConvolutionInputGenerator_{suffix}_{swg_counter}"
                cfg[key] = {
                    "SIMD":        1,
                    "MW":          int(mw),
                    "IFMChannels": int(ifm),  
                    "ram_style":   "auto",
                }
            swg_counter += 1

        mva_counter += 1

    n_layers = mva_counter
    log.info(
        "Found %d compute layers (%d Conv, %d Linear).",
        n_layers, swg_counter, n_layers - swg_counter,
    )

    if n_layers == 0:
        log.warning(
            "No Conv/MatMul/Gemm nodes found. "
            "The ONNX may be in QONNX format with Quant wrapper nodes — "
            "this is normal. FINN will resolve the node names after "
            "step_qonnx_to_finn and step_convert_to_hw."
        )

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Summary + validation
# ─────────────────────────────────────────────────────────────────────────────

def print_config_summary(cfg: Dict[str, Any]) -> None:
    rows = [
        (name, v.get("op_type", "?"), v.get("PE", "-"), v.get("SIMD", "-"),
         v.get("MH", "-"), v.get("MW", "-"))
        for name, v in cfg.items()
        if name != "Defaults"
    ]
    if not rows:
        return
    col_w = max(len(r[0]) for r in rows) + 2
    header = f"{'Node':<{col_w}} {'Op':<8} {'PE':>4} {'SIMD':>6} {'MH':>6} {'MW':>6}"
    print("\n" + "═" * len(header))
    print("  Folding Config Baseline (PE=1, SIMD=1)")
    print("═" * len(header))
    print(header)
    print("─" * len(header))
    for name, op, pe, simd, mh, mw in rows:
        print(f"{name:<{col_w}} {str(op):<8} {str(pe):>4} {str(simd):>6} "
              f"{str(mh):>6} {str(mw):>6}")
    print("═" * len(header) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Export Brevitas model to ONNX and generate baseline folding config."
    )
    p.add_argument("--model_file",  required=True)
    p.add_argument("--model_class", default=None)
    p.add_argument("--config",      default=None, metavar="FILE")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    build_cfg = load_build_config(args.config)

    # ── 1. Load model ──────────────────────────────────────────────────────
    model = load_model_from_file(
        args.model_file, args.model_class, build_cfg["model_kwargs"]
    )
    log.info("Model: %s  |  params: %s",
             model.__class__.__name__,
             f"{sum(p.numel() for p in model.parameters()):,}")

    # ── 2. Export ONNX ─────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        onnx_path = export_to_onnx(
            model,
            tuple(build_cfg["input_shape"]),
            build_cfg["output_onnx"],
            build_cfg["opset"],
        )

    # ── 3. Parse ONNX → baseline folding config ────────────────────────────
    cfg = extract_folding_config_from_onnx(
        onnx_path  = onnx_path,
        board      = build_cfg["board"],
        fpga_part  = build_cfg["fpga_part"],
        clock_ns   = build_cfg["clock_ns"],
        target_fps = build_cfg["target_fps"],
    )

    # ── 4. Save ────────────────────────────────────────────────────────────
    print_config_summary(cfg)

    out_path = Path(build_cfg["output_cfg"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)

    log.info("Folding config written to: %s", out_path)
    log.info(
        "\nNext step:\n"
        "  python finn_raytune_optimizer.py \\\n"
        "      --baseline_cfg  %s \\\n"
        "      --build_script  full_build.py \\\n"
        "      --onnx_path     %s \\\n"
        "      --num_samples   40 \\\n"
        "      --objective lut_slack\n",
        out_path, onnx_path,
    )


if __name__ == "__main__":
    main()