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
        --model_file  models/cnv_w2a2.py \
        --model_class CNV     \
        --config      build_configs/cnv_w2a2_build_config.json

    python generate_folding_config.py \
        --onnx_file   models/cnv_w1a1.onnx \
        --config      build_configs/cnv_w1a1_build_config.json
"""

import argparse
import importlib.util
import inspect
import json
import logging
import os
import sys
import shutil
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

# Known kwarg aliases: maps your standard JSON keys to common alternative
# parameter names used in third-party repos.
_KWARG_ALIASES = {
    "num_classes":      ["num_classes", "n_classes", "classes"],
    "weight_bit_width": ["weight_bit_width", "weight_bit", "wbits", "w_bits"],
    "act_bit_width":    ["act_bit_width", "act_bit", "abits", "a_bits"],
    "inp_bit_width":    ["inp_bit_width", "in_bit_width", "input_bit_width",
                         "in_bit", "inp_bit"],
    "in_channels":      ["in_channels", "in_ch", "input_channels",
                         "input_ch", "n_channels"],
}

def _remap_kwargs(
    model_kwargs: Dict[str, Any],
    target_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Remap model_kwargs to match the actual parameter names of a constructor.

    Strategy:
      1. Keep any key that already matches a parameter name exactly.
      2. For remaining keys, try each known alias — use the first that matches.
      3. Drop any key that has no matching parameter (with a warning).
    """
    remapped = {}
    for json_key, value in model_kwargs.items():
        if json_key in target_params:
            remapped[json_key] = value
            continue
        # Try aliases
        aliases = _KWARG_ALIASES.get(json_key, [])
        matched = next((a for a in aliases if a in target_params), None)
        if matched:
            log.info("  kwarg '%s' → remapped to '%s'", json_key, matched)
            remapped[matched] = value
        else:
            log.warning(
                "  kwarg '%s' has no matching parameter in constructor — skipping.",
                json_key,
            )
    return remapped


def _try_instantiate(target, model_kwargs: Dict[str, Any]):
    """
    Try to instantiate `target` with the provided kwargs, falling back
    gracefully through three strategies:

      1. Direct kwargs             — works for well-behaved models
      2. Remapped kwargs           — works for repos with different param names
      3. Only matching kwargs      — drops unknown keys, uses constructor defaults
    """
    sig = inspect.signature(target.__init__)
    params = {
        k: v for k, v in sig.parameters.items()
        if k not in ("self", "args", "kwargs")
    }

    # Strategy 1: direct
    try:
        return target(**model_kwargs)
    except TypeError as e1:
        log.debug("Direct kwargs failed: %s", e1)

    # Strategy 2: remap aliases
    remapped = _remap_kwargs(model_kwargs, params)
    try:
        return target(**remapped)
    except TypeError as e2:
        log.debug("Remapped kwargs failed: %s", e2)

    # Strategy 3: keep only params the constructor actually accepts
    safe_kwargs = {k: v for k, v in remapped.items() if k in params}
    skipped = set(model_kwargs.keys()) - set(safe_kwargs.keys())
    if skipped:
        log.warning(
            "Instantiating with reduced kwargs (skipped: %s). "
            "Constructor defaults will be used for missing params.",
            sorted(skipped),
        )
    return target(**safe_kwargs)


def _load_module_with_package(model_path: Path):
    """
    Load a Python file as a module. If it uses relative imports (i.e. it is
    part of a package), add its parent directory to sys.path so the imports
    resolve correctly.
    """
    parent = str(model_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
        added_to_path = True
    else:
        added_to_path = False

    # Try as a package member first (handles relative imports like `from .common import`)
    package_name = model_path.parent.name
    module_name  = model_path.stem
    try:
        import importlib
        full_name = f"{package_name}.{module_name}"
        mod = importlib.import_module(full_name)
        log.debug("Loaded as package member: %s", full_name)
        return mod
    except (ImportError, ModuleNotFoundError):
        pass

    # Fall back to standalone file loading
    spec = importlib.util.spec_from_file_location("_user_model", model_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model_from_file(
    model_file: str,
    model_class: Optional[str],
    model_kwargs: Dict[str, Any],
) -> torch.nn.Module:
    """
    Load a PyTorch/Brevitas model from a Python file and instantiate it.

    Handles:
      - Models with a get_model(**kwargs) / build_model / create_model factory
      - Models with a class constructor that uses standard kwargs
      - Models from third-party repos with different parameter names
        (remapped via _KWARG_ALIASES)
      - Models that are part of a package (relative imports resolved
        by adding the parent directory to sys.path)
    """
    model_path = Path(model_file).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    mod = _load_module_with_package(model_path)

    # ── 1. Explicit class name given ──────────────────────────────────────
    if model_class:
        target = getattr(mod, model_class, None)
        if target is None:
            raise AttributeError(f"Class '{model_class}' not found in {model_file}.")
        model = _try_instantiate(target, model_kwargs)
        model.eval()
        return model

    # ── 2. Look for standard factory functions ────────────────────────────
    for fname in ("get_model", "build_model", "create_model"):
        fn = getattr(mod, fname, None)
        if fn and callable(fn):
            try:
                log.info("Using factory function: %s()", fname)
                model = fn(**model_kwargs)
                model.eval()
                return model
            except TypeError:
                # Factory may have different param names — try remapping
                sig    = inspect.signature(fn)
                params = {k: v for k, v in sig.parameters.items()
                          if k not in ("self", "args", "kwargs")}
                remapped = _remap_kwargs(model_kwargs, params)
                try:
                    model = fn(**remapped)
                    model.eval()
                    return model
                except TypeError as e:
                    log.debug("Factory %s failed even after remapping: %s", fname, e)

    # ── 3. Find any nn.Module subclass in the file ────────────────────────
    candidates = [
        obj for name, obj in inspect.getmembers(mod, inspect.isclass)
        if issubclass(obj, torch.nn.Module)
        and obj is not torch.nn.Module
        and obj.__module__ == getattr(mod, "__name__", "_user_model")
    ]

    if not candidates:
        # Widen search - include any nn.Module subclass visible in the module
        candidates = [
            obj for name, obj in inspect.getmembers(mod, inspect.isclass)
            if issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module
        ]

    if not candidates:
        raise RuntimeError(
            f"No nn.Module subclass found in {model_file}. "
            "Use --model_class to specify the class name explicitly."
        )

    if len(candidates) > 1:
        log.warning(
            "Multiple nn.Module subclasses found: %s. "
            "Using the first one: %s. "
            "Use --model_class to specify explicitly.",
            [c.__name__ for c in candidates],
            candidates[0].__name__,
        )

    target = candidates[0]
    log.info("Auto-detected model class: %s", target.__name__)
    model = _try_instantiate(target, model_kwargs)
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

def stage_onnx_into_dataset(onnx_src: str, build_cfg: Dict[str, Any]) -> str:
    """Copy a pre-built ONNX to the location export_to_onnx() would have written it to."""
    src = Path(onnx_src).resolve()
    raw = build_cfg.get("output_onnx")
    if raw and raw != CONFIG_DEFAULTS["output_onnx"]:
        dest = Path(raw)                                    # same path the .py flow uses
    else:
        dest = Path("dataset") / src.stem / f"{src.stem}.onnx"
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest == src:
        log.info("ONNX is already at %s - nothing to copy.", dest)
    else:
        shutil.copy2(src, dest)
        log.info("Copied ONNX: %s -> %s", src, dest)
    return str(dest)

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

def _get_mh_mw(node, initializer_map, producer_map):
    """Return (MH, MW) for a Conv/MatMul/Gemm node from its weight tensor."""
    mh, mw = 1, 1
    if len(node.input) > 1:
        w = _find_weights(node.input[1], initializer_map, producer_map)
        if w is not None:
            if len(w.shape) == 4:           # Conv: [OC, IC, KH, KW]
                mh = w.shape[0]
                mw = w.shape[1] * w.shape[2] * w.shape[3]
            elif len(w.shape) == 2:         # MatMul/Gemm: [out, in]
                mh = w.shape[0]
                mw = w.shape[1]
    return max(1, mh), max(1, mw)


def extract_folding_config_from_onnx(
    onnx_path: str,
    board: str,
    fpga_part: str,
    clock_ns: float,
    target_fps: Optional[float],
    restype_lut: bool = False,
) -> Dict[str, Any]:
    """
    Parse the exported ONNX and build a baseline folding config with PE=1 / SIMD=1.
    Uses the clean Universal Naming format (e.g., MVAU_0).
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

        mh, mw = _get_mh_mw(node, initializer_map, producer_map)

        log.debug(
            "  %s node — op=%s  MH=%d  MW=%d",
            node.name or node.op_type, node.op_type, mh, mw,
        )

        # MVAU — Clean Universal Name
        key = f"MVAU_{mva_counter}"
        layer_dict = {
            "PE":        1,
            "SIMD":      1,
            "MH":        int(mh),
            "MW":        int(mw),
            "ram_style": "auto",
            "op_type":   node.op_type,
        }
        
        # Apply the missing restype_lut flag!
        if restype_lut:
            layer_dict["resType"] = "lut"
            
        cfg[key] = layer_dict

        # ConvolutionInputGenerator — Clean Universal Name
        if node.op_type == "Conv":
            in_channels = 1
            if len(node.input) > 1:
                w = _find_weights(node.input[1], initializer_map, producer_map)
                if w is not None and len(w.shape) == 4:
                    in_channels = int(w.shape[1])  # [OC, IC, KH, KW]
            
            swg_key = f"ConvolutionInputGenerator_{swg_counter}"
            cfg[swg_key] = {
                "SIMD":        1,
                "MW":          int(mw),
                "IFMChannels": in_channels, # Added for your Universal pruner
                "inp_channels": in_channels,
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
        description=(
            "Generate a baseline FINN folding config from either:\n"
            "  (a) a Brevitas model file  →  export ONNX then parse\n"
            "  (b) an existing .onnx file →  parse directly, no export\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Input: mutually exclusive — either a Python model or a pre-built ONNX
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--model_file",
        metavar="FILE.py",
        help="Brevitas model Python file. The script exports it to ONNX first.",
    )
    src.add_argument(
        "--onnx_file",
        metavar="FILE.onnx",
        help="Existing ONNX file (e.g. from the FINN repo). "
             "Skips model loading and export entirely.",
    )
    p.add_argument("--model_class", default=None,
                   help="Class name inside --model_file (optional).")
    p.add_argument("--config",      default=None, metavar="FILE",
                   help="JSON build config. Required for board/fpga_part/output_cfg "
                        "when using --onnx_file.")
    p.add_argument("--output_cfg",  default=None,
                   help="Override output path for the folding config JSON.")
    p.add_argument("--verbose",     action="store_true")
    p.add_argument("--restype_lut", action="store_true",
                   help="Force all MVAU layers to use resType: lut to save DSPs.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    build_cfg = load_build_config(args.config)

    # ── 1 & 2: Model loading + ONNX export (skipped when --onnx_file given) ─
    if args.onnx_file:
        src_path = str(Path(args.onnx_file).resolve())
        if not Path(src_path).exists():
            log.error("ONNX file not found: %s", src_path)
            sys.exit(1)
        # Same layout as the .py flow: the ONNX ends up at output_onnx
        onnx_path  = stage_onnx_into_dataset(src_path, build_cfg)
        model_stem = Path(onnx_path).stem
    else:
        # ── 1. Load model ──────────────────────────────────────────────────
        model = load_model_from_file(
            args.model_file, args.model_class, build_cfg["model_kwargs"]
        )
        log.info("Model: %s  |  params: %s",
                 model.__class__.__name__,
                 f"{sum(p.numel() for p in model.parameters()):,}")

        # ── 2. Export ONNX ─────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            onnx_path = export_to_onnx(
                model,
                tuple(build_cfg["input_shape"]),
                build_cfg["output_onnx"],
                build_cfg["opset"],
            )
        model_stem = Path(onnx_path).stem

    # ── 3. Parse ONNX → baseline folding config ────────────────────────────
    cfg = extract_folding_config_from_onnx(
        onnx_path   = onnx_path,
        board       = build_cfg["board"],
        fpga_part   = build_cfg["fpga_part"],
        clock_ns    = build_cfg["clock_ns"],
        target_fps  = build_cfg["target_fps"],
        restype_lut = args.restype_lut,
    )

    # ── 4. Resolve output path ─────────────────────────────────────────────
    if args.output_cfg:
        out_path = Path(args.output_cfg)
    else:
        # If the config specifies an output_cfg use it, otherwise derive from ONNX stem
        raw = build_cfg.get("output_cfg", "")
        if raw and raw != CONFIG_DEFAULTS["output_cfg"]:
            out_path = Path(raw)
        else:
            out_path = Path(f"dataset/config_files/{model_stem}/folding_config_baseline.json")

    # ── 5. Save ────────────────────────────────────────────────────────────
    print_config_summary(cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)

    log.info("Folding config written to: %s", out_path)


if __name__ == "__main__":
    main()