"""
space_pruner.py
─────────────────────────────────────────────────────────────────────────────
LLM design space pruner for FINN configurations.

Workflow
────────
1. Reads build_config.json to understand hardware constraints.
2. Reads folding_config_baseline.json to understand layer topology (MH, MW,
   op_type) and calculates all mathematically valid PE/SIMD divisors.
3. If --results_csv is provided, reads past trial results so the LLM can
   reason about what has already been explored.
4. Sends a prompt to the LLM.
5. Validates the response strictly against the mathematical boundaries.
6. Saves the pruned space to --out_json for finn_raytune_optimizer.py.

"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import certifi
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    print("FATAL: Please install openai (pip install openai)")
    sys.exit(1)

# Load API keys from .env file if present.
# Variables already set in the environment take precedence (override=False).
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass  # python-dotenv not installed - fall back to environment variables only

# Force the SSL cert file to a path we know is valid inside the container
os.environ["SSL_CERT_FILE"] = certifi.where()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Layer topology + mathematical space
# ─────────────────────────────────────────────────────────────────────────────

def get_layer_context_and_space(
    baseline: Dict[str, Any],
) -> Tuple[Dict[str, dict], Dict[str, List[int]]]:
    """
    Universal version: Supports both clean names (MVAU_0) and legacy names (MVAU_hls_0).
    Returns:
        layer_context : metadata per logical layer (for the LLM to reason about)
        raw_space     : all mathematically valid PE/SIMD choices per hyperparameter
    """
    logical: Dict[str, Dict[str, Any]] = {}
    
    # 1. Normalize and deduplicate the baseline
    for node_name, node_cfg in baseline.items():
        if node_name == "Defaults":
            continue
            
        # Strip _hls_ or _rtl_ if present to get the clean logical name
        logical_name = node_name.replace("_hls_", "_").replace("_rtl_", "_")
        
        # Deduplicate: Add if new, OR if it's specifically the hls variant
        if logical_name not in logical or "_hls_" in node_name:
            logical[logical_name] = node_cfg

    layer_context: Dict[str, dict] = {}
    raw_space: Dict[str, List[int]] = {}

    # 2. Build the context and mathematical bounds
    for logical_name, node_cfg in sorted(logical.items()):
        # Try to extract the base layer type (e.g., MVAU from MVAU_0)
        m = re.match(r"^(.+)_(\d+)$", logical_name)
        layer_type = m.group(1) if m else "Unknown"

        mh = node_cfg.get("MH")
        mw = node_cfg.get("MW")

        layer_context[logical_name] = {
            "type":         layer_type,
            "op_type":      node_cfg.get("op_type", "N/A"),
            "MH_output_dim": mh,
            "MW_input_dim":  mw,
        }

        # Calculate perfect valid divisors
        if "PE" in node_cfg and mh is not None:
            raw_space[f"{logical_name}__PE"] = [
                v for v in range(1, mh + 1) if mh % v == 0
            ]

        if "SIMD" in node_cfg:
            limit_dim = node_cfg.get("IFMChannels") or node_cfg.get("MW")
            if limit_dim is not None:
                raw_space[f"{logical_name}__SIMD"] = [
                    v for v in range(1, limit_dim + 1) if limit_dim % v == 0
                ]

    return layer_context, raw_space


# ─────────────────────────────────────────────────────────────────────────────
# Past results loader
# ─────────────────────────────────────────────────────────────────────────────

def load_results_summary(
    results_csv: Optional[str],
    max_rows: int = 30,
) -> Optional[str]:
    """
    Read the trial results CSV and return a compact text summary for the LLM.
    Limits to the most recent max_rows rows to keep the prompt manageable.
    """
    if not results_csv or not Path(results_csv).exists():
        return None

    rows = []
    try:
        with open(results_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as exc:
        log.warning("Could not read results CSV: %s", exc)
        return None

    if not rows:
        return None

    recent = rows[-max_rows:]
    lines = [f"Past trial results ({len(rows)} total, showing last {len(recent)}):", ""]

    # 1. Output a condensed table for general trends
    param_keys = [k for k in recent[0].keys() if "__PE" in k or "__SIMD" in k]
    metric_keys = ["success", "throughput_fps", "lut_pct", "bram_pct", "dsp_pct", "objective"]
    metric_keys = [k for k in metric_keys if k in recent[0]]

    # We will still truncate the visual table to save tokens, but we will pass the exact best params below
    header = "trial | " + " | ".join(param_keys[:6]) + "... | " + " | ".join(metric_keys)
    lines.append(header)
    lines.append("-" * min(len(header), 120))

    for row in recent:
        param_vals  = " | ".join(str(row.get(k, "?")) for k in param_keys[:6])
        metric_vals = " | ".join(str(row.get(k, "?")) for k in metric_keys)
        lines.append(f"{row.get('trial_name', '?')} | {param_vals}... | {metric_vals}")

    # 2. Output the exact stats and FULL parameter dict for the Best Trial
    successes = [r for r in rows if r.get("success") == "1" or r.get("build_success") == "1"]
    failures  = [r for r in rows if r.get("success") == "0" or r.get("build_success") == "0"]
    
    lines.append("")
    lines.append(f"Overall: {len(successes)} successes, {len(failures)} failures out of {len(rows)} trials.")

    if successes:
        try:
            # Sort by objective
            best = min(successes, key=lambda r: float(r.get("objective", 1e9)))
            
            best_params = {k: int(best[k]) for k in param_keys if k in best and best[k].isdigit()}
            
            # Print both the objective and throughput (if available) for LLM context
            fps_str = best.get('throughput_fps') or best.get('throughput', 'N/A')
            lines.append(f"Best trial so far: {best.get('trial_name')} "
                         f"(objective={best.get('objective')}, throughput={fps_str} FPS)")
            lines.append(f"CRITICAL: Exact parameters for 'Best trial so far':\n{json.dumps(best_params)}")
        except Exception as e:
            log.warning("Could not parse best trial stats: %s", e)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def safe_json_parse(content: str) -> dict:
    """
    Parse LLM JSON output defensively. Falls back to a repair pass if the
    model emitted literal control characters (e.g. raw newlines) inside
    string values, which some reasoning models do despite JSON mode.
    """
    try:
        return json.loads(content, strict=False)
    except json.JSONDecodeError as e:
        log.warning("JSON parse failed (%s), attempting repair…", e)
        repaired = re.sub(r'(?<!\\)\n', '\\n', content)
        repaired = re.sub(r'(?<!\\)\t', '\\t', content)
        return json.loads(repaired, strict=False)

# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

def prompt_agent(
    client: OpenAI,
    build_cfg: dict,
    layer_context: dict,
    raw_space: dict,
    constraints: str,
    results_summary: Optional[str],
    pruning_round: int,
    model: str,
) -> Tuple[dict, str]:
    """
    Send the pruning request to the LLM.
    Returns (pruned_space_dict, reasoning_text).
    """
    board  = build_cfg.get("board",     "Unknown")
    fpga   = build_cfg.get("fpga_part", "Unknown")
    kwargs = build_cfg.get("model_kwargs", {})

    system_prompt = f"""You are an expert FINN/Xilinx FPGA architect.
TASK: Prune a PE/SIMD design space to accelerate optimizer convergence. Keep 2-5 strategic values per parameter.

--- ENVIRONMENT ---
TARGET: {board} ({fpga})
PRECISION: W={kwargs.get('weight_bit_width')}b, A={kwargs.get('act_bit_width')}b
ROUND: {pruning_round}

--- ACCELERATION HEURISTICS ---
1. BOTTLENECKS: Pipeline throughput equals the slowest layer. Prioritize high PE/SIMD on heavy layers (deep MVAUs, large MH/MW).
2. SCALING: Layer speed is proportional to PE × SIMD. To double speed, scale ONE parameter, not both.
3. RESOURCE TRADE-OFFS: 
   - PE increases BRAM (weight storage).
   - SIMD increases LUT/DSP (compute lanes).
   - React to past CSV metrics: If BRAM is >95%, scale SIMD. If DSP is >95%, scale PE.

--- STRICT RULES ---
- SUBSET ONLY: Output values MUST be a strict subset of the provided raw mathematical space.
- NON-BOTTLENECKS: If a layer is NOT a bottleneck, heavily favor low values, but you MUST keep at least one mid-range value (e.g., 8 or 16) in the array just in case there is the need to shift the bottleneck.
- BOTTLENECKS: If a layer IS a bottleneck, keep a mix of low (safe) and high (exploration) values.
- COMPLETENESS: You MUST include EVERY SINGLE KEY provided in the raw mathematical space in your output JSON. Do not omit layers (e.g. ConvolutionInputGenerator); simply prune them to low values (e.g., 1, 2, 4) if they are not bottlenecks.
- ANCHORED EXPLORATION: If a 'Best trial so far' exists in the history, you MUST include its exact PE and SIMD values in your new pruned space. Surround it with 1-2 adjacent values for fine-tuning. Finally, add at least 1 'wildcard' value (significantly higher or lower) to prevent local minima. EXCEPTION: Do not add high wildcards if you are in Emergency Mode.
- CRITICAL SURVIVAL RULE: Calculate the exact failure rate in the CSV. 
  * If failure rate > 80% (EMERGENCY MODE): You MUST physically include the lowest possible integers as the first elements in EVERY array and exclude high values.
  * If failure rate >= 99% (MAX EMERGENCY MODE): You MUST physically delete all numbers greater than 4 from every single array. Every array must ONLY contain the lowest possible integers. If you include any number > 4, the system will crash.
  * If success rate is high: DO NOT apply these emergency rules. Focus on throughput.
  
--- OUTPUT FORMAT ---
You MUST return ONLY a valid JSON object matching this exact schema:
{{
  "pruned_space": {{
    "MVAU_0__PE": [1, 2, 4],
    "MVAU_0__SIMD": [1, 2, 8]
  }},
  "reasoning": "Brief explanation of bottleneck identification and resource trade-offs."
}}"""

    history_section = ""
    if results_summary:
        history_section = f"\n--- PAST TRIAL RESULTS ---\n{results_summary}\n"

    user_prompt = f"""--- HUMAN CONSTRAINTS ---
{constraints}
{history_section}
--- LAYER TOPOLOGY ---
{json.dumps(layer_context, indent=2)}

--- RAW MATHEMATICAL DESIGN SPACE ---
{json.dumps(raw_space, indent=2)}
"""

    log.info("Sending pruning prompt to LLM (round %d) …", pruning_round)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=32768,
    )
    
    # Extract usage data
    usage = {
        "prompt": response.usage.prompt_tokens,
        "completion": response.usage.completion_tokens,
        "total": response.usage.total_tokens
    }

    choice = response.choices[0]
    message = choice.message

    log.info("finish_reason=%s", choice.finish_reason)
    log.debug("Full message object: %r", message)

    if choice.finish_reason == "length":
        reasoning_tokens = getattr(message, "reasoning_content", None)
        raise RuntimeError(
            f"Response truncated (finish_reason=length) before producing content. "
            f"completion_tokens={response.usage.completion_tokens}. "
            f"Had reasoning_content: {reasoning_tokens is not None}"
        )

    if message.content is None:
        raise RuntimeError(
            f"Model returned no content (finish_reason={choice.finish_reason}). "
            f"Raw message: {message!r}"
        )

    raw = safe_json_parse(message.content)

    pruned  = raw.get("pruned_space", raw)   # fallback: treat entire response as space
    reasoning = raw.get("reasoning", "No reasoning provided.")
    return pruned, reasoning, usage



def prompt_agent_multi(
    client: OpenAI,
    model_payloads: Dict[str, Dict[str, Any]],
    constraints: str,
    pruning_round: int,
    model: str,
) -> Tuple[Dict[str, dict], Dict[str, str], str, dict]:
    """
    model_payloads: {
        model_key: {
            "build_cfg": dict,
            "layer_context": dict,
            "raw_space": dict,
            "results_summary": Optional[str],
        },
        ...
    }
    Returns (per_model_pruned_spaces, per_model_reasoning, overall_reasoning, usage)
    """
    model_keys = list(model_payloads.keys())
    board_info = {
        mk: {"board": p["build_cfg"].get("board", "Unknown"),
             "fpga_part": p["build_cfg"].get("fpga_part", "Unknown")}
        for mk, p in model_payloads.items()
    }

    system_prompt = f"""You are an expert FINN/Xilinx FPGA architect.
TASK: You are pruning PE/SIMD design spaces for MULTIPLE neural networks that will be
deployed SIMULTANEOUSLY on the SAME physical FPGA board. Their resource usage
(LUT, DSP, BRAM, FF) is ADDITIVE across networks — the sum must fit on one device.
Keep 2-5 strategic values per parameter, per network.

--- NETWORKS ON THIS BOARD ({len(model_keys)} total) ---
{json.dumps(board_info, indent=2)}

ROUND: {pruning_round}

--- ACCELERATION HEURISTICS (apply within each network) ---
1. BOTTLENECKS: Pipeline throughput equals the slowest layer in that network.
   Prioritize high PE/SIMD on heavy layers (deep MVAUs, large MH/MW).
2. SCALING: Layer speed is proportional to PE × SIMD. To double speed, scale ONE
   parameter, not both.
3. RESOURCE TRADE-OFFS:
   - PE increases BRAM (weight storage).
   - SIMD increases LUT/DSP (compute lanes).
   - React to past CSV metrics: if BRAM is >95%, scale SIMD; if DSP is >95%, scale PE.

--- CROSS-NETWORK BUDGET HEURISTIC ---
Because resource usage is additive across all networks above, do NOT let every network
independently chase maximum utilisation; leave headroom so their SUM stays under the
board's physical limits. If one network's past results show it dominating resource
usage, prune its bottleneck layers more conservatively and give the other network(s)
more room to grow. If a network has no history yet, treat it as unconstrained but note
that its neighbors need headroom too. 
Your GENERAL RULE is: MAXIMIZE THE THROUGHPUT of the networks while MINIMIZING the resource waste;
IF A LAYER DON'T NEED MORE RESOURCES, DON'T GIVE HIM THE POSSIBILITY TO HAVE MORE RESOURCES. 
Still, always aim at increasing the throughput.

--- STRICT RULES (apply to every network's space) ---
- SUBSET ONLY: Output values MUST be a strict subset of that network's raw mathematical space.
- NON-BOTTLENECKS: If a layer is NOT a bottleneck, heavily favor low values, but keep at
  least one mid-range value (e.g., 8 or 16) in case the bottleneck shifts.
- BOTTLENECKS: If a layer IS a bottleneck, keep a mix of low (safe) and high (exploration) values.
- COMPLETENESS: For EVERY network, you MUST include EVERY key from that network's raw
  mathematical space in your output. Do not omit layers; prune non-bottlenecks to low
  values (e.g., 1, 2, 4) instead.
- ANCHORED EXPLORATION: If a 'Best trial so far' exists in a network's history, include
  its exact PE/SIMD values in that network's new space, surrounded by 1-2 adjacent
  values, plus at least one wildcard (significantly higher or lower) to avoid local
  minima. EXCEPTION: no high wildcards in Emergency Mode.
- CRITICAL SURVIVAL RULE (per network, independently): Calculate that network's failure
  rate from its own CSV history.
  * failure rate > 80% (EMERGENCY MODE): put the lowest possible integers first in
    every array for that network and exclude high values.
  * failure rate >= 99% (MAX EMERGENCY MODE): delete all numbers greater than 4 from
    every array for that network. Any number > 4 will crash the system.
  * otherwise: no emergency rules, focus on throughput.

--- OUTPUT FORMAT ---
Return ONLY a valid JSON object with EXACTLY these top-level keys — the network keys
{model_keys} plus "overall_reasoning":

{{
  "{model_keys[0]}": {{
    "pruned_space": {{ "MVAU_0__PE": [1, 2, 4], "MVAU_0__SIMD": [1, 2, 8] }},
    "reasoning": "Brief explanation of this network's bottlenecks and trade-offs."
  }},
  ... one entry per network key ...,
  "overall_reasoning": "How resource budget was balanced across all networks."
}}"""

    sections = []
    for mk, payload in model_payloads.items():
        history_section = ""
        if payload["results_summary"]:
            history_section = f"\n--- PAST TRIAL RESULTS FOR '{mk}' ---\n{payload['results_summary']}\n"
        sections.append(f"""
════════ NETWORK: {mk} ════════
{history_section}
--- LAYER TOPOLOGY ({mk}) ---
{json.dumps(payload["layer_context"], indent=2)}

--- RAW MATHEMATICAL DESIGN SPACE ({mk}) ---
{json.dumps(payload["raw_space"], indent=2)}
""")

    user_prompt = f"""--- HUMAN CONSTRAINTS ---
{constraints}
{''.join(sections)}
"""

    log.info("Sending multi-model pruning prompt to LLM (round %d, %d networks) …",
             pruning_round, len(model_keys))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=32768,
    )

    usage = {
        "prompt": response.usage.prompt_tokens,
        "completion": response.usage.completion_tokens,
        "total": response.usage.total_tokens,
    }

    choice = response.choices[0]
    message = choice.message
    log.info("finish_reason=%s", choice.finish_reason)

    if choice.finish_reason == "length":
        raise RuntimeError(
            f"Response truncated (finish_reason=length) before producing content. "
            f"completion_tokens={response.usage.completion_tokens}."
        )
    if message.content is None:
        raise RuntimeError(
            f"Model returned no content (finish_reason={choice.finish_reason}). "
            f"Raw message: {message!r}"
        )

    raw = safe_json_parse(message.content)
    overall_reasoning = raw.pop("overall_reasoning", "No overall reasoning provided.")

    per_model_spaces:    Dict[str, dict] = {}
    per_model_reasoning: Dict[str, str]  = {}
    for mk in model_keys:
        entry = raw.get(mk, {})
        if not isinstance(entry, dict):
            log.warning("[%s] LLM output for this key was not an object — treating as empty.", mk)
            entry = {}
        per_model_spaces[mk]    = entry.get("pruned_space", {})
        per_model_reasoning[mk] = entry.get("reasoning", "No reasoning provided.")

    return per_model_spaces, per_model_reasoning, overall_reasoning, usage


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_finalize(
    llm_space: dict,
    raw_space: dict,
) -> dict:
    """
    Strictly validate LLM output against mathematically valid divisors.
    Falls back to the full raw space for any parameter the LLM got wrong.
    """
    final_space = {}
    n_ok      = 0
    n_reverted = 0

    for key, valid_divisors in raw_space.items():
        proposed = llm_space.get(key, [])
        # Keep only values that are genuinely valid divisors
        safe_subset = sorted(set(v for v in proposed if v in valid_divisors))

        if not safe_subset:
            log.warning(
                "  %-45s LLM proposed %s — INVALID, reverting to full: %s",
                key, proposed, valid_divisors,
            )
            final_space[key] = valid_divisors
            n_reverted += 1
        else:
            final_space[key] = safe_subset
            n_ok += 1

    log.info("Validation: %d params accepted, %d reverted to full space.", n_ok, n_reverted)
    return final_space


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pruner(
    build_cfg_path:    str,
    baseline_cfg_path: str,
    out_json_path:     str,
    constraints:       str,
    results_csv:       Optional[str] = None,
    pruning_round:     int = 1,
    api_key:           Optional[str] = None,
    model:             str = "qwen-3.8-27b",
) -> Dict[str, List[int]]:
    """
    Main pruning logic, callable both from CLI and from the optimizer.
    Returns the final pruned space dict.
    """
    build_cfg    = json.loads(Path(build_cfg_path).read_text())
    baseline_cfg = json.loads(Path(baseline_cfg_path).read_text())

    layer_context, raw_space = get_layer_context_and_space(baseline_cfg)
    results_summary = load_results_summary(results_csv)

    if results_summary:
        log.info("Loaded past results summary (%d chars).", len(results_summary))
    else:
        log.info("No past results provided — first pruning round.")

    resolved_key = api_key or os.environ.get("CEREBRAS_API_KEY")
    if not resolved_key:
        log.error("No API key found. Set the variable or pass --api_key.")
        sys.exit(1)

    client = OpenAI(
        api_key=resolved_key,
        base_url="https://api.cerebras.ai/v1",
    )

    try:
        # Catch the new usage dict here
        llm_space, reasoning, usage = prompt_agent(
            client, build_cfg, layer_context, raw_space,
            constraints, results_summary, pruning_round, model,
        )
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        llm_space = raw_space
        reasoning = f"LLM failed ({exc})"
        usage = {"prompt": 0, "completion": 0, "total": 0}

    log.info("LLM reasoning: %s", reasoning)

    final_space = validate_and_finalize(llm_space, raw_space)

    out_path = Path(out_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final_space, indent=2))
    log.info("Pruned space written to: %s", out_json_path)

    reasoning_log_path = out_path.parent / "llm_reasoning_history.txt"
    with open(reasoning_log_path, "a") as f:
        f.write(f"═══ PRUNING ROUND {pruning_round} ═══\n")
        f.write(f"TOKENS: {usage['total']} (Prompt: {usage['prompt']}, Completion: {usage['completion']})\n")
        f.write(f"REASONING: {reasoning}\n\n")

    for k, v in sorted(final_space.items()):
        log.info("  %-50s %s", k, v)

    return final_space



def run_pruner_multi(
    models:        Dict[str, Dict[str, Any]],
    constraints:   str,
    pruning_round: int = 1,
    api_key:       Optional[str] = None,
    model:         str = "qwen-3.8-27b",
) -> Dict[str, Dict[str, List[int]]]:
    """
    Multi-model pruning entry point.

    `models` = {
        model_key: {
            "build_cfg_path":    str,
            "baseline_cfg_path": str,
            "out_json_path":     str,
            "results_csv":       Optional[str],
        },
        ...
    }
    Returns { model_key: final_pruned_space_dict }.
    """
    model_payloads: Dict[str, Dict[str, Any]] = {}
    raw_spaces:     Dict[str, dict]            = {}

    for mk, cfg in models.items():
        build_cfg    = json.loads(Path(cfg["build_cfg_path"]).read_text())
        baseline_cfg = json.loads(Path(cfg["baseline_cfg_path"]).read_text())
        layer_context, raw_space = get_layer_context_and_space(baseline_cfg)
        results_summary = load_results_summary(cfg.get("results_csv"))

        if results_summary:
            log.info("[%s] Loaded past results summary (%d chars).", mk, len(results_summary))
        else:
            log.info("[%s] No past results provided.", mk)

        model_payloads[mk] = dict(
            build_cfg       = build_cfg,
            layer_context   = layer_context,
            raw_space       = raw_space,
            results_summary = results_summary,
        )
        raw_spaces[mk] = raw_space

    resolved_key = api_key or os.environ.get("CEREBRAS_API_KEY")
    if not resolved_key:
        log.error("No API key found. Set the variable or pass --api_key.")
        sys.exit(1)

    client = OpenAI(api_key=resolved_key, base_url="https://api.cerebras.ai/v1")

    try:
        llm_spaces, llm_reasoning, overall_reasoning, usage = prompt_agent_multi(
            client, model_payloads, constraints, pruning_round, model,
        )
    except Exception as exc:
        log.error("Multi-model LLM call failed: %s — falling back to full raw spaces.", exc)
        llm_spaces        = {mk: raw_spaces[mk] for mk in models}
        llm_reasoning      = {mk: f"LLM failed ({exc})" for mk in models}
        overall_reasoning  = f"LLM failed ({exc})"
        usage              = {"prompt": 0, "completion": 0, "total": 0}

    log.info("LLM overall reasoning: %s", overall_reasoning)

    final_spaces: Dict[str, Dict[str, List[int]]] = {}
    for mk, cfg in models.items():
        log.info("─── Validating space for '%s' ───", mk)
        final_space = validate_and_finalize(llm_spaces.get(mk, {}), raw_spaces[mk])
        final_spaces[mk] = final_space

        out_path = Path(cfg["out_json_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(final_space, indent=2))
        log.info("[%s] Pruned space written to: %s", mk, cfg["out_json_path"])

        reasoning_log_path = out_path.parent / "llm_reasoning_history.txt"
        with open(reasoning_log_path, "a") as f:
            f.write(f"═══ PRUNING ROUND {pruning_round} ═══\n")
            f.write(f"TOKENS (single call shared across {len(models)} models): "
                    f"{usage['total']} (Prompt: {usage['prompt']}, Completion: {usage['completion']})\n")
            f.write(f"OVERALL REASONING: {overall_reasoning}\n")
            f.write(f"MODEL REASONING ({mk}): {llm_reasoning.get(mk, 'N/A')}\n\n")

        for k, v in sorted(final_space.items()):
            log.info("  %-50s %s", k, v)

    return final_spaces

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="LLM-based FINN design space pruner.")
    p.add_argument("--build_cfg",    required=True)
    p.add_argument("--baseline_cfg", required=True)
    p.add_argument("--out_json",     required=True)
    p.add_argument("--constraints",  required=True)
    p.add_argument("--results_csv",  default=None,
                   help="CSV of past trial results (optional)")
    p.add_argument("--pruning_round", type=int, default=1)
    p.add_argument("--api_key",      default=None)
    p.add_argument("--model",        default="qwen-3.8-27b")
    args = p.parse_args()

    run_pruner(
        build_cfg_path    = args.build_cfg,
        baseline_cfg_path = args.baseline_cfg,
        out_json_path     = args.out_json,
        constraints       = args.constraints,
        results_csv       = args.results_csv,
        pruning_round     = args.pruning_round,
        api_key           = args.api_key,
        model             = args.model,
    )


if __name__ == "__main__":
    main()