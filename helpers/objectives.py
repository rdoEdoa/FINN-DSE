from typing import Mapping, Dict, Any, Optional, Tuple

PENALTY = 10
PENALTY_HARD = 10000


def compute_objective(
    metrics: dict,
    objective: str,
    device: Mapping[str, float],
) -> float:
    if not metrics:
        return PENALTY_HARD

    lut  = metrics.get("lut", PENALTY)
    dsp  = metrics.get("dsp", 0.0)
    bram = metrics.get("bram", 0.0)
    ff   = metrics.get("ff", 0.0)
    tp   = metrics.get("throughput_fps", 0.0)

    # Normalized resource utilization
    lut_u  = lut / device["LUT"] if lut > 0 else 0.0
    dsp_u  = dsp / device["DSP"] if dsp > 0 else 0.0
    bram_u = bram / device["BRAM"] if bram > 0 else 0.0
    ff_u   = ff / device["FF"] if ff > 0 else 0.0

    # Reject configurations that exceed board capacity
    if lut_u > 1 or dsp_u > 1 or bram_u > 1 or ff_u > 1:
        return PENALTY_HARD

    if objective == "throughput":
        return -tp if tp > 0 else PENALTY_HARD

    elif objective == "resource_avg":
        avg = (lut_u + dsp_u + bram_u + ff_u) / 4
        return avg if tp > 0 else PENALTY_HARD

    elif objective == "balanced":
        if tp <= 0:
            return PENALTY_HARD

        return (
            -tp
            + lut_u * 10
            + bram_u * 10
            + ff_u * 10
            + dsp_u * 10
        )

    else:
        raise ValueError(f"Unknown objective: '{objective}'")

def compute_joint_objective(
    results: Dict[str, Dict[str, Any]],
    objective: str,
    instance_counts: Optional[Dict[str, int]] = None,
    *,
    device: Mapping[str, float],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Sums resource fractions ACROSS models against the shared board budget."""
    instance_counts = instance_counts or {}
    for ms, r in results.items():
        if not r["success"] or not r["metrics"]:
            return PENALTY_HARD, {}

    lut_u = dsp_u = bram_u = ff_u = 0.0
    tp_total = 0.0
    per_model: Dict[str, Dict[str, float]] = {}

    for ms, r in results.items():
        count = instance_counts.get(ms, 1)
        m = r["metrics"]
        lut, dsp, bram, ff = m.get("lut", 0.0), m.get("dsp", 0.0), m.get("bram", 0.0), m.get("ff", 0.0)
        l_u = lut / device["LUT"]
        d_u = dsp / device["DSP"] if dsp > 0 else 0.0
        b_u = bram / device["BRAM"] if bram > 0 else 0.0
        f_u = ff / device["FF"] if ff > 0 else 0.0
        if l_u > 1 or d_u > 1 or b_u > 1 or f_u > 1:
            return PENALTY_HARD, {}
        lut_u += l_u * count
        dsp_u += d_u * count 
        bram_u += b_u * count
        ff_u += f_u * count
        tp_total += m.get("throughput_fps", 0.0) * count
        per_model[ms] = {"lut_u": l_u, "dsp_u": d_u, "bram_u": b_u, "ff_u": f_u,
                          "throughput": m.get("throughput_fps", 0.0),
                          "lut_contrib": l_u * count, "dsp_contrib": d_u * count,   
                        "bram_contrib": b_u * count, "ff_contrib": f_u * count,     
                        "throughput_contrib": m.get("throughput_fps", 0.0) * count, 
                        "instance_count": count}

    if lut_u > 1 or dsp_u > 1 or bram_u > 1 or ff_u > 1:
        return PENALTY_HARD, per_model

    if objective == "throughput":
        return (-tp_total, per_model) if tp_total > 0 else (PENALTY_HARD, per_model)
    elif objective == "balanced":
        if tp_total <= 0:
            return PENALTY_HARD, per_model
        return -tp_total + lut_u * 10 + bram_u * 10 + ff_u * 10 + dsp_u * 10, per_model
    elif objective == "resource_avg":
        if tp_total <= 0:
            return PENALTY_HARD, per_model
        return (lut_u + dsp_u + bram_u + ff_u) / 4, per_model
    else:
        raise ValueError(f"Unknown objective: '{objective}'")