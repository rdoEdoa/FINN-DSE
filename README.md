# FINN-DSE — Design Space Exploration of FINN Folding Parameters

Automated exploration of the **PE/SIMD folding space** for quantized neural
networks deployed on FPGA with [FINN](https://github.com/Xilinx/finn). Given an ONNX
model and a baseline folding configuration, the framework searches for folding
parameters that maximize throughput, minimize resource usage, or balance both —
optionally pruning the space with an LLM between search rounds.

The toolchain has three entry points:

| Script | Purpose |
| --- | --- |
| `generate_folding_config.py` | Export a Brevitas/PyTorch model (or ingest an ONNX) and generate a baseline `PE=1, SIMD=1` folding config. |
| `finn_raytune_optimizer_ai.py` | Multi-model joint search with adaptive, LLM-driven space pruning. |
| `space_pruner.py` | Standalone LLM pruner that shrinks the PE/SIMD search space from the baseline topology. |

---

## How it works

```
Brevitas model ──┐
                 ├──► generate_folding_config.py ──► folding_config_baseline.json
ONNX model ──────┘                                        │
                                                          ▼
                                       ┌──────────────────────────────────┐
                                       │  Ray Tune search (per trial)     │
                                       │  1. sample PE/SIMD divisors      │
                                       │  2. patch baseline config        │
                                       │  3. run FINN build / estimate    │
                                       │  4. parse reports → objective    │
                                       └──────────────────────────────────┘
                                                          │
                       ┌──────────────────────────────────┴─────────────────┐
                       ▼                                                    ▼
          folding_config_best_<strategy>.json                       LLM pruning rounds
```

1. **Baseline generation** parses the ONNX weight tensors to recover each
   compute layer's `MH` (output dim) and `MW` (input dim), then writes a config
   with `PE = 1` and `SIMD = 1` for every layer. Following FINN's convention,
   Conv weights `[OC, IC, KH, KW]` give `MH = OC`, `MW = IC·KH·KW`. For MatMul
   the weight *as the node receives it* is `[in, out]` (`MW = in`, `MH = out`),
   so the `Transpose` Brevitas inserts after the stored `[out, in]` initializer
   is taken into account; Gemm weights are read as `[out, in]` when
   `transB = 1` and `[in, out]` otherwise. Layer names are
   normalized to the logical form `MVAU_0`, `MVAU_1`, … which the tuner later
   expands into the `MVAU_hls_0` / `MVAU_rtl_0` pairs FINN expects.
2. **Search space** contains one `PE` and one `SIMD` hyperparameter per logical
   layer. Only divisors of `MH` / `MW` are offered, so FINN never receives an
   illegal value.
3. **Each trial** patches the baseline, writes a folding config, invokes a build
   script as a subprocess, and reads the FINN JSON reports (`status.json`,
   `ooc_synth_and_timing.json`, `rtlsim_performance.json`, or the estimate
   reports) into a scalar objective. Patching overrides only `PE`/`SIMD`; every
   other FINN attribute in the baseline (`resType`, `ram_style`, `mem_mode`, …)
   is carried into both the `_hls_` and `_rtl_` entries.
4. **The best configuration** — the best feasible trial across *all* rounds —
   is re-expanded to `_hls_` / `_rtl_` names and saved next to the baseline.

---

## Repository layout

```
.
├── generate_folding_config.py       # Baseline folding-config generator
├── finn_raytune_optimizer_ai.py     # Multi-model tuner + LLM pruning
├── space_pruner.py                  # LLM design-space pruner
├── build_config.template.json       # Template for model build configs
├── build_configs/                   # Per-model build configuration
│   └── <model>_build_config.json
├── custom_builds/                   # Per-model FINN build scripts
│   └── <model>_build.py
├── helpers/
│   ├── boards.py                    # FPGA resource budgets + Xilinx parts
│   ├── build.py                     # FINN subprocess + feasibility check
│   ├── folding.py                   # Config patching/sanitizing
│   ├── metrics.py                   # Report parsing / warm starts
│   ├── objectives.py                # Single and joint objectives
│   ├── trainable.py                 # Ray Tune joint trainable
│   └── trials.py                    # CSV + thread-safe counters
├── models/                          # Model definitions and shipped ONNX files
│   ├── cnv.py
│   ├── common.py
│   ├── tensor_norm.py
│   ├── MLP_W3A3.onnx
│   ├── cnv_w1a1.onnx
│   ├── mlp_w2a2.onnx
│   ├── resnet18.onnx
│   └── resnet50.onnx
└── requirements.txt
```

`dataset/` and `raytune*` are **generated at runtime** and are gitignored.

### Expected runtime layout

```
dataset/
├── <model>/
│   └── <model>.onnx
├── config_files/
│   ├── <model>/
│   │   ├── folding_config_baseline.json
│   │   ├── folding_config_best_<strategy>.json
│   │   ├── <model>_feasibility_check.json        # Folding configs passed to FINN
│   │   ├── <strategy>_<model>_config_0000.json   #   (single-model run)
│   │   ├── <strategy>_J<hash>_<model>_config_0000.json  # (multi-model run)
│   │   └── <strategy>/              # Per-model search artifacts
│   │       ├── config_0000/         # Per-trial dir (single-model run only)
│   │       │   ├── folding_config.json
│   │       │   ├── trial_params.json
│   │       │   └── build.log
│   │       ├── trial_results.csv
│   │       ├── pruned_space.json
│   │       ├── llm_reasoning_history.txt
│   │       └── optuna_study.db      # (single-model run only)
│   └── JOINT__<modelA>+<modelB>/
│       └── <strategy>/              # Multi-model search artifacts
│           ├── config_0000/<model>/ # Per-trial, per-model dir
│           └── optuna_study.db
└── results_synth/
    └── <model>/<config_stem>/        # FINN reports per trial
```

`J<hash>` is a short hash of the model combination. Trial numbers are counted
separately for each combination, so the tag keeps one combination's configs and
FINN results from overwriting another's.

A fresh clone has no `dataset/`; create it with `generate_folding_config.py` or
copy ONNX files and baselines into the layout above.

---

## Prerequisites and environment setup

The full workflow runs inside the **FINN Singularity container** with Vivado and
Vitis sourced. Outside the container the scripts cannot import FINN/QONNX/
Brevitas or shell out to FINN builds.

Start the container from your home directory (adjust `FINN_ROOT` to point to
your FINN clone and the `.sif` path to your container image):

```sh
export FINN_XILINX_PATH=/opt/AMD
export FINN_XILINX_VERSION=2025.2
export FINN_ROOT=$(pwd)/finn
export FINN_SINGULARITY=/space/finn/v0.10.1-6-g8ac41e46-dirty.xrt_202220.2.14.354_22.04-amd64-xrt.sif
export FINN_DOCKER_EXTRA="-v /usr/lib/x86_64-linux-gnu/libpixman-1.so.0:/usr/lib/x86_64-linux-gnu/libpixman-1.so.0 -v /opt/AMD:/opt/AMD"

./finn/run-docker.sh
```

Inside the container, set up FINN and the Xilinx tools:

```sh
export PYTHONPATH=$FINN_ROOT/src:$FINN_ROOT/deps/qonnx/src:$FINN_ROOT/deps/brevitas/src:$FINN_ROOT/deps/pyverilator:$PYTHONPATH

export VIVADO_PATH=/opt/AMD/2025.2/Vivado
export VITIS_PATH=/opt/AMD/2025.2/Vitis
export CCACHE_DIR=/tmp/.ccache

source /opt/AMD/2025.2/Vivado/settings64.sh
source /opt/AMD/2025.2/Vitis/settings64.sh
```

Then, from this repository, install the Python dependencies into a local
directory (the container's default Python paths are read-only):

```sh
mkdir -p ./.python_libs
export PYTHONUSERBASE=$(pwd)/.python_libs
pip install --user -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)/.python_libs/lib/python3.10/site-packages
```

Run all commands **from the repository root** — runtime paths are relative to
it.

---

## Quick start

### 1. Generate a baseline folding config

From a pre-exported ONNX file (the ONNX is copied into `dataset/<stem>/`):

```sh
python generate_folding_config.py \
  --onnx_file models/cnv_w1a1.onnx \
  --config    build_configs/cnv_w1a1_build_config.json
```

From a Brevitas source model (exported to ONNX first):

```sh
python generate_folding_config.py \
  --model_file  models/cnv.py \
  --model_class CNV \
  --config      build_configs/cnv_w2a2_build_config.json
```

Useful options:

- `--output_cfg <path>` — override the output path for the baseline JSON.
- `--restype_lut` — force `resType: lut` on MVAU layers to save DSPs.
- `--verbose` — debug logging.

If `--config` is omitted, the generator uses built-in defaults and writes
`build_config.template.json` only if that file does not already exist.

### 2. Tuning with LLM pruning

The AI tuner handles one or more models; pass a single name to `--model` for
single-model tuning. It derives every path from `--model <name>`:

```
build_configs/<name>_build_config.json
custom_builds/<name>_build.py
dataset/<name>/<name>.onnx
dataset/config_files/<name>/folding_config_baseline.json
```

```sh
python finn_raytune_optimizer_ai.py \
  --model cnv_w1a1 mlp_w2a2 \
  --board Zynq-7015 \
  --epoch_schedule 50,100,200 \
  --objective balanced \
  --search_strategy optuna \
  --prune_on_start \
  --extra_build_args --estimate-only
```

| Option | Description |
| --- | --- |
| `--model` | One or more model names sharing the same board. Repeating a name deploys multiple instances but builds it once. |
| `--board` | Target board from `helpers/boards.py` (default `Zynq-7015`). Sets both the resource budget and the `--fpga-part` passed to the build scripts. |
| `--epoch_schedule` | Comma-separated trials per pruning round, e.g. `50,100,200`. |
| `--objective` | `throughput`, `resource_avg`, or `balanced`. |
| `--search_strategy` | `optuna`, `random`, `nevergrad`, `nevergrad_de`, `hyperopt`. |
| `--max_concurrent` | Concurrent trials (default `1`). Each trial requests 2 CPUs per distinct model, capped at the CPUs available to Ray. |
| `--build_timeout` | Per-trial timeout in seconds (default `7200`). |
| `--seed` | Random seed (default `20`), offset by the round number. Seeds Optuna (TPE sampler), HyperOpt and random search; Nevergrad is not seeded. |
| `--raytune_dir` | Ray Tune output directory (default `./raytune_logs`). |
| `--ray_address` | Connect to an existing Ray cluster. |
| `--constraints` | Natural-language goal passed to the LLM pruner. |
| `--prune_on_start` | Run the pruner once before the first round. |
| `--api_key` | API key forwarded to the pruner (despite the name, this is the **Cerebras** key, unless the provider is changed inside the script). |
| `--extra_build_args` | Verbatim arguments forwarded to the build scripts. **Must be last.** Passing `--fpga-part <part>` here overrides the part chosen from `--board`. |

On a fresh experiment (when `trial_results.csv` does not yet exist), the AI
tuner runs an **all-PE/SIMD=1 feasibility build** for each model and aborts the
whole run if the model does not fit the board.

Each round starts a **fresh search** over that round's pruned space (for
Optuna, a new study per round inside the same `optuna_study.db`). It is
warm-started with up to 5 of the best past configurations that are still legal
in the new space, paired by rank across models. A pairing is skipped when the
models' combined LUT/DSP/BRAM/FF, counting duplicate instances, exceeds the
board.

After each round (except the last), the tuner calls the LLM pruner with the
round's `trial_results.csv` and writes a new `pruned_space.json`. If a round
produces at most one successful build, a guardrail clamps PE/SIMD candidates to
`<= 8`.

In `trial_results.csv`, `build_success = 1` means the FINN build succeeded
**and** the model fits the board on its own. Over-capacity builds are recorded
as failures with the `10000` penalty objective, so the LLM's failure rate, the
guardrail, and warm-starting all treat them as failed.

### 3. Standalone LLM pruning

```sh
python space_pruner.py \
  --build_cfg    build_configs/cnv_w1a1_build_config.json \
  --baseline_cfg dataset/config_files/cnv_w1a1/folding_config_baseline.json \
  --out_json     /tmp/pruned_space.json \
  --constraints  "maximize throughput" \
  --pruning_round 1
```

Optional: `--results_csv <csv>` lets the LLM reason about past trials, and
`--model <name>` selects the LLM (default `qwen-3.8-27b`).

The pruner needs a Cerebras API key (or another API key if the provider is
changed inside the script), supplied via `--api_key` or the
`CEREBRAS_API_KEY` environment variable (a local `.env` is loaded when
`python-dotenv` is installed). It reads layer topology and the build config,
computes all valid `PE`/`SIMD` divisors, asks the LLM to select a subset, and
validates the answer against the mathematical bounds before writing
`pruned_space.json`.

---

## Build configuration

`build_configs/<model>_build_config.json` drives baseline generation:

```json
{
  "output_onnx": "dataset/cnv_w1a1/cnv_w1a1.onnx",
  "output_cfg":  "dataset/config_files/cnv_w1a1/folding_config_baseline.json",
  "opset": 13,
  "input_shape": [1, 3, 32, 32],
  "board": "Zynq-7000",
  "fpga_part": "xc7z020clg400-1",
  "clock_ns": 10.0,
  "target_fps": null,
  "model_kwargs": {
    "num_classes": 10,
    "weight_bit_width": 1,
    "act_bit_width": 1,
    "inp_bit_width": 8,
    "in_channels": 3
  }
}
```

| Key | Meaning |
| --- | --- |
| `output_onnx` | Destination ONNX path (`dataset/<model>/<model>.onnx`). |
| `output_cfg` | Destination baseline folding config. |
| `opset` | ONNX opset used during export. |
| `input_shape` | Model input tensor shape. |
| `board`, `fpga_part`, `clock_ns`, `target_fps` | Baseline target metadata written by the generator; the actual build part comes from `--board` (see below). |
| `model_kwargs` | Constructor arguments forwarded to the model class (ignored for `--onnx_file`). |
| `build_script`, `baseline_timeout` | Legacy keys; the tuner uses `custom_builds/*_build.py` instead. |

`custom_builds/<model>_build.py` accepts:

```sh
python custom_builds/<model>_build.py <model_dir> \
  --model-name <model.onnx> \
  --folding-config <config.json> \
  [--fpga-part <xilinx_part>] \
  --estimate-only
```

The positional directory and model name are relative to `dataset/`. The script
writes `status.json` and collects FINN reports into
`dataset/results_synth/<model>/<config_stem>/`. `--estimate-only` stops after
analytical estimation, skipping Vivado codegen and synthesis — use it for fast
DSE iterations.

---

## Objectives

| Objective | Direction | Meaning |
| --- | --- | --- |
| `throughput` | minimize | Negated throughput in images/s. |
| `balanced` | minimize | Negated total throughput plus a weighted resource sum. |
| `resource_avg` | minimize | Average LUT/DSP/BRAM/FF utilization. |

In every case, configurations exceeding board capacity (per model, or summed
across all models in a joint run) receive a penalty objective of `10000`, as do
failed builds.

---

## Supported boards and models

Resource budgets available to the AI tuner via `--board` (`helpers/boards.py`):

`XC7S25`, `XC7S25-65`, `Zynq-7010`, `Zynq-7015`, `Zynq-7020`, `Zynq-7030`,
`ZU3EG`, `U250`.

`--board` selects both the resource budget used to score trials and the synthesis
target: the tuner passes the matching part from `BOARD_PARTS` in
`helpers/boards.py` to the build script as `--fpga-part` (edit that table if your
board uses a different package or speed grade, or pass `--fpga-part` yourself via
`--extra_build_args`). The clock is still hardcoded in each
`custom_builds/*_build.py`. The `board` key in
`build_configs/*.json` is baseline metadata and is not validated against this
list.

`XC7S25-65` is a reduced `XC7S25` budget using 65% of the total resources.

Models with an ONNX shipped in `models/`:

| Model | ONNX | Build config | Build script |
| --- | --- | --- | --- |
| `MLP_W3A3` | yes | yes | yes |
| `cnv_w1a1` | yes | yes | yes |
| `mlp_w2a2` | yes | yes | yes |
| `resnet50` | yes | yes | yes |
| `cnv_w2a2` | no — export from `models/cnv.py` | yes | yes |
| `resnet18` | yes | yes | yes |

To generate `cnv_w2a2`, run the baseline generator with `--model_file
models/cnv.py --model_class CNV --config
build_configs/cnv_w2a2_build_config.json`.

---

## Results and artifacts

- `dataset/config_files/<model>/folding_config_best_<strategy>.json` — best
  configuration found by the tuner.
- `dataset/config_files/<model>/<strategy>/` — per-trial configs, trial params,
  build logs, CSV history, Optuna study database, pruned spaces, and LLM
  reasoning history.
- `dataset/results_synth/<model>/<config_stem>/` — FINN reports consumed by
  `helpers/metrics.py`.
- `raytune_logs/` — Ray Tune experiment output.

The joint tuner writes final best configs for each model to its own
`folding_config_best_<strategy>.json` under
`dataset/config_files/<model>/`. The best config is the lowest-objective
**feasible** trial across all rounds. If no trial fits the board, nothing is
saved and a warning is logged.

After each round, the tuner prints a summary table with the objective, success
flag, combined throughput and resource percentages, and per-model throughput.

---

## Troubleshooting

- **`--extra_build_args` ignored** — it uses `argparse.REMAINDER`; place it last.
- **AI tuner exits immediately** — verify all four per-model paths derived from
  `--model` exist (config, build script, ONNX, baseline). It does not accept
  explicit paths.
- **Feasibility check fails** — the model does not fit the selected board even
  at `PE=1, SIMD=1`; choose a larger board or reduce the model.
- **"No API key found"** — export `CEREBRAS_API_KEY` or pass `--api_key`. Install `openai`, `certifi`, and optionally
  `python-dotenv`.
- **Pre-expanded baseline rejected** — feed the tuner logical baselines
  (`MVAU_0`), not configs that already contain `MVAU_hls_0` / `MVAU_rtl_0`;
  `helpers/folding.py` performs the expansion.
- **No valid search space** — the baseline lacks `MH`/`MW`/`IFMChannels`
  metadata. Regenerate it with `generate_folding_config.py`.
- **Slower-than-expected runs** — start with `--extra_build_args
  --estimate-only`; omit it only for full Vivado synthesis.
- **Wrong part or package in synthesis** — edit `BOARD_PARTS` in
  `helpers/boards.py`, or pass `--extra_build_args --fpga-part <part>`.
- **"CSV has no columns for …" warning** — `trial_results.csv` was created with
  a different parameter set (e.g. after regenerating the baseline). Rows are
  appended under the existing header, so start a fresh CSV for the new space.
  CSVs written before the column-alignment fix may hold values under the wrong
  headers; delete them.
- **Fewer CPUs than requested** — the tuner logs a warning and lowers the
  per-trial CPU request to what Ray reports as available.

There is no test suite, CI, lint, or formatter configuration. Verify changes
with `--help`, a baseline generation run, and a FINN build (`--estimate-only`
for quick checks).
