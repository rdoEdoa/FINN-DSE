# FINN-DSE — Design Space Exploration of FINN Folding Parameters

Automated exploration of the **PE/SIMD folding space** for quantized neural
networks deployed with [FINN](https://github.com/Xilinx/finn). Given an ONNX
model and a baseline folding configuration, the framework searches for folding
parameters that maximize throughput, minimize resource usage, or balance both —
optionally pruning the space with an LLM between search rounds.

The toolchain has four entry points:

| Script | Purpose |
| --- | --- |
| `generate_folding_config.py` | Export a Brevitas/PyTorch model (or ingest an ONNX) and generate a baseline `PE=1, SIMD=1` folding config. |
| `finn_raytune_optimizer.py` | Single-model PE/SIMD search with Ray Tune (Optuna, random, Nevergrad, HyperOpt). |
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
          folding_config_best_<strategy>.json              LLM pruning rounds
                                                          (AI tuner only)
```

1. **Baseline generation** parses the ONNX weight tensors to recover each
   compute layer's `MH` (output dim) and `MW` (input dim), then writes a config
   with `PE = 1` and `SIMD = 1` for every layer. Layer names are normalized to
   the logical form `MVAU_0`, `MVAU_1`, … which the tuners later expand into the
   `MVAU_hls_0` / `MVAU_rtl_0` pairs FINN expects.
2. **Search space** contains one `PE` and one `SIMD` hyperparameter per logical
   layer. Only divisors of `MH` / `MW` are offered, so FINN never receives an
   illegal value.
3. **Each trial** patches the baseline, writes a folding config, invokes a build
   script as a subprocess, and reads the FINN JSON reports (`status.json`,
   `ooc_synth_and_timing.json`, `rtlsim_performance.json`, or the estimate
   reports) into a scalar objective.
4. **The best configuration** is re-expanded to `_hls_` / `_rtl_` names and
   saved next to the baseline.

---

## Repository layout

```
.
├── generate_folding_config.py       # Baseline folding-config generator
├── finn_raytune_optimizer.py        # Single-model Ray Tune tuner
├── finn_raytune_optimizer_ai.py     # Multi-model tuner + LLM pruning
├── space_pruner.py                  # LLM design-space pruner (Cerebras)
├── build_config.template.json       # Template for model build configs
├── build_configs/                   # Per-model build configuration
│   └── <model>_build_config.json
├── custom_builds/                   # Per-model FINN build scripts
│   └── <model>_build.py
├── helpers/
│   ├── boards.py                    # FPGA resource budgets
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
│   │   └── <strategy>/              # Per-search artifacts (single model)
│   │       ├── config_0000/
│   │       │   ├── folding_config.json
│   │       │   ├── trial_params.json
│   │       │   └── build.log
│   │       ├── <strategy>_config_0000.json
│   │       ├── trial_results.csv    # AI tuner only
│   │       ├── pruned_space.json    # AI tuner only
│   │       ├── llm_reasoning_history.txt
│   │       └── optuna_study.db
│   └── JOINT__<modelA>+<modelB>/
│       └── <strategy>/              # Joint search artifacts
└── results_synth/
    └── <model>/<config_stem>/        # FINN reports per trial
```

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

The LLM pruner needs three additional packages that are not in
`requirements.txt`:

```sh
pip install --user openai certifi python-dotenv
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

If `--config` is omitted, the generator writes `build_config.template.json` and
uses built-in defaults.

### 2. Single-model tuning

```sh
python finn_raytune_optimizer.py \
  --baseline_cfg  dataset/config_files/cnv_w1a1/folding_config_baseline.json \
  --build_script  custom_builds/cnv_w1a1_build.py \
  --onnx_path     dataset/cnv_w1a1/cnv_w1a1.onnx \
  --num_samples   40 \
  --objective     lut_slack \
  --search_strategy optuna \
  --extra_build_args --estimate-only
```

| Option | Description |
| --- | --- |
| `--baseline_cfg` | Baseline JSON from `generate_folding_config.py` (required). |
| `--build_script` | Per-model FINN build script (required). |
| `--onnx_path` | Exported ONNX file (required). |
| `--best_cfg_out` | Best-config output. Defaults to `dataset/config_files/<model>/folding_config_best_<strategy>.json`. |
| `--num_samples` | Number of trials (default `100`). |
| `--max_concurrent` | Concurrent builds (default `1`; raise only with enough Vivado licences). |
| `--objective` | `lut`, `lut_slack`, `throughput`, or `balanced`. |
| `--search_strategy` | `optuna`, `random`, `nevergrad`, `nevergrad_de`, `hyperopt`. |
| `--build_timeout` | Per-trial timeout in seconds (default `7200`). |
| `--parallelism_threshold` | Skip builds whose summed PE+SIMD exceeds this value. |
| `--database_dir` | Folder of pre-built configs to reuse instead of rebuilding. |
| `--ray_address` | Connect to an existing Ray cluster. |
| `--extra_build_args` | Verbatim arguments forwarded to the build script. **Must be last.** |

> The single-model tuner evaluates objectives against a hardcoded **U250**
> resource budget and ignores the board in `build_configs/`.

### 3. Multi-model tuning with LLM pruning

The AI tuner derives every path from `--model <name>`:

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
| `--board` | Target board from `helpers/boards.py` (default `Zynq-7015`). |
| `--epoch_schedule` | Comma-separated trials per pruning round, e.g. `50,100,200`. |
| `--objective` | `throughput`, `resource_avg`, or `balanced`. |
| `--search_strategy` | `optuna`, `random`, `nevergrad`, `nevergrad_de`, `hyperopt`. |
| `--constraints` | Natural-language goal passed to the LLM pruner. |
| `--prune_on_start` | Run the pruner once before the first round. |
| `--groq_api_key` | API key forwarded to the pruner (despite the name, this is the **Cerebras** key). |
| `--parallelism_threshold` | Skip builds above this summed PE+SIMD (default `1000000`). |
| `--extra_build_args` | Verbatim arguments forwarded to the build scripts. **Must be last.** |

Before the first round, the AI tuner runs an **all-PE/SIMD=1 feasibility build**
for each model and aborts the whole run if the model does not fit the board.

After each round (except the last), the tuner calls the LLM pruner with the
round's `trial_results.csv` and writes a new `pruned_space.json`. If a round
produces at most one successful build, a guardrail clamps PE/SIMD candidates to
`<= 8`.

### 4. Standalone LLM pruning

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

The pruner needs a Cerebras API key, supplied via `--api_key` or the
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
    "weight_bit_width": 2,
    "act_bit_width": 2,
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
| `board`, `fpga_part`, `clock_ns`, `target_fps` | FINN/Vivado target settings. |
| `model_kwargs` | Constructor arguments forwarded to the model class (ignored for `--onnx_file`). |
| `build_script`, `baseline_timeout` | Legacy keys; the tuners use `custom_builds/*_build.py` instead. |

`custom_builds/<model>_build.py` accepts:

```sh
python custom_builds/<model>_build.py <model_dir> \
  --model-name <model.onnx> \
  --folding-config <config.json> \
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
| `lut` | minimize | Normalized LUT usage (single-model only). |
| `lut_slack` | minimize | LUT usage combined with timing slack (single-model only). |
| `throughput` | maximize | Estimated/simulated throughput in images/s. |
| `balanced` | minimize | Single-model: bottleneck utilization divided by throughput. AI tuner: negated total throughput plus a weighted resource sum. |
| `resource_avg` | minimize | Average LUT/DSP/BRAM/FF utilization (AI tuner only). |

In every case, configurations exceeding board capacity (or 85 % utilization)
receive a penalty, and failed builds receive a large sentinel value.

---

## Supported boards and models

Boards available to the AI tuner via `--board` (`helpers/boards.py`):

`XC7S25`, `XC7S25-65`, `Zynq-7010`, `Zynq-7015`, `Zynq-7020`, `Zynq-7030`,
`ZU3EG`, `U250`.

Models with an ONNX shipped in `models/`:

| Model | ONNX | Build config | Build script |
| --- | --- | --- | --- |
| `MLP_W3A3` | yes | yes | yes |
| `cnv_w1a1` | yes | yes | yes |
| `mlp_w2a2` | yes | yes | yes |
| `resnet50` | yes | yes | yes |
| `cnv_w2a2` | no — export from `models/cnv.py` | yes | yes |
| `resnet18` | no — needs an external ONNX | yes | yes |

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
`dataset/config_files/<model>/`.

---

## Troubleshooting

- **`--extra_build_args` ignored** — it uses `argparse.REMAINDER`; place it last.
- **AI tuner exits immediately** — verify all four per-model paths derived from
  `--model` exist (config, build script, ONNX, baseline). It does not accept
  explicit paths.
- **Feasibility check fails** — the model does not fit the selected board even
  at `PE=1, SIMD=1`; choose a larger board or reduce the model.
- **"No API key found"** — export `CEREBRAS_API_KEY` or pass `--api_key` /
  `--groq_api_key`. Install `openai`, `certifi`, and optionally
  `python-dotenv`.
- **Pre-expanded baseline rejected** — feed the tuners logical baselines
  (`MVAU_0`), not configs that already contain `MVAU_hls_0` / `MVAU_rtl_0`;
  `helpers/folding.py` performs the expansion.
- **No valid search space** — the baseline lacks `MH`/`MW`/`IFMChannels`
  metadata. Regenerate it with `generate_folding_config.py`.
- **Resource budget confusion** — the single-model tuner is fixed to U250
  limits; use the AI tuner with `--board` for other targets.
- **Slower-than-expected runs** — start with `--extra_build_args
  --estimate-only`; omit it only for full Vivado synthesis.

There is no test suite, CI, lint, or formatter configuration. Verify changes
with `--help`, a baseline generation run, and a FINN build (`--estimate-only`
for quick checks).
