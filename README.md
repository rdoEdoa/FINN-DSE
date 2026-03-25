# Workflow

## Prerequisites

The full workflow will take place inside a working FINN container. Make sure it is available before starting.

Then, install the require packages:

```sh
pip install -r requirements.txt
```

## Generate the baseline folding configuration file

```sh
python generate_folding_config.py \
    --model_file  lenet5_quantized.py \
    --model_class LeNet5Quantized     \
    --config      build_config.json
```

## Run the RayTune optimization

```sh
python finn_raytune_optimizer.py \
    --baseline_cfg      dataset/config_files/lenet5/folding_config_baseline.json \
    --build_script      full_build.py \
    --onnx_path         dataset/lenet5/lenet5.onnx \
    --num_samples       25 \
    --search_strategy   nevergrad \
    --objective         resource_avg
```

**Practical notes:**

- `--num_samples 10` is a reasonable starting point; each trial takes as long as one full FINN build
- `lut_slack` minimises LUT usage while penalising any timing failure (WNS < 0); switch to throughput if throughput is your primary goal
- RayTune prints a live table every 60 seconds showing objective, lut, dsp, wns, and build_success per trial

## Appendix - RayTune in already built Singularity container
If the process needs to be done in a singularity container that does not already have the needed packages, it not possible to simply install them with `pip`, as it tries to install them in read-only folders. Therefore, it is necessary to follow these steps:

1) Create a folder in the current directory for the libraries.

```sh
mkdir -p ./python_libs
```

2) Point Python to that folder and install there.

```sh
export PYTHONUSERBASE=$(pwd)/python_libs
pip install --user -r requirements.txt
```

3) Every time the script needs to be run, it is necessary yo tun the following command, so that Python can find the packages, paying attention that the actual correct path is being set.

```sh
export PYTHONPATH=$PYTHONPATH:$(pwd)/python_libs/lib/python3.10/site-packages
```