# Workflow

## Prerequisites

The full workflow will take place inside a working FINN container. Make sure it is available before starting.

Setup some environment variables and start the singularity container. 
**NOTE:** This snippet is done in the home directory, assuming to have cloned there the finn repository (therefore NOT in this repository folder); adjust the FINN_ROOT variable accordingly and properly launch the container after all the exports.

```sh
export FINN_XILINX_PATH=/opt/AMD
export FINN_XILINX_VERSION=2025.2
export FINN_ROOT=${pwd}/finn
export FINN_SINGULARITY=/space/finn/v0.10.1-6-g8ac41e46-dirty.xrt_202220.2.14.354_22.04-amd64-xrt.sif
export FINN_DOCKER_EXTRA="-v /usr/lib/x86_64-linux-gnu/libpixman-1.so.0:/usr/lib/x86_64-linux-gnu/libpixman-1.so.0 -v /opt/AMD:/opt/AMD"

./finn/run-docker.sh
```

Once inside the container, setup other environment variables and the needed tools by running:

```sh
export PYTHONPATH=$FINN_ROOT/src:$FINN_ROOT/deps/qonnx/src:$FINN_ROOT/deps/brevitas/src:$FINN_ROOT/deps/pyverilator:$PYTHONPATH

export VIVADO_PATH=/opt/AMD/2025.2/Vivado
export VITIS_PATH=/opt/AMD/2025.2/Vitis

export CCACHE_DIR=/tmp/.ccache

source /opt/AMD/2025.2/Vivado/settings64.sh
source /opt/AMD/2025.2/Vitis/settings64.sh
```
Eventual warning messages can be safely ignored.

Then, enter this repository folder and install the required packages; it is not possible to do it in the default folders as singularity sees them as read-only, therefore it is necessary to create a new folder and tell Python to use that:

```sh
mkdir -p ./.python_libs

export PYTHONUSERBASE=$(pwd)/.python_libs
pip install --user -r requirements.txt

export PYTHONPATH=$PYTHONPATH:$(pwd)/.python_libs/lib/python3.10/site-packages

```

At this point, the environment is ready.

## Generate the baseline folding configuration file
First, it is necessary to generate the baseline configuration file that also contains crucial informations to generate the proper search space. To do so, run the `generate_folding_config.py` script; this script also uses informations store in the `build_config.json` file, and so it needs to be modified according to the model under test; the current file is set to work with the provided model. The following snippet works with the network provided.

```sh
python generate_folding_config.py \
    --model_file  lenet5_quantized.py \
    --model_class LeNet5Quantized     \
    --config      build_config.json
```

## Run the RayTune optimization
Once the baseline configuration file is ready, it is possible to run the optimization file that uses RayTune. The following snippet works with the provided network. It is possible to change the number of trials (`num_samples`), the search strategy (up to now, it is possible to use `optuna`, `nevergrad`, `random`) and the objective (`throughput`, `resource_avg`, `balanced`).

```sh
python finn_raytune_optimizer.py \
    --baseline_cfg      dataset/config_files/lenet5/folding_config_baseline.json \
    --build_script      full_build.py \
    --onnx_path         dataset/lenet5/lenet5.onnx \
    --num_samples       25 \
    --search_strategy   optuna \
    --objective         throughput
```