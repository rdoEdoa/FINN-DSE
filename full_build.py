import argparse
import os
import shutil
import json
import numpy as np
import sys
import glob
import getpass

from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
from finn.builder.build_dataflow_steps import build_dataflow_step_lookup
from qonnx.core.modelwrapper import ModelWrapper
from finn.builder.build_dataflow_steps import *

from qonnx.core.datatype import DataType
import warnings
warnings.filterwarnings("ignore")

def thresholds_round(model):
    """Convert floating point thresholds to INT32."""
    for node in model.graph.node:
        if node.op_type == "MultiThreshold":
            thresh_name = node.input[1]
            thresholds = model.get_initializer(thresh_name)
            if thresholds is not None and thresholds.dtype != np.int32:
                thresholds_int = np.round(thresholds).astype(np.int32)
                model.set_initializer(thresh_name, thresholds_int)
                model.set_tensor_datatype(thresh_name, DataType["INT32"])
    return model

def get_build_steps(estimate_only=False):
    """Gives the list of build steps to be executed"""
    steps = [
        "step_qonnx_to_finn",
        "step_tidy_up",
        "step_streamline",
        "step_convert_to_hw",
        "step_create_dataflow_partition",
        "step_specialize_layers",
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports", 
    ]
    if not estimate_only:
        steps.extend([
            "step_hw_codegen",
            "step_hw_ipgen",
            "step_set_fifo_depths",
            "step_create_stitched_ip",
            "step_measure_rtlsim_performance",
            "step_out_of_context_synthesis",
        ])
    return steps

def create_build_config(output_dir, folding_config_path, estimate_only=False):
    """Creates and configures the DataflowBuildConfig"""
    outputs = [DataflowOutputType.ESTIMATE_REPORTS]
    if not estimate_only:
        outputs.extend([
            DataflowOutputType.RTLSIM_PERFORMANCE,
            DataflowOutputType.OOC_SYNTH, 
            DataflowOutputType.STITCHED_IP
        ])

    cfg = DataflowBuildConfig(
        output_dir=output_dir,
        generate_outputs=outputs,
        synth_clk_period_ns=10.0,
        hls_clk_period_ns=10.0,
        folding_config_file=folding_config_path,
        auto_fifo_depths=False,
        fpga_part="xc7z020clg400-1",
        mvau_wwidth_max=256,
        split_large_fifos=True,
        enable_build_pdb_debug=True,
        standalone_thresholds=True,
        save_intermediate_models=False
    )
    return cfg

def execute_build_steps(model, cfg, build_steps, output_dir, verbose=False):
    """Execute the build steps and write a status receipt."""
    step_lookup = build_dataflow_step_lookup.copy()
    status_log = {"success": 0, "last_step_executed": "None", "error_message": "None"}
    os.makedirs(output_dir, exist_ok=True)
    status_path = os.path.join(output_dir, "status.json")

    for i, step_name in enumerate(build_steps):
        if verbose:
            print(f"Running step: {step_name} [{i+1}/{len(build_steps)}]")
        try:
            if step_name == "step_convert_to_hw":
                model = thresholds_round(model)
            step_function = step_lookup[step_name]
            model = step_function(model, cfg)

            if step_name == "step_convert_to_hw":
                hw_model_path = os.path.join(output_dir, "model_after_hw_conversion.onnx")
                model.save(hw_model_path)
                if verbose:
                    print(f"  Saved post-HW-conversion model to {hw_model_path}")
            status_log["last_step_executed"] = step_name

        except Exception as e:
            print(f"Error during the execution of the step '{step_name}': {e}")
            status_log["error_message"] = str(e)
            with open(status_path, "w") as f:
                json.dump(status_log, f, indent=4)
            return None

    status_log["success"] = 1
    status_log["error_message"] = "None"
    with open(status_path, "w") as f:
        json.dump(status_log, f, indent=4)
    return model

def collect_reports_and_cleanup(output_dir):
    """Gathers all reports to the root of output_dir and deletes the heavy Vivado/IP folders"""
    print("\nCollecting reports and cleaning up project files...")
    reports_dir = os.path.join(output_dir, "final_reports")
    os.makedirs(reports_dir, exist_ok=True)
    exts = ["json", "txt", "log", "csv", "rpt", "onnx"]
    for ext in exts:
        pattern = os.path.join(output_dir, "**", f"*.{ext}") 
        for f in glob.glob(pattern, recursive=True):
            try:
                if "final_reports" not in f: 
                    shutil.copy(f, reports_dir)
            except Exception as e:
                print(f"Warning: could not copy {f}: {e}")
                
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if item_path != reports_dir:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
                
    for report_file in os.listdir(reports_dir):
        source = os.path.join(reports_dir, report_file)
        destination = os.path.join(output_dir, report_file)
        shutil.move(source, destination)
        
    os.rmdir(reports_dir)
    print(f"All FINN synthesis reports collected in: {output_dir}")

def clean_finn_internal_tmp(target_dir):
    """Safely wipes the specific isolated FINN temporary folder for this build."""
    my_uid = os.getuid()
    print(f"\nSafely cleaning up isolated temporary files in {target_dir}...")
    if os.path.exists(target_dir):
        if os.stat(target_dir).st_uid == my_uid:
            try: shutil.rmtree(target_dir)
            except Exception as e: print(f"Warning: Could not remove {target_dir}: {e}")
        else:
            print(f"Safety abort: {target_dir} is not owned by your user ID")
    else:
        print("No temporary FINN directory found. Skipping cleanup.")

def main():
    parser = argparse.ArgumentParser(
        description='Execute the FINN synthesis process on a specified ONNX model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('directory', help='Directory containing the ONNX model (e.g., onnx_models)')
    parser.add_argument('--model-name', '-m', default='model.onnx', help='Name of the ONNX model file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Detailed output log')
    parser.add_argument('--folding-config', '-fc', default='config_00000.json', help='Set the folding configuration file name')
    parser.add_argument('--estimate-only', action='store_true', help='Run only up to analytical estimations, skip Vivado synthesis')
    args = parser.parse_args()
    
    model_directory = os.path.join("dataset", args.directory)
    if not os.path.exists(model_directory):
        print(f"Error: The model directory '{model_directory}' does not exist")
        sys.exit(1)
    
    model_file = os.path.join("dataset", args.directory, args.model_name)
    if not os.path.exists(model_file):
        print(f"Error: The model '{model_file}' does not exist")
        sys.exit(1)
        
    model_base_name = os.path.splitext(args.model_name)[0]
    config_base_name = os.path.splitext(args.folding_config)[0]
    output_dir = os.path.join("dataset", "results_synth", model_base_name, config_base_name)
    folding_config_path = os.path.join("dataset", "config_files", model_base_name, args.folding_config)
    
    username = getpass.getuser()
    unique_tmp_dir = f"/tmp/finn_dev_{username}_{config_base_name}"
    os.environ["FINN_BUILD_DIR"] = unique_tmp_dir
    os.makedirs(unique_tmp_dir, exist_ok=True)
    
    if args.verbose:
        print("-" * 50)
        print(f"Working directory: {model_directory}")
        print(f"Model: {model_file}")
        print(f"Output directory: {output_dir}")
        print(f"Folding configuration file: {folding_config_path}")
        print("-" * 50)
    
    try:
        print("Loading the model...")
        model = ModelWrapper(model_file)
        print("Create build configuration...")
        cfg = create_build_config(output_dir, folding_config_path, estimate_only=args.estimate_only)
        build_steps = get_build_steps(estimate_only=args.estimate_only)
        print("=" * 50)
        
        result_model = execute_build_steps(model, cfg, build_steps, output_dir, args.verbose)
        
        if result_model is not None:
            print("=" * 50)
            print("Synthesis/Estimation executed successfully!")
        else:
            print("Build failed!")
            collect_reports_and_cleanup(output_dir)
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        collect_reports_and_cleanup(output_dir)
        clean_finn_internal_tmp(unique_tmp_dir)
        
    return 0

if __name__ == "__main__":
    main()