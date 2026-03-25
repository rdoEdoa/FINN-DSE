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

# Importing DataType to set the thresholds to INT32 
from qonnx.core.datatype import DataType

# Ignore warnings for cleaner output    
import warnings
warnings.filterwarnings("ignore")


# FINN (as FPGAs) uses integer thresholds, but sometimes PyTorch/Brevitas leave them as floating point
# Built-in functions not always work, therefore this function performs a rounding to the neares integer
def thresholds_round(model):
    """Convert floating point thresholds to INT32."""
    # Iterate through the nodes in the model graph
    for node in model.graph.node:
        # Check if the node is a MultiThreshold operation
        if node.op_type == "MultiThreshold":
            # Get the name of the thresholds initializer
            thresh_name = node.input[1]
            thresholds = model.get_initializer(thresh_name)
            
            # If thresholds are found and are not already INT32, round them and update the model
            if thresholds is not None and thresholds.dtype != np.int32:
                thresholds_int = np.round(thresholds).astype(np.int32) # Rounding
                model.set_initializer(thresh_name, thresholds_int)
                model.set_tensor_datatype(thresh_name, DataType["INT32"])
                    
    return model

# Define the sequence of build steps to execute
def get_build_steps():
    """Gives the list of build steps to be executed"""
    return [
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
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
        "step_measure_rtlsim_performance",
        "step_out_of_context_synthesis",
    ]

# FINN build configuration
# The used values are standard
def create_build_config(output_dir, folding_config_path):
    """Creates and configures the DataflowBuildConfig"""
    cfg = DataflowBuildConfig(
        output_dir=output_dir,
        generate_outputs=[
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.RTLSIM_PERFORMANCE,
            DataflowOutputType.OOC_SYNTH, 
            DataflowOutputType.STITCHED_IP
        ],
        synth_clk_period_ns=10.0,
        hls_clk_period_ns=10.0,
        folding_config_file=folding_config_path,
        auto_fifo_depths=False,
        # This FPGA is the Zynq-7000
        fpga_part="xc7z020clg400-1",
        mvau_wwidth_max=256,
        split_large_fifos=True,
        enable_build_pdb_debug=True,
        standalone_thresholds=True,
        save_intermediate_models=False # Keeps things clean
    )

    # TODO: is it nexessary to set/change this?    
    # cfg.mvau_optimization = "resource"
    return cfg

def execute_build_steps(model, cfg, build_steps, output_dir, verbose=False):
    """Execute the build steps and write a status receipt."""
    step_lookup = build_dataflow_step_lookup.copy()

    status_log = {
        "success": 0,
        "last_step_executed": "None",
        "error_message": "None"
    }

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

            # --- Save intermediate model after HW conversion ---
            # At this point ONNX nodes correspond 1-to-1 with FINN config layers
            if step_name == "step_convert_to_hw":
                hw_model_path = os.path.join(output_dir, "model_after_hw_conversion.onnx")
                model.save(hw_model_path)
                if verbose:
                    print(f"  Saved post-HW-conversion model to {hw_model_path}")
            # ----------------------------------------------------

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

# Clean up the output directory, keeping only the report and the status log
def collect_reports_and_cleanup(output_dir):
    """Gathers all reports to the root of output_dir and deletes the heavy Vivado/IP folders"""
    print("\nCollecting reports and cleaning up project files...")
    reports_dir = os.path.join(output_dir, "final_reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Extensions to save
    exts = ["json", "txt", "log", "csv", "rpt", "onnx"]
    
    # Recursively find and copy all report files
    for ext in exts:
        pattern = os.path.join(output_dir, "**", f"*.{ext}") 
        for f in glob.glob(pattern, recursive=True):
            try:
                # Avoid copying a file onto itself if it's already in final_reports
                if "final_reports" not in f: 
                    shutil.copy(f, reports_dir)
            except Exception as e:
                print(f"Warning: could not copy {f}: {e}")
                
    # Delete everything else
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if item_path != reports_dir:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
                
    # Move the collected reports
    for report_file in os.listdir(reports_dir):
        source = os.path.join(reports_dir, report_file)
        destination = os.path.join(output_dir, report_file)
        shutil.move(source, destination)
        
    os.rmdir(reports_dir)
    print(f"All FINN synthesis reports collected in: {output_dir}")

def clean_finn_internal_tmp():
    """Safely wipes FINN temporary files strictly owned by the current user."""
    # Get the exact username and physical Linux User ID
    username = getpass.getuser()
    my_uid = os.getuid()

    # Target only the specific FINN folder
    target_dir = "/tmp/finn_dev_" + username

    print("\nSafely cleaning up YOUR temporary files in " + target_dir + "...")

    if os.path.exists(target_dir):
        # Double-check the ownership of the base folder
        if os.stat(target_dir).st_uid == my_uid:
            for item in os.listdir(target_dir):
                item_path = os.path.join(target_dir, item)
                try:
                    # Triple-check ownership of the specific file/sub-folder
                    if os.stat(item_path).st_uid == my_uid:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                except Exception as e:
                    print("Warning: Could not remove " + item_path + ": " + str(e))
        else:
            print("Safety abort: " + target_dir + " is not owned by your user ID")
    else:
        print("No temporary FINN directory found for your user. Skipping cleanup.")

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Execute the FINN synthesis process on a specified ONNX model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Arguments
    parser.add_argument('directory', 
                        help='Directory containing the ONNX model (e.g., onnx_models)')
    parser.add_argument('--model-name', '-m',
                        default='model.onnx',
                        help='Name of the ONNX model file')
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Detailed output log')
    parser.add_argument('--folding-config', '-fc',
                        default='config_00000.json',
                        help='Set the folding configuration file name')
    
    args = parser.parse_args()
    
    # Prepare the directories
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
        cfg = create_build_config(output_dir, folding_config_path)
        
        build_steps = get_build_steps()
        
        print("=" * 50)
        
        result_model = execute_build_steps(model, cfg, build_steps, output_dir, args.verbose)
        
        if result_model is not None:
            print("=" * 50)
            print("Synthesis executed successfully!")
        else:
            print("Build failed!")
            # Even if the build fails, clean the output directory
            collect_reports_and_cleanup(output_dir)
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # At the end of the process, gather all reports and clean up
    finally:
        collect_reports_and_cleanup(output_dir)
        clean_finn_internal_tmp()
        
    return 0

if __name__ == "__main__":
    main()