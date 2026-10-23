import argparse
import os
import re
import shutil
import json
import sys
import glob
import getpass
import warnings

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

warnings.filterwarnings("ignore")


def create_build_config(output_dir, folding_config_path, estimate_only=False):
    """Creates the DataflowBuildConfig (default FINN flow, custom folding)."""
    outputs = [build_cfg.DataflowOutputType.ESTIMATE_REPORTS]
    if not estimate_only:
        outputs.extend([
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            build_cfg.DataflowOutputType.OOC_SYNTH,
            build_cfg.DataflowOutputType.STITCHED_IP,
        ])

    kwargs = dict(
        output_dir=output_dir,
        generate_outputs=outputs,
        synth_clk_period_ns=10.0,
        hls_clk_period_ns=10.0,
        folding_config_file=folding_config_path,
        auto_fifo_depths=False,
        fpga_part="xc7z020clg400-1",
        mvau_wwidth_max=256,
        split_large_fifos=True,
        # Must be False for unattended runs (e.g. Optuna): True opens pdb on errors
        enable_build_pdb_debug=False,
        save_intermediate_models=False,
    )
    if estimate_only:
        # Same step list as the official estimate-only example
        kwargs["steps"] = build_cfg.estimate_only_dataflow_steps
    # Otherwise steps=None -> FINN default pipeline, filtered by generate_outputs
    return build_cfg.DataflowBuildConfig(**kwargs)


def parse_build_log(output_dir):
    """Best-effort extraction of last step and error info from FINN's log."""
    last_step, error = "None", "None"
    log_path = os.path.join(output_dir, "build_dataflow.log")
    if os.path.exists(log_path):
        with open(log_path, errors="ignore") as f:
            text = f.read()
        steps = re.findall(r"Running step: (\w+)", text)
        if steps:
            last_step = steps[-1]
        tail = [l for l in text.strip().splitlines() if l.strip()][-5:]
        error = " | ".join(tail) if tail else "None"
    return last_step, error


def write_status(output_dir, success, last_step, error):
    os.makedirs(output_dir, exist_ok=True)
    status = {
        "success": int(success),
        "last_step_executed": last_step,
        "error_message": "None" if success else error,
    }
    with open(os.path.join(output_dir, "status.json"), "w") as f:
        json.dump(status, f, indent=4)


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
            try:
                shutil.rmtree(target_dir)
            except Exception as e:
                print(f"Warning: Could not remove {target_dir}: {e}")
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

    exit_code = 0
    try:
        print("Create build configuration...")
        cfg = create_build_config(output_dir, folding_config_path, estimate_only=args.estimate_only)
        print("=" * 50)

        # Returns 0 on success, -1 on failure
        ret = build.build_dataflow_cfg(model_file, cfg)
        last_step, error = parse_build_log(output_dir)

        if ret == 0:
            write_status(output_dir, True, last_step, "None")
            print("=" * 50)
            print("Synthesis/Estimation executed successfully!")
        else:
            write_status(output_dir, False, last_step, error)
            print("Build failed!")
            exit_code = 1

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        write_status(output_dir, False, "None", str(e))
        exit_code = 1
    finally:
        collect_reports_and_cleanup(output_dir)
        clean_finn_internal_tmp(unique_tmp_dir)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()