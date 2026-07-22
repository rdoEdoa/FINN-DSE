import argparse
import os
import shutil
import json
import numpy as np
import sys
import glob
import getpass
import warnings

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType

from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
from finn.builder.build_dataflow_steps import build_dataflow_step_lookup
from finn.builder.build_dataflow_steps import *

# ─── RESNET50 SPECIFIC TRANSFORMATION IMPORTS ─────────────────────────────────
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    ConvertSubToAdd, ConvertDivToMul, GiveReadableTensorNames,
    GiveUniqueNodeNames, SortGraph, RemoveUnusedTensors,
    GiveUniqueParameterTensors, RemoveStaticGraphInputs
)
from finn.transformation.streamline.absorb import (
    AbsorbScalarMulAddIntoTopK, AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold, FactorOutMulSignMagnitude,
    Absorb1BitMulIntoMatMul, Absorb1BitMulIntoConv,
    AbsorbConsecutiveTransposes, AbsorbTransposeIntoMultiThreshold
)
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd, CollapseRepeatedMul
)
from finn.transformation.streamline.reorder import (
    MoveAddPastMul, MoveScalarMulPastMatMul, MoveScalarAddPastMatMul,
    MoveAddPastConv, MoveScalarMulPastConv, MoveScalarLinearPastInvariants,
    MoveMaxPoolPastMultiThreshold, MoveLinearPastEltwiseAdd, MoveLinearPastFork
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.insert_topk import InsertTopK
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM RESNET50 STEPS
# ─────────────────────────────────────────────────────────────────────────────
def step_resnet50_tidy(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(InsertTopK())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    return model

def step_resnet50_streamline_linear(model: ModelWrapper, cfg: DataflowBuildConfig):
    streamline_transformations = [
        AbsorbScalarMulAddIntoTopK(), 
        ConvertSubToAdd(),
        ConvertDivToMul(),
        RemoveIdentityOps(),
        CollapseRepeatedMul(),
        BatchNormToAffine(),
        ConvertSignToThres(),
        MoveAddPastMul(),
        MoveScalarAddPastMatMul(),
        MoveAddPastConv(),
        MoveScalarMulPastMatMul(),
        MoveScalarMulPastConv(),
        MoveScalarLinearPastInvariants(),
        MoveAddPastMul(),
        CollapseRepeatedAdd(),
        CollapseRepeatedMul(),
        AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(),
        MoveMaxPoolPastMultiThreshold(),
        AbsorbMulIntoMultiThreshold(),
        Absorb1BitMulIntoMatMul(),
        Absorb1BitMulIntoConv(),
        RoundAndClipThresholds(),
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model

def step_resnet50_streamline_nonlinear(model: ModelWrapper, cfg: DataflowBuildConfig):
    streamline_transformations = [
        MoveLinearPastEltwiseAdd(),
        MoveLinearPastFork(),
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model

def step_resnet50_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    for iter_id in range(4):
        model = step_resnet50_streamline_linear(model, cfg)
        model = step_resnet50_streamline_nonlinear(model, cfg)

        # big loop tidy up
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(SortGraph())

    model = model.transform(DoubleToSingleFloat())
    return model

def step_resnet50_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InferDataLayouts())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())

    to_hw_transformations = [
        to_hw.InferAddStreamsLayer,
        LowerConvsToMatMul,
        to_hw.InferChannelwiseLinearLayer,
        to_hw.InferPool,
        AbsorbTransposeIntoMultiThreshold,
        RoundAndClipThresholds,
        to_hw.InferQuantizedMatrixVectorActivation,
        to_hw.InferThresholdingLayer,
        AbsorbConsecutiveTransposes,
        to_hw.InferConvInpGen,
        to_hw.InferDuplicateStreamsLayer,
        to_hw.InferLabelSelectLayer,
    ]
    for trn in to_hw_transformations:
        model = model.transform(trn())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())

    return model

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD BUILD PIPELINE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
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
    """Gives the list of build steps to be executed using custom ResNet steps."""
    steps = [
        "step_qonnx_to_finn",
        "step_resnet50_tidy",           # Custom
        "step_resnet50_streamline",     # Custom
        "step_resnet50_convert_to_hw",  # Custom
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
    """Creates and configures the DataflowBuildConfig for Alveo U250."""
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
        synth_clk_period_ns=3.33,  
        hls_clk_period_ns=3.33,
        folding_config_file=folding_config_path,
        auto_fifo_depths=False,
        fpga_part="xcu250-figd2104-2L-e",  
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
    
    # Register the custom ResNet50 steps
    step_lookup["step_resnet50_tidy"] = step_resnet50_tidy
    step_lookup["step_resnet50_streamline"] = step_resnet50_streamline
    step_lookup["step_resnet50_convert_to_hw"] = step_resnet50_convert_to_hw
    
    status_log = {"success": 0, "last_step_executed": "None", "error_message": "None"}
    os.makedirs(output_dir, exist_ok=True)
    status_path = os.path.join(output_dir, "status.json")

    for i, step_name in enumerate(build_steps):
        if verbose:
            print(f"Running step: {step_name} [{i+1}/{len(build_steps)}]")
        try:
            if step_name == "step_resnet50_convert_to_hw":
                model = thresholds_round(model)
            step_function = step_lookup[step_name]
            model = step_function(model, cfg)

            if step_name == "step_resnet50_convert_to_hw":
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

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
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
    
    original_model_file = os.path.join("dataset", args.directory, args.model_name)
    if not os.path.exists(original_model_file):
        print(f"Error: The model '{original_model_file}' does not exist")
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
        print(f"Model: {original_model_file}")
        print(f"Output directory: {output_dir}")
        print(f"Folding configuration file: {folding_config_path}")
        print("-" * 50)
    
    try:
        print(f"Loading native model directly: {original_model_file}")
        # NO INTERCEPTOR: Pass original file directly to ModelWrapper
        model = ModelWrapper(original_model_file)
        
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