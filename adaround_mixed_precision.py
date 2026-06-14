#!/usr/bin/env python3

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

import aimet_onnx
from aimet_common.defs import QuantScheme
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from aimet_onnx.quantsim import QuantizationSimModel


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run AIMET AdaRound on an ONNX model with per-tensor encoding "
            "configuration from JSON and calibration data stored as raw .f32 files."
        )
    )
    parser.add_argument("onnx_model", type=str, help="Path to input ONNX model file")
    parser.add_argument(
        "layer_config_json",
        type=str,
        help="Path to JSON file describing activation and parameter encoding settings",
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        required=True,
        help=(
            "Calibration directory. Expected layout:\n"
            "  calib_dir/frame_0000/<input_name>.f32\n"
            "  calib_dir/frame_0001/<input_name>.f32\n"
            "  ..."
        ),
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help=(
            "Optional ORT providers. Example: "
            "--providers CUDAExecutionProvider CPUExecutionProvider"
        ),
    )
    parser.add_argument(
        "--num-calib-samples",
        type=int,
        default=32,
        help="Maximum number of frame_* directories to use for calibration",
    )
    parser.add_argument(
        "--adaround-iterations",
        type=int,
        default=10000,
        help="AdaRound iterations per layer",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_adaround",
        help="Directory to save exported model and encodings",
    )
    parser.add_argument(
        "--export-prefix",
        type=str,
        default="model_after_adaround",
        help="Filename prefix for exported outputs",
    )
    return parser.parse_args()


def _validate_encoding_entry(section_name: str, tensor_name: str, entry: dict):
    if not isinstance(entry, dict):
        raise ValueError(
            f"{section_name}.{tensor_name} must be an object with fields 'bw' and 'type'."
        )

    if "bw" not in entry:
        raise ValueError(f"{section_name}.{tensor_name} is missing 'bw'.")
    if "type" not in entry:
        raise ValueError(f"{section_name}.{tensor_name} is missing 'type'.")

    bw = entry["bw"]
    enc_type = entry["type"]

    if isinstance(bw, str):
        if not bw.isdigit():
            raise ValueError(f"{section_name}.{tensor_name}.bw must be integer-like.")
        bw = int(bw)

    if bw not in (8, 16):
        raise ValueError(
            f"{section_name}.{tensor_name}.bw must be 8 or 16, got {bw}."
        )

    if enc_type not in ("int", "fp16"):
        raise ValueError(
            f"{section_name}.{tensor_name}.type must be 'int' or 'fp16', got {enc_type}."
        )

    return {"bw": bw, "type": enc_type}


def load_layer_config(json_path: str) -> Dict[str, Dict[str, dict]]:
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    activation_encodings = config.get("activation_encodings", {})
    parameter_encodings = config.get("parameter_encodings", {})

    if not isinstance(activation_encodings, dict):
        raise ValueError("'activation_encodings' must be an object.")
    if not isinstance(parameter_encodings, dict):
        raise ValueError("'parameter_encodings' must be an object.")

    validated_activation_encodings = {}
    validated_parameter_encodings = {}

    for tensor_name, entry in activation_encodings.items():
        validated_activation_encodings[tensor_name] = _validate_encoding_entry(
            "activation_encodings", tensor_name, entry
        )

    for tensor_name, entry in parameter_encodings.items():
        validated_parameter_encodings[tensor_name] = _validate_encoding_entry(
            "parameter_encodings", tensor_name, entry
        )

    return {
        "activation_encodings": validated_activation_encodings,
        "parameter_encodings": validated_parameter_encodings,
    }


def get_providers(user_providers: Optional[List[str]]) -> List[str]:
    if user_providers:
        return user_providers

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_model(model_path: str) -> onnx.ModelProto:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model file not found: {model_path}")
    return onnx.load_model(str(model_path))


def onnx_elem_type_to_numpy(elem_type: int):
    mapping = {
        TensorProto.FLOAT: np.float32,
        TensorProto.FLOAT16: np.float16,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT8: np.int8,
        TensorProto.INT16: np.int16,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
        TensorProto.UINT8: np.uint8,
        TensorProto.UINT16: np.uint16,
        TensorProto.BOOL: np.bool_,
    }
    if elem_type not in mapping:
        raise ValueError(f"Unsupported ONNX input dtype enum: {elem_type}")
    return mapping[elem_type]


def get_model_inputs(model: onnx.ModelProto):
    inputs = []
    initializer_names = {init.name for init in model.graph.initializer}

    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue

        tensor_type = value_info.type.tensor_type
        elem_type = tensor_type.elem_type
        np_dtype = onnx_elem_type_to_numpy(elem_type)

        shape = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                raise ValueError(
                    f"Input '{value_info.name}' has dynamic/unknown shape. "
                    "Raw .f32 loading requires static ONNX input shapes."
                )

        inputs.append({"name": value_info.name, "shape": shape, "dtype": np_dtype})

    if not inputs:
        raise ValueError("No runtime ONNX graph inputs found.")

    return inputs


def print_model_inputs(model_inputs):
    print("\n[Model inputs]")
    for inp in model_inputs:
        print(
            f"  name={inp['name']!r}, "
            f"shape={tuple(inp['shape'])}, "
            f"dtype={np.dtype(inp['dtype']).name}"
        )


def print_model_nodes(model: onnx.ModelProto):
    print("\n[Model nodes]")
    for node in model.graph.node:
        print(f"  name={node.name!r}, op_type={node.op_type}")


def load_raw_tensor_from_f32(path: Path, shape: List[int], target_dtype):
    arr = np.fromfile(path, dtype=np.float32)

    expected_size = 1
    for d in shape:
        expected_size *= d

    if arr.size != expected_size:
        raise ValueError(
            f"File {path} has {arr.size} float32 elements, but expected {expected_size} "
            f"for shape {tuple(shape)}"
        )

    arr = arr.reshape(shape)

    if np.dtype(target_dtype) != np.dtype(np.float32):
        arr = arr.astype(target_dtype)

    return arr


def load_calibration_data(calib_dir: str, model_inputs, num_calib_samples: int):
    calib_path = Path(calib_dir)
    if not calib_path.exists() or not calib_path.is_dir():
        raise ValueError(f"Calibration directory does not exist: {calib_dir}")

    frame_dirs = sorted(
        [p for p in calib_path.iterdir() if p.is_dir() and p.name.startswith("frame_")]
    )

    if not frame_dirs:
        raise ValueError(
            f"No frame_* directories found in calibration directory: {calib_dir}"
        )

    frame_dirs = frame_dirs[:num_calib_samples]

    required_input_names = [inp["name"] for inp in model_inputs]
    input_map = {inp["name"]: inp for inp in model_inputs}

    onnx_data = []

    print("\n[Calibration frames]")
    for frame_dir in frame_dirs:
        print(f"  using: {frame_dir}")
        sample_feed = {}

        for input_name in required_input_names:
            file_path = frame_dir / f"{input_name}.f32"
            if not file_path.is_file():
                raise ValueError(
                    f"Missing calibration file: {file_path}\n"
                    "Each frame directory must contain one .f32 file per ONNX input."
                )

            input_info = input_map[input_name]
            tensor = load_raw_tensor_from_f32(
                file_path,
                shape=input_info["shape"],
                target_dtype=input_info["dtype"],
            )
            sample_feed[input_name] = tensor

        onnx_data.append(sample_feed)

    return onnx_data


def create_quantsim(model: onnx.ModelProto, providers: List[str]) -> QuantizationSimModel:
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.min_max,
        param_type=aimet_onnx.int8,
        activation_type=aimet_onnx.int8,
        providers=providers,
    )
    return sim


def _try_set_quantizer_bitwidth(quantizer, bitwidth: int):
    if hasattr(quantizer, "bitwidth"):
        quantizer.bitwidth = bitwidth
    elif hasattr(quantizer, "_bitwidth"):
        quantizer._bitwidth = bitwidth
    else:
        raise AttributeError("Quantizer does not expose bitwidth attribute.")


def _try_disable_quantizer(quantizer):
    if hasattr(quantizer, "enabled"):
        quantizer.enabled = False
    elif hasattr(quantizer, "_enabled"):
        quantizer._enabled = False
    else:
        raise AttributeError("Quantizer does not expose enabled attribute.")


def apply_encoding_config(sim: QuantizationSimModel, layer_config: Dict[str, Dict[str, dict]]):
    activation_cfg = layer_config.get("activation_encodings", {})
    parameter_cfg = layer_config.get("parameter_encodings", {})

    if not hasattr(sim, "qc_quantize_op_dict"):
        raise RuntimeError(
            "This AIMET build does not expose sim.qc_quantize_op_dict. "
            "Please adapt encoding override logic for your AIMET 2.28.0 build."
        )

    applied_activation = []
    missing_activation = []
    applied_parameter = []
    missing_parameter = []

    for op_name, qc_op in sim.qc_quantize_op_dict.items():
        input_quantizers = getattr(qc_op, "input_quantizers", [])
        output_quantizers = getattr(qc_op, "output_quantizers", [])
        param_quantizers = getattr(qc_op, "param_quantizers", {})

        if hasattr(qc_op, "op"):
            op = qc_op.op
            output_names = list(getattr(op, "output", []))
            input_names = list(getattr(op, "input", []))
        else:
            output_names = []
            input_names = []

        for idx, tensor_name in enumerate(output_names):
            if tensor_name not in activation_cfg:
                continue
            if idx >= len(output_quantizers):
                missing_activation.append(tensor_name)
                continue

            cfg = activation_cfg[tensor_name]
            quantizer = output_quantizers[idx]
            if cfg["type"] == "fp16":
                _try_disable_quantizer(quantizer)
            else:
                _try_set_quantizer_bitwidth(quantizer, cfg["bw"])
            applied_activation.append((tensor_name, cfg))

        for idx, tensor_name in enumerate(input_names):
            if tensor_name not in activation_cfg:
                continue
            if idx >= len(input_quantizers):
                missing_activation.append(tensor_name)
                continue

            cfg = activation_cfg[tensor_name]
            quantizer = input_quantizers[idx]
            if cfg["type"] == "fp16":
                _try_disable_quantizer(quantizer)
            else:
                _try_set_quantizer_bitwidth(quantizer, cfg["bw"])
            applied_activation.append((tensor_name, cfg))

        for param_name, quantizer in param_quantizers.items():
            if param_name not in parameter_cfg:
                continue

            cfg = parameter_cfg[param_name]
            if cfg["type"] == "fp16":
                _try_disable_quantizer(quantizer)
            else:
                _try_set_quantizer_bitwidth(quantizer, cfg["bw"])
            applied_parameter.append((param_name, cfg))

    configured_activation_names = set(activation_cfg.keys())
    matched_activation_names = {name for name, _ in applied_activation}
    for tensor_name in sorted(configured_activation_names - matched_activation_names):
        missing_activation.append(tensor_name)

    configured_parameter_names = set(parameter_cfg.keys())
    matched_parameter_names = {name for name, _ in applied_parameter}
    for tensor_name in sorted(configured_parameter_names - matched_parameter_names):
        missing_parameter.append(tensor_name)

    print("\n[Activation encoding config]")
    for tensor_name, cfg in applied_activation:
        print(f"  applied: {tensor_name} -> bw={cfg['bw']}, type={cfg['type']}")

    print("\n[Parameter encoding config]")
    for tensor_name, cfg in applied_parameter:
        print(f"  applied: {tensor_name} -> bw={cfg['bw']}, type={cfg['type']}")

    if missing_activation:
        print("\n[WARNING] Activation entries not matched:")
        for name in sorted(set(missing_activation)):
            print(f"  missing: {name}")

    if missing_parameter:
        print("\n[WARNING] Parameter entries not matched:")
        for name in sorted(set(missing_parameter)):
            print(f"  missing: {name}")


def main():
    args = parse_args()

    providers = get_providers(args.providers)
    print(f"[INFO] Using providers: {providers}")

    model = load_model(args.onnx_model)
    model_inputs = get_model_inputs(model)
    layer_config = load_layer_config(args.layer_config_json)

    print_model_inputs(model_inputs)
    print_model_nodes(model)

    print("[INFO] Folding batch norms...")
    fold_all_batch_norms_to_weight(model)

    print("[INFO] Creating QuantizationSimModel...")
    sim = create_quantsim(model, providers)

    print("[INFO] Applying activation/parameter encoding config...")
    apply_encoding_config(sim, layer_config)

    onnx_data = load_calibration_data(
        calib_dir=args.calib_dir,
        model_inputs=model_inputs,
        num_calib_samples=args.num_calib_samples,
    )

    if not onnx_data:
        raise RuntimeError("No calibration samples were loaded.")

    print("[INFO] Computing encodings...")
    sim.compute_encodings(onnx_data)

    print("[INFO] Running AdaRound...")
    aimet_onnx.apply_adaround(
        sim,
        onnx_data,
        iterations=args.adaround_iterations,
    )

    print("[INFO] Recomputing encodings after AdaRound...")
    sim.compute_encodings(onnx_data)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Exporting quantized model and encodings...")
    sim.export(path=str(output_dir), filename_prefix=args.export_prefix)

    print("[DONE]")
    print(f"Export directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
