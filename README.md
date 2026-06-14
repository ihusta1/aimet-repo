# adaround_mixed_precision.py

A command-line tool that converts the AIMET ONNX AdaRound notebook flow into a reusable script for **AIMET 2.28.0**.

This script is intended for ONNX models that use:

- calibration data stored as raw **`.f32`** files
- per-tensor mixed-precision / encoding settings supplied by a **JSON file**

It is based on the AIMET ONNX AdaRound example notebook:

```text
Examples/onnx/quantization/adaround.ipynb
```

Reference:

```text
https://github.com/qualcomm/aimet/blob/develop/Examples/onnx/quantization/adaround.ipynb
```

---

## Overview

The script automates the following workflow:

1. Load an ONNX model
2. Inspect ONNX runtime input ports automatically
3. Load calibration data from a directory of raw `.f32` tensors
4. Fold batch normalization layers into weights
5. Create an AIMET `QuantizationSimModel`
6. Apply activation / parameter encoding settings from JSON
7. Compute encodings
8. Run AdaRound
9. Recompute encodings
10. Export the quantized model and encodings

---

## Requirements

- Python environment with **AIMET 2.28.0**
- `onnx`
- `onnxruntime`
- `numpy`

Depending on your environment, CUDA-enabled ONNX Runtime may also be used.

---

## Usage

### Command

```bash
python adaround_mixed_precision.py \
  <model.onnx> \
  <layer_config.json> \
  --calib-dir <calib_dir> \
  [--num-calib-samples 32] \
  [--adaround-iterations 10000] \
  [--output-dir output_adaround] \
  [--export-prefix model_after_adaround] \
  [--providers CUDAExecutionProvider CPUExecutionProvider]
```

### Example

```bash
python adaround_mixed_precision.py \
  model.onnx \
  layer_config.json \
  --calib-dir ./calib_dir \
  --num-calib-samples 32 \
  --adaround-iterations 10000 \
  --output-dir ./adaround_out \
  --export-prefix my_model
```

---

## CLI Arguments

### Positional arguments

#### `onnx_model`
Path to the input ONNX model file.

#### `layer_config_json`
Path to the JSON file describing activation and parameter encoding settings.

### Options

#### `--calib-dir`
Path to calibration data directory.

This argument is required.

#### `--num-calib-samples`
Maximum number of `frame_*` calibration directories to use.

Default:

```text
32
```

#### `--adaround-iterations`
Number of AdaRound iterations per layer.

Default:

```text
10000
```

#### `--output-dir`
Directory where exported model artifacts will be written.

Default:

```text
output_adaround
```

#### `--export-prefix`
Filename prefix used by AIMET export.

Default:

```text
model_after_adaround
```

#### `--providers`
Optional ONNX Runtime execution providers.

Example:

```bash
--providers CUDAExecutionProvider CPUExecutionProvider
```

If omitted, the intended behavior is:

- use `CUDAExecutionProvider` + `CPUExecutionProvider` if CUDA is available
- otherwise use `CPUExecutionProvider`

---

## Specification

### ONNX input analysis

The script automatically inspects ONNX runtime inputs and determines:

- input name
- input shape
- input dtype

This means you do **not** need to manually specify input port names or shapes.

The script should ignore graph initializers that also appear in `model.graph.input`, and only use true runtime inputs.

### Static shape requirement

All ONNX runtime inputs must have **static shapes**.

Dynamic or symbolic dimensions are not supported, because raw `.f32` files do not contain shape metadata.

### Calibration file behavior

Each `.f32` file is treated as:

- raw binary
- no header
- `float32` storage format
- one full tensor for one ONNX input

The intended load behavior is:

1. load file as `np.float32`
2. verify element count matches ONNX input shape
3. reshape to ONNX input shape
4. cast to ONNX input dtype if needed

### Quantization flow

The script is expected to follow this sequence:

1. `fold_all_batch_norms_to_weight(model)`
2. create `QuantizationSimModel(...)`
3. apply encoding config from JSON
4. `sim.compute_encodings(calibration_data)`
5. `aimet_onnx.apply_adaround(sim, calibration_data, iterations=...)`
6. `sim.compute_encodings(calibration_data)` again
7. `sim.export(...)`

---

## calib_dir Format

Calibration data must be organized as one subdirectory per sample.

### Directory structure

```text
calib_dir/
  frame_0000/
    A.f32
    B.f32
    C.f32
  frame_0001/
    A.f32
    B.f32
    C.f32
  ...
  frame_0031/
    A.f32
    B.f32
    C.f32
```

Where:

- `frame_0000`, `frame_0001`, ... are calibration samples
- `A`, `B`, `C`, ... are ONNX input port names

### Naming rule

For each `frame_xxxx` directory:

- there must be one `.f32` file per ONNX runtime input
- file name must exactly match the ONNX input name

Format:

```text
<input_name>.f32
```

### Example

If the ONNX model inputs are:

- `image`
- `scale`

then each frame directory must contain:

```text
image.f32
scale.f32
```

### Tensor size rule

Each file must contain exactly one tensor's worth of data.

Example:

If input `image` has shape:

```text
(1, 3, 224, 224)
```

then `image.f32` must contain:

```text
1 * 3 * 224 * 224 = 150528
```

float32 values.

---

## Input JSON Format

The JSON file is **encoding-oriented**.

It defines quantization settings separately for:

- activation tensors
- parameter tensors

### JSON structure

```json
{
  "activation_encodings": {
    "activation_tensor_name": {
      "bw": 8,
      "type": "int"
    }
  },
  "parameter_encodings": {
    "parameter_tensor_name": {
      "bw": 8,
      "type": "int"
    }
  }
}
```

### Fields

#### `activation_encodings`
Object mapping activation tensor names to encoding settings.

#### `parameter_encodings`
Object mapping parameter tensor names to encoding settings.

#### `bw`
Bitwidth.

Supported values:

- `8`
- `16`

Integer form is preferred.

#### `type`
Encoding type.

Supported values:

- `"int"`
- `"fp16"`

### Example

```json
{
  "activation_encodings": {
    "Conv_0_output_0": {
      "bw": 8,
      "type": "int"
    },
    "Conv_3_output_0": {
      "bw": 16,
      "type": "int"
    }
  },
  "parameter_encodings": {
    "conv1.weight": {
      "bw": 8,
      "type": "int"
    },
    "conv2.weight": {
      "bw": 16,
      "type": "int"
    }
  }
}
```

---

## Meaning of JSON Settings

### Activation example

```json
"Conv_0_output_0": {
  "bw": 8,
  "type": "int"
}
```

means the activation quantizer should use **int8**.

### Parameter example

```json
"conv1.weight": {
  "bw": 16,
  "type": "int"
}
```

means the parameter quantizer should use **int16**.

### `fp16`

```json
"some_tensor": {
  "bw": 16,
  "type": "fp16"
}
```

is intended to mean that the corresponding quantizer is disabled or treated as FP16 in the script logic.

The exact implementation depends on the AIMET quantizer API.

---

## Important Note About JSON Keys

The keys in the JSON file are **not arbitrary labels**.

They must match actual names in the ONNX / AIMET graph, such as:

- activation tensor names
- parameter tensor names

If a JSON key does not match any real tensor or quantizer, the script should report it as unmatched.

---

## Outputs

The script exports artifacts to `--output-dir`.

Typical outputs include:

- ONNX model after AdaRound
- encoding files generated by AIMET

Exact filenames depend on:
- `--export-prefix`
- AIMET export behavior

---

## Limitations

- calibration inputs must be `.f32`
- runtime ONNX inputs must have static shapes
- JSON names must match real ONNX / AIMET tensor names
- mixed-precision application may require small adjustments if AIMET internal APIs differ in a given 2.28.0 build

---

## Origin

This script is derived from the AIMET ONNX AdaRound example notebook:

```text
Examples/onnx/quantization/adaround.ipynb
```

Reference URL:

```text
https://github.com/qualcomm/aimet/blob/develop/Examples/onnx/quantization/adaround.ipynb
```
