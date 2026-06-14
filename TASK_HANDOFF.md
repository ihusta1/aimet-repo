# Task Handoff: adaround_mixed_precision.py

This note is for the next Copilot / engineer continuing the task.

## Goal

Implement or finalize `adaround_mixed_precision.py`, a command-line script for **AIMET 2.28.0** that:

- loads an ONNX model
- loads calibration tensors from a `calib_dir` directory structure
- reads activation / parameter encoding settings from JSON
- applies AdaRound
- exports the resulting quantized artifacts

The target behavior is documented in `README.md`.

---

## Source Reference

The implementation is based on:

```text
Examples/onnx/quantization/adaround.ipynb
```

GitHub URL:

```text
https://github.com/qualcomm/aimet/blob/develop/Examples/onnx/quantization/adaround.ipynb
```

Core notebook flow to preserve:

1. fold batch norms
2. build `QuantizationSimModel`
3. compute encodings
4. run `apply_adaround`
5. recompute encodings
6. export

---

## Finalized interface requirements

These items were explicitly clarified and should be treated as fixed requirements.

### 1. Inputs
The script should accept:

- ONNX model path
- JSON config path
- `--calib-dir`

### 2. Calibration directory format
Calibration directory must look like:

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
```

Where:
- each `frame_*` directory is one sample
- each file name corresponds to an ONNX runtime input name

### 3. `.f32` meaning
Each `.f32` file is:
- raw binary
- float32
- no header

Expected load behavior:
- `np.fromfile(path, dtype=np.float32)`
- verify tensor size from ONNX shape
- reshape
- cast to ONNX input dtype if needed

### 4. ONNX input detection
Input names, dtypes, and shapes should be discovered automatically from the ONNX model.

No manual `--input-name` or `--input-shape` should be required.

### 5. JSON format
The JSON format is **encoding-oriented**, not layer-policy-oriented.

Expected format:

```json
{
  "activation_encodings": {
    "tensor_name": {
      "bw": 8,
      "type": "int"
    }
  },
  "parameter_encodings": {
    "param_name": {
      "bw": 8,
      "type": "int"
    }
  }
}
```

Supported values:
- `bw`: `8` or `16`
- `type`: `"int"` or `"fp16"`

---

## Most important unresolved area

The biggest risk is the exact AIMET 2.28.0 quantizer API.

The draft script logic assumed something like:

- `sim.qc_quantize_op_dict`
- each op has:
  - `input_quantizers`
  - `output_quantizers`
  - `param_quantizers`
- each quantizer has:
  - `bitwidth`
  - `enabled`

This needs to be verified against the real AIMET 2.28.0 environment or source.

### Why this matters
The script’s calibration loading and ONNX input parsing are relatively straightforward.

The fragile part is:
- mapping JSON tensor names to actual AIMET quantizers
- setting activation / parameter bitwidths correctly
- implementing `"fp16"` behavior correctly

---

## Recommended next steps

### Step 1: verify AIMET 2.28.0 API
Check the actual AIMET ONNX classes and runtime objects for:
- quantizer container names
- quantizer attribute names
- how tensor names map to quantizer instances

### Step 2: tighten the encoding-application logic
Need robust handling for:
- activation tensor name matching
- parameter tensor name matching
- unmatched JSON keys
- disabling quantizers for `fp16`

### Step 3: add a template-dump helper
Recommended enhancement:
- dump candidate activation tensor names
- dump candidate parameter tensor names
- optionally generate starter JSON template

This will make user configuration much easier.

### Step 4: improve validation
Recommended:
- fail clearly if any `frame_*` directory is missing a required input file
- fail clearly if ONNX input shape is dynamic
- print which JSON entries were matched / unmatched

---

## Expected script behavior

The final script should approximately do this:

1. parse CLI args
2. load ONNX model
3. inspect runtime ONNX inputs
4. load calibration frames from `calib_dir/frame_xxxx/<input_name>.f32`
5. fold batch norms
6. create `QuantizationSimModel`
7. apply JSON encoding config to quantizers
8. compute encodings
9. run AdaRound
10. recompute encodings
11. export model and encodings

---

## Notes on JSON keys

The config keys are expected to be real graph names, not aliases.

Examples:
- activation tensor names like `Conv_0_output_0`
- parameter names like `conv1.weight`

Actual names depend on the ONNX model.

A helper to enumerate these names would be valuable.

---

## Notes on static shapes

Dynamic / symbolic ONNX input shapes are currently out of scope.

Because calibration files are raw `.f32`, there is no embedded shape information.

---

## Deliverable expectation

A good completion of this task would provide:

1. working `adaround_mixed_precision.py`
2. `README.md`
3. clear validation / logging
4. ideally a helper to generate or dump candidate JSON keys

---

## If you need to inspect source

Primary notebook reference:

```text
Examples/onnx/quantization/adaround.ipynb
```

GitHub URL:

```text
https://github.com/qualcomm/aimet/blob/develop/Examples/onnx/quantization/adaround.ipynb
```

Repository:

```text
qualcomm/aimet
```
