# PyCoreML

CoreML inference from Python — powered by Swift & [ApplePy](../ApplePy).

Direct access to Apple's CoreML framework with explicit compute unit selection (CPU, GPU, Neural Engine).

> **macOS only** — requires Swift 6.0+ toolchain

## Install

```bash
pip install -e .
```

## Usage

```python
import pycoreml

# Load and describe a model
info = pycoreml.load_model("/path/to/model.mlmodelc")
desc = pycoreml.model_description("/path/to/model.mlmodelc")

# Run prediction
result = pycoreml.predict("/path/to/model.mlmodelc", {"x": 42.0})

# Predict with explicit compute unit (the killer feature)
result = pycoreml.predict_with_options(
    "/path/to/model.mlmodelc",
    {"x": 42.0},
    "cpuAndNeuralEngine"  # or "cpuOnly", "cpuAndGPU", "all"
)
```

## API

| Function | Returns | Description |
|----------|---------|-------------|
| `load_model(path)` | `str` | Load model and return description |
| `model_description(path)` | `dict[str, str]` | Model metadata |
| `predict(path, inputs)` | `dict[str, float]` | Run inference |
| `list_compute_units()` | `list[str]` | Available compute units |
| `predict_with_options(path, inputs, unit)` | `dict[str, float]` | Predict with compute unit control |

## Examples

See [`pycoreml/examples/demo.py`](pycoreml/examples/demo.py) for a full demo.

## License

BSD-3-Clause © Jagtesh Chadha — see [LICENSE](LICENSE).
