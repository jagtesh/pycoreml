"""pycoreml — CoreML inference from Python

Powered by Swift & ApplePy. Direct access to Apple's CoreML framework
with explicit compute unit selection (CPU, GPU, Neural Engine).

Usage:
    import pycoreml
    desc = pycoreml.load_model("/path/to/model.mlmodelc")
    result = pycoreml.predict("/path/to/model.mlmodelc", {"x": 42.0})
"""
import importlib
import os
import sys

if sys.platform != "darwin":
    raise ImportError("pycoreml only supports macOS")


def _load_native():
    """Load the compiled Swift extension module."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(pkg_dir, "_native", "pycoreml.so")

    if not os.path.exists(so_path):
        raise ImportError(
            "Native extension not found. Build it first:\n"
            "  pip install -e .\n"
            "  # or: python setup.py build_ext --inplace"
        )

    spec = importlib.util.spec_from_file_location("pycoreml._native.pycoreml", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_native = _load_native()

# Re-export all public functions
load_model = _native.load_model
model_description = _native.model_description
predict = _native.predict
list_compute_units = _native.list_compute_units
predict_with_options = _native.predict_with_options

__all__ = [
    "load_model",
    "model_description",
    "predict",
    "list_compute_units",
    "predict_with_options",
]

__version__ = "0.2.0"
