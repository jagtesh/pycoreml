#!/usr/bin/env python3
"""PyCoreML — Example Usage

Demonstrates CoreML inference from Python via ApplePy.
Requires a compiled CoreML model (.mlmodelc) to run predictions.
"""
import pycoreml

# ── Available Compute Units ─────────────────────────────────

print("=== Available Compute Units ===\n")

units = pycoreml.list_compute_units()
for unit in units:
    print(f"  • {unit}")

# ── Loading a Model ─────────────────────────────────────────

print("\n=== Model Loading ===\n")

# To run this demo, you need a compiled CoreML model.
# You can create one with coremltools:
#
#   import coremltools as ct
#   import torch
#
#   class SimpleModel(torch.nn.Module):
#       def forward(self, x):
#           return x * 2 + 1
#
#   model = ct.convert(torch.jit.trace(SimpleModel(), torch.tensor([1.0])))
#   model.save("simple.mlpackage")
#
# Then compile it:
#   xcrun coremlcompiler compile simple.mlpackage .

MODEL_PATH = "simple.mlmodelc"  # Change to your model path

try:
    # Load and describe model
    info = pycoreml.load_model(MODEL_PATH)
    print(info)

    # Get metadata
    desc = pycoreml.model_description(MODEL_PATH)
    print(f"\nMetadata: {desc}")

    # ── Running Predictions ─────────────────────────────────

    print("\n=== Predictions ===\n")

    # Basic prediction
    result = pycoreml.predict(MODEL_PATH, {"x": 42.0})
    print(f"predict(x=42.0) → {result}")

    # Prediction with explicit compute unit
    result = pycoreml.predict_with_options(
        MODEL_PATH,
        {"x": 42.0},
        "cpuAndNeuralEngine"
    )
    print(f"predict(x=42.0, compute_unit='cpuAndNeuralEngine') → {result}")

    # Compare compute units
    print("\n=== Compute Unit Comparison ===\n")
    for unit in ["cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine", "all"]:
        result = pycoreml.predict_with_options(MODEL_PATH, {"x": 42.0}, unit)
        print(f"  {unit:25s} → {result}")

except Exception as e:
    print(f"⚠ Skipping model demo: {e}")
    print("  (Provide a valid .mlmodelc path to run predictions)")

print("\n🎉 Done!")
