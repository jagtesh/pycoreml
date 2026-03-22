// PyCoreML — CoreML inference from Python
// A showcase example for ApplePy.
//
// Usage from Python:
//   import pycoreml
//   model = pycoreml.load_model("/path/to/model.mlmodelc")
//   desc = pycoreml.model_description("/path/to/model.mlmodelc")
//   result = pycoreml.predict("/path/to/model.mlmodelc", {"input": [1.0, 2.0, 3.0]})

import ApplePy
import Foundation
import CoreML
@preconcurrency import ApplePyFFI

// MARK: - Custom Exception

let CoreMLError = PyExceptionType(name: "pycoreml.CoreMLError", doc: "CoreML operation failed")

enum CoreMLBridgeError: Error, PyExceptionMapping {
    case modelLoadFailed(String)
    case predictionFailed(String)
    case invalidInput(String)

    var pythonExceptionType: PyExceptionType { CoreMLError }
    var pythonMessage: String {
        switch self {
        case .modelLoadFailed(let msg): return "Model load failed: \(msg)"
        case .predictionFailed(let msg): return "Prediction failed: \(msg)"
        case .invalidInput(let msg): return "Invalid input: \(msg)"
        }
    }
}

// MARK: - load_model

/// Load a CoreML model and return a description string.
/// This validates the model can be loaded.
///
/// ```python
/// desc = pycoreml.load_model("/path/to/model.mlmodelc")
/// ```
@PyFunction
func load_model(path: String) throws -> String {
    let url = URL(fileURLWithPath: path)
    let model: MLModel
    do {
        model = try MLModel(contentsOf: url)
    } catch {
        throw CoreMLBridgeError.modelLoadFailed("\(error)")
    }

    let desc = model.modelDescription
    var info: [String] = []
    info.append("Model loaded: \(path)")

    // Input features
    let inputs = desc.inputDescriptionsByName
    info.append("Inputs (\(inputs.count)):")
    for (name, feature) in inputs {
        info.append("  - \(name): \(feature.type.rawValue)")
    }

    // Output features
    let outputs = desc.outputDescriptionsByName
    info.append("Outputs (\(outputs.count)):")
    for (name, feature) in outputs {
        info.append("  - \(name): \(feature.type.rawValue)")
    }

    return info.joined(separator: "\n")
}

// MARK: - model_description

/// Get model metadata as a dictionary.
///
/// ```python
/// desc = pycoreml.model_description("/path/to/model.mlmodelc")
/// # {"inputs": {"x": "Double"}, "outputs": {"y": "Double"}, "metadata": {...}}
/// ```
@PyFunction
func model_description(path: String) throws -> [String: String] {
    let url = URL(fileURLWithPath: path)
    let model: MLModel
    do {
        model = try MLModel(contentsOf: url)
    } catch {
        throw CoreMLBridgeError.modelLoadFailed("\(error)")
    }

    var result: [String: String] = [:]

    // Collect input names
    let inputNames = model.modelDescription.inputDescriptionsByName.keys.sorted()
    result["inputs"] = inputNames.joined(separator: ", ")

    // Collect output names
    let outputNames = model.modelDescription.outputDescriptionsByName.keys.sorted()
    result["outputs"] = outputNames.joined(separator: ", ")

    // Metadata
    if let metadata = model.modelDescription.metadata[.description] as? String {
        result["description"] = metadata
    }
    if let author = model.modelDescription.metadata[.author] as? String {
        result["author"] = author
    }
    if let version = model.modelDescription.metadata[.versionString] as? String {
        result["version"] = version
    }

    return result
}

// MARK: - predict

/// Run prediction on a CoreML model with a dictionary of input values.
/// Input values should be Double arrays. Returns output as dictionary of arrays.
///
/// ```python
/// result = pycoreml.predict("/path/to/model.mlmodelc", {"x": 42.0})
/// ```
@PyFunction
func predict(path: String, inputs: [String: Double]) throws -> [String: Double] {
    let url = URL(fileURLWithPath: path)
    let model: MLModel
    do {
        model = try MLModel(contentsOf: url)
    } catch {
        throw CoreMLBridgeError.modelLoadFailed("\(error)")
    }

    // Build feature provider from inputs
    let featureDict = try inputs.mapValues { value -> MLFeatureValue in
        MLFeatureValue(double: value)
    }
    let provider = try MLDictionaryFeatureProvider(dictionary: featureDict)

    // Run prediction
    let prediction: MLFeatureProvider
    do {
        prediction = try model.prediction(from: provider)
    } catch {
        throw CoreMLBridgeError.predictionFailed("\(error)")
    }

    // Extract output values
    var result: [String: Double] = [:]
    for name in prediction.featureNames {
        if let value = prediction.featureValue(for: name) {
            switch value.type {
            case .double:
                result[name] = value.doubleValue
            case .int64:
                result[name] = Double(value.int64Value)
            default:
                result[name] = 0.0  // Unsupported type
            }
        }
    }

    return result
}

// MARK: - list_compute_units

/// List available compute units on this device.
///
/// ```python
/// units = pycoreml.list_compute_units()
/// # ["cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly", "all"]
/// ```
@PyFunction
func list_compute_units() -> [String] {
    return ["all", "cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine"]
}

// MARK: - predict_with_options

/// Run prediction with explicit compute unit selection.
///
/// ```python
/// result = pycoreml.predict_with_options(
///     "/path/to/model.mlmodelc",
///     {"x": 42.0},
///     "cpuAndNeuralEngine"
/// )
/// ```
@PyFunction
func predict_with_options(path: String, inputs: [String: Double], compute_unit: String = "all") throws -> [String: Double] {
    let url = URL(fileURLWithPath: path)

    // Map compute unit string to MLComputeUnits
    let config = MLModelConfiguration()
    switch compute_unit {
    case "cpuOnly":
        config.computeUnits = .cpuOnly
    case "cpuAndGPU":
        config.computeUnits = .cpuAndGPU
    case "cpuAndNeuralEngine":
        config.computeUnits = .cpuAndNeuralEngine
    default:
        config.computeUnits = .all
    }

    let model: MLModel
    do {
        model = try MLModel(contentsOf: url, configuration: config)
    } catch {
        throw CoreMLBridgeError.modelLoadFailed("\(error)")
    }

    let featureDict = try inputs.mapValues { value -> MLFeatureValue in
        MLFeatureValue(double: value)
    }
    let provider = try MLDictionaryFeatureProvider(dictionary: featureDict)

    let prediction: MLFeatureProvider
    do {
        prediction = try model.prediction(from: provider)
    } catch {
        throw CoreMLBridgeError.predictionFailed("\(error)")
    }

    var result: [String: Double] = [:]
    for name in prediction.featureNames {
        if let value = prediction.featureValue(for: name) {
            switch value.type {
            case .double:
                result[name] = value.doubleValue
            case .int64:
                result[name] = Double(value.int64Value)
            default:
                result[name] = 0.0
            }
        }
    }

    return result
}

// MARK: - Module Entry Point

@PyModule("pycoreml", functions: [
    load_model,
    model_description,
    predict,
    list_compute_units,
    predict_with_options,
])
func pycoreml() {}
