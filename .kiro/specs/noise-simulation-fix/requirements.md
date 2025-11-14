# Requirements Document

## Introduction

This document specifies the requirements for fixing and enhancing the quantum noise simulation in the SQUARE_Mamba drought forecasting model. The current implementation has issues where noise injection does not properly influence training beyond epoch 1, and the noise configuration lacks proper validation and documentation. The system must support configurable, reproducible noise injection that statistically matches experimental setups while maintaining code quality through optimization and cleanup.

## Glossary

- **SQUARE_Mamba**: The quantum-enhanced drought forecasting neural network model that combines classical Mamba blocks with quantum temporal encoding modules (QLTEM)
- **QLTEM**: Quantum Local Temporal Encoding Module - the quantum circuit component that processes temporal features using parameterized quantum circuits
- **NoiseManager**: The class responsible for parsing, validating, and applying quantum noise channels to the quantum circuits
- **Training Pipeline**: The end-to-end workflow including data loading, model initialization, training loop, validation, and checkpoint saving
- **Noise Channel**: A quantum error model (e.g., depolarizing, amplitude damping, phase damping) that simulates realistic quantum hardware imperfections
- **Qiskit Aer**: The quantum simulator backend used for executing noisy quantum circuits with the Estimator primitive
- **Noise Configuration**: A structured dictionary or command-line specification defining noise channel types, magnitudes, and simulation parameters

## Requirements

### Requirement 1

**User Story:** As a researcher, I want the noise injection to properly affect training across all epochs, so that I can study the model's robustness to quantum hardware noise.

#### Acceptance Criteria

1. WHEN the Training Pipeline executes with noise enabled, THE SQUARE_Mamba SHALL complete at least 3 training epochs without errors or stalls
2. WHEN noise is injected during training, THE SQUARE_Mamba SHALL produce measurably different loss values compared to noiseless training
3. WHEN the Training Pipeline logs metrics, THE SQUARE_Mamba SHALL record both training loss and validation R² scores for each epoch
4. WHEN comparing noisy versus noiseless runs, THE SQUARE_Mamba SHALL demonstrate statistically distinguishable performance metrics
5. THE SQUARE_Mamba SHALL maintain gradient flow through the quantum layers when noise is enabled

### Requirement 2

**User Story:** As a developer, I want a centralized noise injection utility with clear configuration options, so that noise behavior is consistent across training and evaluation.

#### Acceptance Criteria

1. THE NoiseManager SHALL accept noise configuration as a structured dictionary with channel specifications
2. THE NoiseManager SHALL validate all noise channel types against a defined set of supported channels (depolarizing, amplitude_damping, phase_damping, bit_flip, phase_flip)
3. WHEN an unsupported noise channel is specified, THE NoiseManager SHALL emit a warning and skip that channel
4. WHEN noise parameters are out of valid range, THE NoiseManager SHALL clamp values to [0.0, 1.0] and log a warning
5. THE NoiseManager SHALL construct a Qiskit NoiseModel that applies noise to all relevant quantum gates in the circuit

### Requirement 3

**User Story:** As a researcher, I want command-line parameters for noise configuration, so that I can easily run experiments with different noise profiles without modifying code.

#### Acceptance Criteria

1. THE Training Pipeline SHALL accept a `--noise` parameter with comma-separated noise specifications in format "channel_name=value"
2. THE Training Pipeline SHALL accept a `--noise-device` parameter to override the quantum simulator backend
3. THE Training Pipeline SHALL accept a `--noise-shots` parameter to enable shot-based stochastic sampling
4. THE Training Pipeline SHALL accept a `--noise-seed` parameter to ensure reproducible noise simulation
5. WHEN no noise parameters are provided, THE Training Pipeline SHALL execute with noiseless quantum simulation as default
6. THE Training Pipeline SHALL pass all noise configuration parameters to the model initialization function

### Requirement 4

**User Story:** As a researcher, I want reproducible noise experiments with configurable random seeds, so that I can verify results and compare experiments reliably.

#### Acceptance Criteria

1. THE Training Pipeline SHALL accept a seed parameter for controlling random number generation
2. WHEN a seed is provided, THE SQUARE_Mamba SHALL produce identical results across multiple runs with the same configuration
3. THE NoiseManager SHALL support seed configuration for the quantum simulator backend
4. THE Training Pipeline SHALL set seeds for PyTorch, NumPy, and the quantum simulator when a seed parameter is provided
5. THE Training Pipeline SHALL log the seed value used for each experiment run

### Requirement 5

**User Story:** As a developer, I want to eliminate code duplication and dead code, so that the codebase is maintainable and easier to understand.

#### Acceptance Criteria

1. THE SQUARE_Mamba codebase SHALL consolidate duplicate data transformation logic into shared utility functions
2. THE SQUARE_Mamba codebase SHALL remove unused imports and dead code paths from all modules
3. THE SQUARE_Mamba codebase SHALL extract repeated training and evaluation loop patterns into reusable helper functions
4. THE SQUARE_Mamba codebase SHALL move shared utilities from individual modules into a common utilities module
5. THE SQUARE_Mamba codebase SHALL maintain backward compatibility with existing checkpoint files and data formats

### Requirement 6

**User Story:** As a developer, I want aligned dependency specifications, so that the environment setup matches the actual code requirements.

#### Acceptance Criteria

1. THE requirements.txt file SHALL list only dependencies that are imported and used in the codebase
2. THE requirements-cpu.txt file SHALL contain the minimal dependency set for CPU-only execution
3. THE requirements.txt and requirements-cpu.txt files SHALL specify compatible version ranges for all packages
4. WHEN dependencies are updated, THE SQUARE_Mamba SHALL verify compatibility through automated testing
5. THE requirements files SHALL include comments explaining why optional dependencies are needed

### Requirement 7

**User Story:** As a researcher, I want clear documentation of noise experiment configurations and results, so that I can understand and reproduce the experiments.

#### Acceptance Criteria

1. THE SQUARE_Mamba repository SHALL include a noise experiments documentation file describing the noise injection mechanism
2. THE documentation SHALL specify the supported noise channel types and their parameter ranges
3. THE documentation SHALL provide example command-line invocations for common noise experiment scenarios
4. THE documentation SHALL explain how noise affects training metrics and model performance
5. THE documentation SHALL include validation results demonstrating noise effects on at least one experiment configuration

### Requirement 8

**User Story:** As a developer, I want targeted code comments where control flow is non-obvious, so that future maintainers can understand the implementation quickly.

#### Acceptance Criteria

1. THE NoiseManager SHALL include comments explaining the noise model construction process
2. THE QLTEM forward pass SHALL include comments describing the tensor reshaping operations
3. THE Training Pipeline SHALL include comments explaining the noise configuration parsing and validation
4. THE SQUARE_Mamba codebase SHALL avoid redundant comments that merely restate obvious code
5. THE SQUARE_Mamba codebase SHALL use docstrings for all public functions and classes describing parameters and return values
