# Implementation Plan

- [ ] 1. Fix critical bugs in noise configuration and model initialization
  - Remove duplicate `noise_config=noise_config` parameters in `make_model()`, `QLTEM.__init__()`, and model instantiation calls
  - Fix duplicate parameter in `test_SQUARE_Mamba.py` model initialization
  - Verify noise model is properly attached to Qiskit Aer Estimator before circuit execution
  - _Requirements: 1.5, 2.5_

- [ ] 2. Enhance NoiseManager with seed support and logging
- [ ] 2.1 Add seed parameter to NoiseManager class
  - Modify `NoiseManager.__init__()` to accept optional `seed` parameter
  - Store seed as instance variable for later use
  - _Requirements: 4.3, 4.5_

- [ ] 2.2 Implement noise configuration logging
  - Add `get_config_summary()` method to NoiseManager that returns a dict with channel names, parameters, and settings
  - Add initialization logging in `NoiseManager.__init__()` when noise is enabled
  - Log warnings with specific guidance when validation fails
  - _Requirements: 2.3, 2.4, 7.2_

- [ ] 2.3 Add seed support to Qiskit Estimator
  - Modify `create_estimator()` to accept optional `seed` parameter
  - Set `estimator.options.seed_simulator` when seed is provided
  - Automatically use density_matrix method when noise model is present
  - _Requirements: 4.2, 4.3_

- [ ] 3. Add seed configuration to training pipeline
- [ ] 3.1 Add CLI argument for noise seed
  - Add `--noise-seed` argument to training script argument parser
  - Add `--log-noise-config` flag for detailed noise logging
  - _Requirements: 3.4, 4.1_

- [ ] 3.2 Implement seed propagation through model initialization
  - Set PyTorch seed using `torch.manual_seed()`
  - Set NumPy seed using `np.random.seed()`
  - Set CUDA seed using `torch.cuda.manual_seed_all()` when available
  - Pass seed to noise_config dict before model creation
  - Pass seed to `make_model()` and propagate to QLTEM
  - _Requirements: 4.2, 4.4, 4.5_

- [ ] 3.3 Add noise configuration logging at training startup
  - Log noise configuration summary after model initialization when `--log-noise-config` is set
  - Log seed value used for the experiment
  - _Requirements: 4.5, 7.3_

- [ ] 4. Update QLTEM module to support seed and improve initialization
- [ ] 4.1 Add seed parameter to QLTEM constructor
  - Modify `QLTEM.__init__()` signature to accept `seed` parameter
  - Pass seed to NoiseManager and create_estimator
  - _Requirements: 4.3_

- [ ] 4.2 Add initialization logging for noise configuration
  - Log noise configuration summary when QLTEM is initialized with noise enabled
  - Include channel types and parameters in log message
  - _Requirements: 7.2, 7.3_

- [ ] 5. Enhance parse_noise_spec with validation and documentation
- [ ] 5.1 Add comprehensive docstring to parse_noise_spec
  - Document expected input format with examples
  - Document return value structure
  - Add usage examples in docstring
  - _Requirements: 7.2, 7.3_

- [ ] 5.2 Add parameter range validation
  - Validate that probability values are in [0.0, 1.0] range
  - Log warnings when values are clamped
  - Add validation for numeric conversion errors
  - _Requirements: 2.4, 3.5_

- [ ] 6. Create shared utility modules for code consolidation
- [ ] 6.1 Create tensor_utils.py module
  - Extract `reshape_spatial_features()` function for tensor reshaping operations
  - Extract `normalize_features()` function with zero-std handling
  - Extract `pad_spatial_grid()` function for spatial padding
  - _Requirements: 5.1, 5.4_

- [ ] 6.2 Create training_utils.py module
  - Extract `train_epoch()` function with gradient clipping support
  - Extract `validate_epoch()` function with metric computation
  - Extract `save_checkpoint()` function with metadata support
  - Extract `load_checkpoint()` function with error handling
  - _Requirements: 5.1, 5.3, 5.4_

- [ ] 6.3 Update existing code to use shared utilities
  - Refactor train_SQUARE_Mamba.py to use training_utils functions
  - Refactor SQUARE_Mamba.py to use tensor_utils functions
  - Remove duplicate code from original locations
  - _Requirements: 5.1, 5.2, 5.4_

- [ ] 7. Remove dead code and unused imports
- [ ] 7.1 Remove unused imports across all modules
  - Audit and remove unused imports from train_SQUARE_Mamba.py
  - Audit and remove unused imports from SQUARE_Mamba.py
  - Audit and remove unused imports from util.py
  - _Requirements: 5.2_

- [ ] 7.2 Remove backwards compatibility code
  - Remove `apply()` method from NoiseManager class
  - Add migration note in NoiseManager docstring explaining removal
  - _Requirements: 5.2_

- [ ] 8. Validate noise injection with training experiments
- [ ] 8.1 Run baseline training without noise
  - Execute training for 3 epochs with default configuration
  - Record training loss and validation R² for each epoch
  - Save checkpoint and log output
  - _Requirements: 1.1, 1.3_

- [ ] 8.2 Run training with depolarizing noise
  - Execute training for 3 epochs with `--noise depolarizing=0.02`
  - Record training loss and validation R² for each epoch
  - Compare metrics with baseline to verify noise effect
  - _Requirements: 1.1, 1.2, 1.4_

- [ ] 8.3 Verify reproducibility with seed
  - Run same noisy configuration twice with same seed
  - Verify loss values match across runs
  - Document seed value and results
  - _Requirements: 4.2_

- [ ] 8.4 Test multiple noise channels
  - Run training with `--noise depolarizing=0.02,amplitude_damping=0.05`
  - Verify training completes without errors
  - Record metrics and compare with single-channel noise
  - _Requirements: 1.1, 2.5_

- [ ] 9. Update dependency specifications
- [ ] 9.1 Review and update requirements-cpu.txt
  - List only dependencies used in CPU-only execution
  - Add version constraints for critical packages
  - Add comments explaining each dependency
  - _Requirements: 6.1, 6.2, 6.5_

- [ ] 9.2 Review and update requirements.txt
  - Include all dependencies from requirements-cpu.txt
  - Add optional GPU dependencies with comments
  - Specify compatible version ranges
  - Remove unused dependencies
  - _Requirements: 6.1, 6.3, 6.4_

- [ ] 10. Create noise experiments documentation
- [ ] 10.1 Create docs/noise_experiments.md file
  - Document the noise injection mechanism and architecture
  - List supported noise channel types with parameter descriptions
  - Provide example CLI commands for common scenarios
  - Explain how noise affects training metrics
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 10.2 Add validation results to documentation
  - Include results from validation experiments (baseline vs noisy)
  - Document the noise configuration used for validation
  - Show loss and R² comparisons in table format
  - Add interpretation of results
  - _Requirements: 7.5_

- [ ] 10.3 Update README.md with noise configuration section
  - Add section explaining noise simulation capabilities
  - Reference docs/noise_experiments.md for detailed information
  - Update example commands to include noise parameters
  - _Requirements: 7.3_

- [ ] 11. Add targeted code comments for clarity
- [ ] 11.1 Add comments to NoiseManager
  - Comment the noise model construction logic in `_build_noise_model()`
  - Explain gate targeting strategy for 1-qubit and 2-qubit gates
  - Document the error creation process in `_make_errors()`
  - _Requirements: 8.1_

- [ ] 11.2 Add comments to QLTEM forward pass
  - Explain the tensor reshaping operations in `_apply_block()`
  - Document the batch and feature dimension handling
  - Clarify the rearrange operations using einops
  - _Requirements: 8.2_

- [ ] 11.3 Add comments to training pipeline
  - Explain noise configuration parsing and validation flow
  - Document seed propagation through the initialization sequence
  - Clarify the model factory call with noise parameters
  - _Requirements: 8.3_

- [ ] 11.4 Add docstrings to public functions
  - Add docstrings to `create_estimator()` with parameter descriptions
  - Add docstrings to `parse_noise_spec()` with examples
  - Add docstrings to utility functions in tensor_utils and training_utils
  - _Requirements: 8.5_
