# Design Document

## Overview

This design addresses the noise simulation issues in SQUARE_Mamba by fixing the noise injection mechanism, ensuring proper gradient flow, and implementing a robust configuration system. The solution involves:

1. **Root Cause Analysis**: The current implementation has duplicate parameter passing and lacks proper noise model integration with the Qiskit Aer Estimator
2. **Centralized Noise Management**: Enhanced NoiseManager class with validation, logging, and proper Qiskit NoiseModel construction
3. **Configuration Pipeline**: Streamlined parameter flow from CLI → training script → model → quantum layers
4. **Reproducibility**: Seed management across PyTorch, NumPy, and Qiskit simulators
5. **Code Optimization**: Consolidation of duplicate code, removal of dead code, and extraction of reusable utilities

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Pipeline                        │
│  (train_SQUARE_Mamba.py)                                    │
│                                                              │
│  ┌──────────────┐    ┌─────────────────┐                   │
│  │ CLI Parser   │───▶│ Noise Config    │                   │
│  │ --noise      │    │ Parser          │                   │
│  │ --noise-seed │    │ (parse_noise_   │                   │
│  │ --noise-shots│    │  spec)          │                   │
│  └──────────────┘    └────────┬────────┘                   │
│                               │                             │
│                               ▼                             │
│                      ┌─────────────────┐                    │
│                      │ Model Factory   │                    │
│                      │ (make_model)    │                    │
│                      └────────┬────────┘                    │
└───────────────────────────────┼─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    SQUARE_Mamba Model                        │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ SEB          │    │ LTEM         │    │ QLTEM        │ │
│  │ (Spatial)    │    │ (Classical   │    │ (Quantum)    │ │
│  │              │    │  Mamba)      │    │              │ │
│  └──────────────┘    └──────────────┘    └──────┬───────┘ │
│                                                   │         │
│                                                   ▼         │
│                                          ┌─────────────────┐│
│                                          │ NoiseManager    ││
│                                          │ - Validation    ││
│                                          │ - NoiseModel    ││
│                                          │ - Logging       ││
│                                          └────────┬────────┘│
│                                                   │         │
│                                                   ▼         │
│                                          ┌─────────────────┐│
│                                          │ Qiskit Aer      ││
│                                          │ Estimator       ││
│                                          │ + NoiseModel    ││
│                                          └─────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Configuration Phase**:
   - CLI arguments → `parse_noise_spec()` → structured dict
   - Validation and normalization in NoiseManager
   - Qiskit NoiseModel construction

2. **Training Phase**:
   - Input data → SEB (spatial encoding) → feature maps
   - Feature maps → LTEM (classical) + QLTEM (quantum) → combined features
   - QLTEM applies noise via Qiskit Aer Estimator
   - Gradients flow back through TorchConnector

3. **Noise Injection**:
   - NoiseModel attached to Estimator options
   - Applied during quantum circuit execution
   - Affects expectation values and gradients

## Components and Interfaces

### 1. NoiseManager (Enhanced)

**Location**: `SQUARE_Mamba/main/networks/SQUARE_Mamba.py`

**Responsibilities**:
- Parse and validate noise channel configurations
- Construct Qiskit NoiseModel with proper gate targeting
- Provide logging for debugging noise application
- Support seed configuration for reproducibility

**Interface**:
```python
class NoiseManager:
    def __init__(self, config: dict | list | None, seed: int | None = None):
        """
        Args:
            config: Noise specification (dict with 'channels' key or list of channels)
            seed: Random seed for reproducible noise simulation
        """
        
    @property
    def enabled(self) -> bool:
        """Returns True if any noise channels are configured"""
        
    def get_noise_model(self) -> NoiseModel | None:
        """Returns the constructed Qiskit NoiseModel"""
        
    def get_config_summary(self) -> dict:
        """Returns a summary of the noise configuration for logging"""
```

**Key Changes**:
- Add seed parameter and pass to NoiseModel
- Add logging of noise configuration at initialization
- Add validation error messages with specific guidance
- Add `get_config_summary()` for experiment tracking

### 2. Noise Configuration Parser

**Location**: `SQUARE_Mamba/main/functions/util.py`

**Current Function**: `parse_noise_spec(spec: str | None) -> dict | None`

**Enhancement**:
```python
def parse_noise_spec(spec: str | None) -> dict | None:
    """
    Parse comma-separated noise specification string.
    
    Args:
        spec: String like "depolarizing=0.02,amplitude_damping=0.05"
              or None for no noise
    
    Returns:
        Dictionary with 'channels' list, or None if no noise
        
    Example:
        >>> parse_noise_spec("depolarizing=0.02")
        {'channels': [{'name': 'depolarizing', 'probability': 0.02}]}
    """
```

**Key Changes**:
- Add comprehensive docstring with examples
- Add validation for parameter ranges
- Log warnings for invalid specifications
- Return None for empty/invalid specs

### 3. Training Pipeline

**Location**: `SQUARE_Mamba/main/train_SQUARE_Mamba.py`

**New CLI Arguments**:
```python
parser.add_argument('--noise-seed', type=int, default=None,
                    help='Random seed for reproducible noise simulation')
parser.add_argument('--log-noise-config', action='store_true',
                    help='Log detailed noise configuration at startup')
```

**Initialization Sequence**:
```python
# 1. Set all seeds
seed = opt.noise_seed if opt.noise_seed is not None else 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 2. Parse noise configuration
noise_config = parse_noise_spec(opt.noise)
if noise_config is not None:
    noise_config['seed'] = seed
    if opt.noise_device is not None:
        noise_config['device'] = opt.noise_device
    if opt.noise_shots is not None:
        noise_config['shots'] = opt.noise_shots

# 3. Create model with noise config
model = Net.make_model(
    noise_config=noise_config,
    quantum_device=opt.noise_device,
    shots=opt.noise_shots,
).to(device)

# 4. Log configuration if requested
if opt.log_noise_config and noise_config:
    print(f"Noise configuration: {model.QLTEM.noise_manager.get_config_summary()}")
```

### 4. Estimator Factory

**Location**: `SQUARE_Mamba/main/networks/SQUARE_Mamba.py`

**Current Function**: `create_estimator(noise_manager, override_device, shots)`

**Enhancement**:
```python
def create_estimator(
    noise_manager: NoiseManager,
    override_device: str | None = None,
    shots: int | None = None,
    seed: int | None = None
) -> AerEstimator:
    """
    Create Qiskit Aer Estimator with noise model and configuration.
    
    Args:
        noise_manager: NoiseManager instance with noise configuration
        override_device: Optional backend method override
        shots: Number of shots for stochastic simulation
        seed: Random seed for reproducibility
        
    Returns:
        Configured AerEstimator instance
    """
    estimator = AerEstimator()
    
    # Set simulation method
    method = _resolve_method(override_device or noise_manager.device_name)
    if method is not None:
        estimator.options.method = method
    
    # Attach noise model
    noise_model = noise_manager.get_noise_model()
    if noise_model is not None:
        estimator.options.noise_model = noise_model
        # Use density matrix method when noise is present
        if method is None:
            estimator.options.method = "density_matrix"
    
    # Set shots
    resolved_shots = shots if shots is not None else noise_manager.shots
    if resolved_shots is not None:
        estimator.options.default_shots = int(resolved_shots)
    
    # Set seed for reproducibility
    if seed is not None:
        estimator.options.seed_simulator = seed
    
    return estimator
```

**Key Changes**:
- Add seed parameter and set `seed_simulator` option
- Automatically use density_matrix method when noise is enabled
- Add docstring explaining configuration options

### 5. QLTEM Module

**Location**: `SQUARE_Mamba/main/networks/SQUARE_Mamba.py`

**Current Class**: `QLTEM(nn.Module)`

**Enhancement**:
```python
class QLTEM(nn.Module):
    NUM_QUBITS = 3
    NUM_GROUPS = 5
    OBSERVABLES = [
        SparsePauliOp.from_list([("ZII", 1.0)]),
        SparsePauliOp.from_list([("IZI", 1.0)]),
        SparsePauliOp.from_list([("IIZ", 1.0)]),
    ]

    def __init__(self, noise_config=None, quantum_device=None, shots=None, seed=None):
        super(QLTEM, self).__init__()
        
        self.noise_manager = NoiseManager(noise_config, seed=seed)
        self.estimator = create_estimator(
            self.noise_manager,
            override_device=quantum_device,
            shots=shots,
            seed=seed
        )
        
        # Log noise configuration if enabled
        if self.noise_manager.enabled:
            config_summary = self.noise_manager.get_config_summary()
            print(f"QLTEM initialized with noise: {config_summary}")
        
        self.temporal_blocks = nn.ModuleList([
            self._build_temporal_block() for _ in range(self.NUM_GROUPS)
        ])
```

**Key Changes**:
- Add seed parameter
- Pass seed to NoiseManager and create_estimator
- Add initialization logging for noise configuration

## Data Models

### Noise Configuration Schema

```python
NoiseConfig = {
    "channels": [
        {
            "name": str,  # One of: depolarizing, amplitude_damping, phase_damping, bit_flip, phase_flip
            "probability": float,  # For depolarizing, bit_flip, phase_flip (0.0-1.0)
            "gamma": float,  # For amplitude_damping (0.0-1.0)
            "lam": float,  # For phase_damping (0.0-1.0)
        }
    ],
    "device": str | None,  # Optional: backend method override
    "shots": int | None,  # Optional: number of shots for stochastic simulation
    "seed": int | None,  # Optional: random seed for reproducibility
}
```

### CLI to Config Mapping

| CLI Argument | Config Key | Type | Default | Description |
|--------------|------------|------|---------|-------------|
| `--noise` | `channels` | list | None | Comma-separated noise specs |
| `--noise-device` | `device` | str | None | Backend method override |
| `--noise-shots` | `shots` | int | None | Number of shots |
| `--noise-seed` | `seed` | int | 42 | Random seed |

## Error Handling

### Validation Errors

1. **Invalid Noise Channel**:
   - **Detection**: Channel name not in SUPPORTED_CHANNELS
   - **Action**: Log warning, skip channel, continue
   - **Message**: "Noise channel '{name}' is not supported; skipping."

2. **Out-of-Range Parameter**:
   - **Detection**: Probability/gamma/lam outside [0.0, 1.0]
   - **Action**: Clamp to valid range, log warning, continue
   - **Message**: "Noise parameter {value} clamped to [{min}, {max}]"

3. **Malformed Specification**:
   - **Detection**: Missing '=' in CLI spec or non-numeric value
   - **Action**: Log warning, skip entry, continue
   - **Message**: "Noise specification '{entry}' is invalid; skipping."

### Runtime Errors

1. **Gradient Flow Issues**:
   - **Detection**: NaN or Inf in loss values
   - **Action**: Log error with noise config, suggest reducing noise or learning rate
   - **Recovery**: Gradient clipping (already in improved script)

2. **Estimator Failures**:
   - **Detection**: Exception during quantum circuit execution
   - **Action**: Log full noise config and circuit parameters
   - **Recovery**: Fallback to noiseless simulation with warning

3. **Checkpoint Loading with Noise Mismatch**:
   - **Detection**: Quantum layer weights missing or unexpected
   - **Action**: Log warning about reinitialization
   - **Recovery**: Continue with reinitialized quantum weights

## Testing Strategy

### Unit Tests

**File**: `tests/test_noise_manager.py`

1. **Test Noise Channel Parsing**:
   - Valid single channel
   - Multiple channels
   - Invalid channel names
   - Out-of-range parameters
   - Malformed specifications

2. **Test NoiseModel Construction**:
   - Verify gates are targeted correctly
   - Verify noise parameters match specification
   - Verify seed is applied

3. **Test Configuration Flow**:
   - CLI → parse_noise_spec → dict
   - Dict → NoiseManager → NoiseModel
   - NoiseModel → Estimator options

### Integration Tests

**File**: `tests/test_noise_training.py`

1. **Test Training with Noise**:
   - Run 3 epochs with depolarizing noise
   - Verify loss values differ from noiseless
   - Verify no NaN/Inf in gradients
   - Verify checkpoints save correctly

2. **Test Reproducibility**:
   - Run same config with same seed twice
   - Verify identical loss trajectories
   - Verify identical final weights

3. **Test Multiple Noise Channels**:
   - Combine depolarizing + amplitude_damping
   - Verify both are applied
   - Verify training completes

### Validation Tests

**File**: `tests/test_noise_validation.py`

1. **Test Noise Effect on Metrics**:
   - Train noiseless model for 10 epochs
   - Train noisy model (depolarizing=0.02) for 10 epochs
   - Compare final R² scores
   - Verify statistical difference (t-test)

2. **Test Noise Levels**:
   - Train with noise levels [0.01, 0.02, 0.05, 0.1]
   - Verify monotonic degradation in performance
   - Log results for documentation

## Code Optimization Plan

### 1. Consolidate Duplicate Code

**Target**: Data transformation and tensor reshaping

**Current State**: Repeated in multiple places
- `map_generation()` in SQUARE_Mamba.py
- `Create_dataset()` in util.py
- Tensor reshaping in QLTEM forward pass

**Refactoring**:
```python
# New file: SQUARE_Mamba/main/functions/tensor_utils.py

def reshape_spatial_features(tensor, batch_size, height, width, channels):
    """Reshape flat tensor to spatial grid format"""
    
def normalize_features(features, axis=0):
    """Normalize features with mean/std, handling zero std"""
    
def pad_spatial_grid(grid, method='nearest'):
    """Pad spatial grid using specified method"""
```

### 2. Remove Dead Code

**Targets**:
- Unused imports (warnings, os in some files)
- `apply()` method in NoiseManager (marked as backwards compatibility but never called)
- Duplicate parameter passing (`noise_config=noise_config` appears twice)

**Actions**:
- Remove unused imports
- Remove `apply()` method and add migration note in docstring
- Fix duplicate parameters in function calls

### 3. Extract Reusable Training Utilities

**Target**: Training and validation loops

**Current State**: Similar patterns in train_SQUARE_Mamba.py and train_SQUARE_Mamba_improved.py

**Refactoring**:
```python
# New file: SQUARE_Mamba/main/functions/training_utils.py

def train_epoch(model, dataloader, optimizer, loss_fn, device, clip_grad=None):
    """Execute one training epoch with optional gradient clipping"""
    
def validate_epoch(model, dataloader, loss_fn, metric_fn, device):
    """Execute one validation epoch and return metrics"""
    
def save_checkpoint(model, path, metadata=None):
    """Save model checkpoint with optional metadata"""
    
def load_checkpoint(model, path, strict=True):
    """Load model checkpoint with error handling"""
```

### 4. Align Dependencies

**Current Issues**:
- requirements.txt includes packages not used in CPU mode
- No version pinning for critical packages
- Missing comments explaining optional dependencies

**Actions**:
```
# requirements-cpu.txt (minimal)
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
einops>=0.7.0
tqdm>=4.65.0
qiskit>=1.0.0
qiskit-aer>=0.13.0
qiskit-machine-learning>=0.7.0

# requirements.txt (full, includes GPU support)
# Includes all from requirements-cpu.txt plus:
# Optional: Mamba SSM for GPU acceleration
mamba-ssm>=1.0.0  # Requires CUDA toolkit

# Optional: Additional quantum backends
# pennylane>=0.33.0  # Alternative quantum framework (not currently used)
```

## Implementation Notes

### Critical Fixes

1. **Remove Duplicate Parameters**: 
   - In `make_model()`, `QLTEM.__init__()`, and model instantiation
   - Currently: `noise_config=noise_config,` appears twice
   - Fix: Remove duplicate lines

2. **Fix Gradient Flow**:
   - Ensure TorchConnector properly wraps EstimatorQNN
   - Verify `input_gradients=True` is set
   - Add gradient clipping in training loop

3. **Fix Noise Model Application**:
   - Ensure NoiseModel is attached to Estimator before circuit execution
   - Verify density_matrix method is used when noise is enabled
   - Add logging to confirm noise is applied

### Performance Considerations

1. **Noise Simulation Overhead**:
   - Density matrix simulation: O(2^(2n)) memory for n qubits
   - For 3 qubits: 64 complex numbers vs 8 for statevector
   - Expect 2-3x slowdown with noise enabled

2. **Batch Processing**:
   - Quantum circuits process one sample at a time
   - Batch size affects classical layers only
   - Consider reducing batch size if memory is constrained

3. **Checkpoint Size**:
   - Quantum layer weights are small (11 parameters per block)
   - Total quantum weights: 5 blocks × 11 params = 55 parameters
   - Checkpoint size dominated by classical layers

### Migration Path

1. **Phase 1**: Fix critical bugs (duplicate parameters, noise model attachment)
2. **Phase 2**: Add seed support and logging
3. **Phase 3**: Refactor duplicate code
4. **Phase 4**: Add comprehensive tests
5. **Phase 5**: Update documentation

This phased approach allows incremental validation and reduces risk of introducing new bugs.
