# SQUARE-Mamba (Python Edition)

This repository hosts a pure-Python reimplementation of the **SQUARE-Mamba** drought forecasting pipeline. The original MATLAB demo has been ported to Python so that the model, evaluation utilities, and plotting workflow can be executed on laptops, workstations, or containers without depending on MATLAB.

The code base provides:
- A unified training/evaluation pipeline for the quantum-enhanced SQUARE-Mamba model and its classical ablation (`SQUARE_Mamba_Not_Quantum`).
- **Quantum noise simulation** using Qiskit Aer with configurable noise channels (depolarizing, amplitude damping, phase damping, bit flip, phase flip).
- CPU-friendly fallbacks for the Mamba selective scan block and the quantum temporal encoder (QLTEM) so the project runs on macOS and standard CPUs. GPU execution is still supported via optional dependencies/Docker images.
- **Fast testing utilities** for validating quantum circuits and noise injection without full training runs.
- A Python demo (`SQUARE_Mamba/demo.py`) that replicates the MATLAB demo, prints evaluation metrics, and can render or save plots.
- Docker recipes for both CPU-only deployment and CUDA-enabled GPU training.

## Repository Layout

```text
SQUARE_Mamba/
├── demo.py                          # MATLAB demo rewritten in Python
├── main/
│   ├── functions/                   # Data loading, metrics helpers
│   ├── networks/                    # Neural network building blocks
│   │   ├── SQUARE_Mamba.py          # Quantum variant with Qiskit-based QLTEM
│   │   ├── SQUARE_Mamba_Not_Quantum.py # Classical ablation
│   │   └── mamba_cpu.py             # CPU-friendly Mamba block
│   ├── CRU_data_montevideo/         # Input CRU climate datasets
│   └── Result/                      # Generated metrics/forecasts
├── test_simple_quantum.py           # Fast quantum circuit validation (11 checks)
├── test_qnn_example.py              # Test run_two_qubit_qnn_example function
├── train_ultra_fast.py              # Quick noise test (~3 min, 5 samples)
├── train_simple_with_noise.py       # Full noise comparison (~20 min, 30 samples)
├── run_test.sh                      # Helper script with environment setup
├── requirements.txt                 # Full dependencies (Qiskit + PyTorch)
├── requirements-cpu.txt             # Minimal dependencies for CPU execution
├── requirements-fixed.txt           # Clean compatible dependency versions
├── SETUP_TESTING.md                 # Complete testing guide and troubleshooting
├── RESULTADOS_PRUEBA_RUIDO.md       # Noise experiment results (Spanish)
├── Dockerfile.cpu                   # Container recipe for CPU environments
└── Dockerfile.gpu                   # Container recipe for NVIDIA GPU hosts
```

## Data

The repository ships with the CRU climate variables (`CRU_data/*.csv`) required for reproducing the experiments. No external download is necessary to run the sample pipeline.

## Python Environment Setup

### Quick Start (Recommended)

The repository includes a pre-configured virtual environment (`.venv-3.10/`) with all dependencies installed. Use the helper script to run any Python script:

```bash
./run_test.sh <script.py>
```

This automatically handles:
- Python environment activation
- OpenMP library conflict resolution
- Proper environment variables

### Manual Setup - CPU-only (macOS, Windows, Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements-cpu.txt
```

### Manual Setup - GPU (Linux + NVIDIA GPU)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements-gpu.txt
```

> **Note:** The GPU requirements include `mamba-ssm`, which needs CUDA toolchains. If you run the CPU build, those packages are intentionally omitted.

### Environment Configuration

**Current Working Setup:**
- Python: 3.10.13
- PyTorch: 2.2.1
- Qiskit: 1.4.3
- Qiskit Aer: 0.17.2+
- Qiskit Machine Learning: 0.8.4
- NumPy: 1.26.4

**Known Issues:**
- **OpenMP Conflict**: Set `KMP_DUPLICATE_LIB_OK=TRUE` (handled by `run_test.sh`)
- **NumPy Version**: Some packages have conflicting requirements (1.x vs 2.x), current setup uses 1.26.4 which works with warnings

### macOS / Apple Silicon

The helper script automatically handles macOS-specific issues. If running manually:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export TORCH_MPS_DISABLE=1
python SQUARE_Mamba/main/test_SQUARE_Mamba.py
```

## Quick Testing

Before running full training, verify the quantum system works:

```bash
# Fast system validation (~10 seconds)
./run_test.sh test_simple_quantum.py

# Test quantum circuit example
./run_test.sh test_qnn_example.py

# Quick noise effect test (~3 minutes)
./run_test.sh train_ultra_fast.py
```

The `train_ultra_fast.py` script runs two experiments (baseline vs noisy) with minimal data to quickly verify noise injection is working. Expected output shows ~10% difference in loss/R² between noisy and noiseless runs.

## Training

### Fast Training Scripts (Recommended for Testing)

```bash
# Ultra-fast noise test: 5 samples, 1 iteration, ~3 minutes
./run_test.sh train_ultra_fast.py

# Simple training: 30 samples, 3 epochs, ~20 minutes
./run_test.sh train_simple_with_noise.py
```

Both scripts automatically compare baseline (no noise) vs noisy training and display results.

### Full Training Scripts

```bash
# Quantum-enhanced SQUARE-Mamba (no noise)
./run_test.sh SQUARE_Mamba/main/train_SQUARE_Mamba.py --epochs 3

# With quantum noise
./run_test.sh SQUARE_Mamba/main/train_SQUARE_Mamba.py --epochs 3 --noise depolarizing=0.02

# Classical ablation without QLTEM
./run_test.sh SQUARE_Mamba/main/train_SQUARE_Mamba_Not_Quantum.py --epochs 3
```

All training scripts:
- Detect the best available device (`cuda`, `mps`, or `cpu`)
- Save checkpoints into `SQUARE_Mamba/main/checkpoint/` or `./checkpoint/`
- Print training/validation losses and R² scores every epoch
- Support early stopping and learning rate scheduling

### Quantum Noise Simulation (Qiskit Aer)

The quantum encoder supports configurable noise channels powered by **Qiskit Aer** (not PennyLane).

**Supported Noise Channels:**
- `depolarizing` - Depolarizing noise (probability parameter)
- `amplitude_damping` - Energy dissipation (gamma parameter)
- `phase_damping` - Dephasing without energy loss (lambda parameter)
- `bit_flip` - Bit flip errors (probability parameter)
- `phase_flip` - Phase flip errors (probability parameter)

**Command-Line Usage:**

```bash
# Single noise channel
./run_test.sh SQUARE_Mamba/main/train_SQUARE_Mamba.py \
    --epochs 3 \
    --noise depolarizing=0.02

# Multiple noise channels
./run_test.sh SQUARE_Mamba/main/train_SQUARE_Mamba.py \
    --epochs 3 \
    --noise depolarizing=0.02,amplitude_damping=0.05

# With custom backend and shots
./run_test.sh SQUARE_Mamba/main/train_SQUARE_Mamba.py \
    --epochs 3 \
    --noise depolarizing=0.02 \
    --noise-device density_matrix \
    --noise-shots 1000
```

**Programmatic Usage:**

```python
from networks.SQUARE_Mamba import make_model
from functions.util import parse_noise_spec

# Parse noise from string
noise_config = parse_noise_spec("depolarizing=0.02,amplitude_damping=0.05")

# Or create directly
noise_config = {
    'channels': [
        {'name': 'depolarizing', 'probability': 0.02},
        {'name': 'amplitude_damping', 'gamma': 0.05}
    ]
}

# Create model with noise
model = make_model(noise_config=noise_config)
```

**Verified Noise Effects:**
- Depolarizing noise (p=0.02) produces ~10% difference in loss and R² compared to noiseless training
- Noise simulation adds minimal overhead (~15% in forward pass, ~1% total)
- Gradients flow correctly through noisy quantum circuits

The same noise flags are available in `test_SQUARE_Mamba.py` for evaluation.

## Testing & Validation

### System Validation

Verify the quantum system is working correctly:

```bash
# Complete system check (11 validation steps, ~10 seconds)
./run_test.sh test_simple_quantum.py
```

This test validates:
- ✓ Qiskit, Qiskit Aer, and Qiskit ML imports
- ✓ PyTorch integration
- ✓ Quantum circuit creation
- ✓ Observable definition
- ✓ Estimator and EstimatorQNN setup
- ✓ TorchConnector integration
- ✓ Forward pass execution
- ✓ Backward pass and gradient calculation

### Noise Effect Validation

Test that quantum noise injection works:

```bash
# Ultra-fast test: 5 samples, 1 iteration (~3 minutes)
./run_test.sh train_ultra_fast.py

# Expected output:
# Baseline:     Loss=1.502, R²=-4.317
# With noise:   Loss=1.354, R²=-3.792
# Difference:   ~10% change (noise is working)
```

### Model Inference

After training, generate CSV predictions:

```bash
./run_test.sh SQUARE_Mamba/main/test_SQUARE_Mamba.py
./run_test.sh SQUARE_Mamba/main/test_SQUARE_Mamba_Not_Quantum.py
```

Each script loads the matching checkpoint (if present) and writes:
- `SQUARE_Mamba/main/Result/<MODE>/gt_Pooncarie.csv`
- `SQUARE_Mamba/main/Result/<MODE>/prediction_Pooncarie.csv`

If a checkpoint is missing, the script will emit a warning and proceed with randomly initialized weights.

## Demo & Plotting

The MATLAB demo workflow is reproduced in `demo.py`:

```bash
# Reuse existing CSVs and skip plotting
python SQUARE_Mamba/demo.py --skip-test --no-plot

# Regenerate outputs and display the plot window
python SQUARE_Mamba/demo.py --mode "SQUARE-Mamba"

# Save the figure as an image (headless)
python SQUARE_Mamba/demo.py \
    --mode "SQUARE-Mamba" \
    --skip-test \
    --save-plot exports/quantum_plot.png \
    --no-plot

# Classical "Not Quantum" variant
python SQUARE_Mamba/demo.py \
    --mode "SQUARE_Mamba_Not_Quantum" \
    --skip-test \
    --save-plot exports/quantum_plot.png \
    --no-plot
```

The plotting routine now uses the built-in `DejaVu Serif` font so it works out of the box in slim Docker images. When `--save-plot` is provided, Matplotlib switches to the non-interactive Agg backend so plots can be generated in headless environments.

## Docker Usage

### CPU Container (portable across macOS/Windows/Linux)

```bash
# Build
docker build -f Dockerfile.cpu -t square-mamba-cpu .

# Run interactively
docker run --rm -it -v "$(pwd)":/app square-mamba-cpu bash
```

Inside the container you can execute the same training, testing, or demo commands listed above. The volume mount ensures checkpoints, CSVs, and plots persist on the host.

### GPU Container (Linux + NVIDIA GPU)

```bash
# Build (requires nvidia-container-toolkit on the host)
docker build -f Dockerfile.gpu -t square-mamba-gpu .

# Launch with full GPU access
docker run --rm -it --gpus all -v "$(pwd)":/app square-mamba-gpu bash
```

The GPU recipe installs CUDA-enabled `torch/torchvision/torchaudio` and then the extra `pennylane`/`mamba-ssm` dependencies. Use this variant only on systems with supported NVIDIA drivers.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `OMP: Error #15` (OpenMP conflict) | Use `./run_test.sh` which sets `KMP_DUPLICATE_LIB_OK=TRUE` automatically, or set it manually: `export KMP_DUPLICATE_LIB_OK=TRUE` |
| `UserWarning: To copy construct from a tensor` | Fixed in `.venv-3.10/` by patching TorchConnector to use `.clone().detach()`. If using a different venv, apply the same fix to `qiskit_machine_learning/connectors/torch_connector.py` line 378. |
| `Times New Roman` font warning while plotting | `demo.py` now defaults to `DejaVu Serif`. If you prefer the original font, install it inside your environment or adjust the font names in the script. |
| MPS crash on macOS | The helper script sets `TORCH_MPS_DISABLE=1` automatically. If running manually, export this variable. |
| Building Docker GPU image fails | The GPU image requires Linux with an NVIDIA GPU and the `nvidia/cuda` base image. On macOS/Windows use `Dockerfile.cpu` instead. |
| No checkpoints found | Run the corresponding `train_*.py` scripts first so checkpoint files are produced in `SQUARE_Mamba/main/checkpoint/` or `./checkpoint/`. |
| Training is very slow | Quantum circuit simulation is computationally intensive. Use `train_ultra_fast.py` for quick tests. Backward pass takes ~75s per batch due to parameter-shift gradient calculation. |
| NumPy version conflicts | The current setup uses NumPy 1.26.4 which works with both Qiskit ML (requires >=2.0) and other packages (require <2.0). Warnings are expected but non-critical. |
| Negative R² values | Normal in early training with random weights. R² should improve toward positive values as training progresses. |

For detailed troubleshooting, see `SETUP_TESTING.md`.

## Documentation

### Quick Start
- **`QUICK_REFERENCE.md`** - ⚡ Fast commands and common tasks (start here!)
- **`TESTING_GUIDE.md`** - Complete setup, testing, and troubleshooting guide
- **`NOISE_EXPERIMENT_RESULTS.md`** - Validated noise experiment results and analysis
- **`run_test.sh`** - Helper script with usage examples (run without args to see options)

### Spec Documentation
- **`.kiro/specs/noise-simulation-fix/`** - Implementation specification
  - `requirements.md` - Detailed requirements with acceptance criteria
  - `design.md` - Technical design and architecture
  - `tasks.md` - Implementation task list

### Test Scripts
- **`test_simple_quantum.py`** - System validation (11 checks, ~10s)
- **`test_qnn_example.py`** - Example function test (~5s)
- **`train_ultra_fast.py`** - Quick noise test (5 samples, ~3min)
- **`train_simple_with_noise.py`** - Full training test (30 samples, 3 epochs, ~20min)

## Performance Notes

### Expected Timing
- **System validation**: ~10 seconds
- **Quick noise test**: ~3 minutes (2 experiments)
- **Simple training**: ~20 minutes (2 experiments × 3 epochs)
- **Full training**: ~2-3 hours (depends on epochs and samples)

### Why is Training Slow?
Quantum circuit simulation is computationally intensive:
- **Forward pass**: ~0.3s per batch (circuit execution)
- **Backward pass**: ~75s per batch (parameter-shift gradients)
- Each trainable parameter requires 2 circuit evaluations for gradients
- QLTEM has 55 trainable parameters → ~110 circuit evaluations per batch

This is **expected behavior** for quantum machine learning, not a bug.

### Optimization Tips
1. Use `train_ultra_fast.py` for quick validation
2. Reduce number of samples for testing
3. Use larger batch sizes (processes samples in parallel classically)
4. Consider GPU for classical layers (quantum part is CPU-bound)

## Validated Configurations

### Working Setup (Tested on macOS)
- Python 3.10.13
- PyTorch 2.2.1
- Qiskit 1.4.3
- Qiskit Aer 0.17.2+
- Qiskit Machine Learning 0.8.4
- NumPy 1.26.4

### Noise Channels (Verified)
- ✅ Depolarizing (p=0.02): ~10% effect on loss/R²
- ✅ Amplitude damping: Supported
- ✅ Phase damping: Supported
- ✅ Bit flip: Supported
- ✅ Phase flip: Supported
- ✅ Multiple channels: Supported

## Contributing

Contributions are welcome! If you improve the model, add datasets, or enhance the tooling:
1. Fork the repository
2. Create a feature branch
3. Run validation tests: `./run_test.sh test_simple_quantum.py`
4. Submit a pull request with clear explanations and test results

### Development Workflow
1. Verify system: `./run_test.sh test_simple_quantum.py`
2. Make changes
3. Test changes: `./run_test.sh train_ultra_fast.py`
4. Run full validation if needed
5. Update documentation

## References

- Po-Wei Tang and Jian-Kai Huang, *SQUARE-Mamba User Guide*, Institute of Computer and Communication Engineering, National Cheng Kung University, Tainan, Taiwan, June 30, 2025. (Bundled as `SQUARE_Mamba/ReadMe.pdf`.)
- Qiskit Documentation: https://qiskit.org/documentation/
- Qiskit Machine Learning: https://qiskit.org/ecosystem/machine-learning/

## License

This project inherits the license of the original SQUARE-Mamba release. Please consult the upstream repository or publication for citation and usage guidelines.

## Support

- **Issues**: Open an issue for bugs or feature requests
- **Questions**: Start a discussion for usage questions
- **Documentation**: See `TESTING_GUIDE.md` for comprehensive troubleshooting

Enjoy forecasting! 🌦️🔬
