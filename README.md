# SQUARE-Mamba (Python Edition)

This repository hosts a pure-Python reimplementation of the **SQUARE-Mamba** drought forecasting pipeline. The original MATLAB demo has been ported to Python so that the model, evaluation utilities, and plotting workflow can be executed on laptops, workstations, or containers without depending on MATLAB.

The code base provides:
- A unified training/evaluation pipeline for the quantum-enhanced SQUARE-Mamba model and its classical ablation (`SQUARE_Mamba_Not_Quantum`).
- CPU-friendly fallbacks for the Mamba selective scan block and the quantum temporal encoder (QLTEM) so the project runs on macOS and standard CPUs. GPU execution is still supported via optional dependencies/Docker images.
- A Python demo (`SQUARE_Mamba/demo.py`) that replicates the MATLAB demo, prints evaluation metrics, and can render or save plots.
- Docker recipes for both CPU-only deployment and CUDA-enabled GPU training.

## Quick Start

```bash
# 1. Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Train models
python SQUARE_Mamba/main/train_SQUARE_Mamba.py
python SQUARE_Mamba/main/train_SQUARE_Mamba_Not_Quantum.py

# 3. Test with best checkpoint (replace with actual filename)
python SQUARE_Mamba/main/test_SQUARE_Mamba.py --checkpoint SQUARE_Mamba_BEST_2025.10.02_14.30.22-ubuntu.pkl

# 4. Run demo
python SQUARE_Mamba/demo.py --mode SQUARE-Mamba
```

## Repository Layout

```text
SQUARE_Mamba/
├── demo.py                  # MATLAB demo rewritten in Python
├── main/
│   ├── functions/           # Data loading, metrics helpers
│   ├── networks/            # Neural network building blocks
│   │   ├── SQUARE_Mamba.py          # Quantum variant (pure PyTorch QLTEM)
│   │   ├── SQUARE_Mamba_Not_Quantum.py # Classical ablation
│   │   └── mamba_cpu.py             # CPU-friendly Mamba block
│   ├── CRU_data/            # Input CRU climate datasets
│   └── Result/              # Generated metrics/forecasts
├── requirements-cpu.txt     # Minimal dependencies for CPU execution
├── requirements-gpu.txt     # Full dependency set (adds PennyLane + mamba-ssm)
├── Dockerfile.cpu           # Container recipe for CPU environments
└── Dockerfile.gpu           # Container recipe for NVIDIA GPU hosts
```

## Data

The repository ships with the CRU climate variables (`CRU_data/*.csv`) required for reproducing the experiments. No external download is necessary to run the sample pipeline.

## Python Environment Setup

### CPU-only (macOS, Windows, Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements-cpu.txt
```

### GPU (Linux + NVIDIA GPU)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements-gpu.txt
```

> **Note:** The GPU requirements include `pennylane` and `mamba-ssm`, which need CUDA toolchains. If you run the CPU build, those packages are intentionally omitted.

### macOS / Apple Silicon tip

If you experiment outside the virtual environment, force CPU execution to avoid MPS complex-number bugs:

```bash
env TORCH_MPS_DISABLE=1 python SQUARE_Mamba/main/test_SQUARE_Mamba.py
```

The training and testing scripts set this automatically, but it is useful to remember when launching ad-hoc commands.

## Training

Two training scripts are available with enhanced checkpoint management:

```bash
# Quantum-enhanced SQUARE-Mamba
python SQUARE_Mamba/main/train_SQUARE_Mamba.py

# Classical ablation without QLTEM
python SQUARE_Mamba/main/train_SQUARE_Mamba_Not_Quantum.py

# Using custom data directory
python SQUARE_Mamba/main/train_SQUARE_Mamba.py --data-dir CRU_data_montevideo
```

Both scripts:
- Detect the best available device (`cuda`, `mps`, or `cpu`).
- Save timestamped checkpoints into `SQUARE_Mamba/main/checkpoint/` with format: `<model>_yyyy.mm.dd_hh.mm.ss-<username>.pkl`
- Generate three types of checkpoints:
  - **Regular**: Saved each epoch for progress tracking
  - **BEST**: Saved when validation R² improves (suffix `_BEST_`)
  - **FINAL**: Saved at training completion (suffix `_FINAL_`)
- Print training/validation losses and R² scores every epoch.
- Support custom data directories via `--data-dir` parameter.

## Testing / Inference

After training, generate CSV outputs with:

```bash
python SQUARE_Mamba/main/test_SQUARE_Mamba.py --checkpoint SQUARE_Mamba_BEST_2025.10.02_14.30.22-ubuntu.pkl
python SQUARE_Mamba/main/test_SQUARE_Mamba_Not_Quantum.py --checkpoint SQUARE_Mamba_Not_Quantum_BEST_2025.10.02_16.22.15-ubuntu.pkl

# Using custom data directory
python SQUARE_Mamba/main/test_SQUARE_Mamba.py --checkpoint model.pkl --data-dir CRU_data_montevideo
```

Each script:
- **Requires** a `--checkpoint` parameter specifying the exact `.pkl` file to load
- Supports `--data-dir` parameter for custom data directories
- Writes timestamped results to: `SQUARE_Mamba/main/Result/<MODE>_yyyy-mm-dd_hh-mm-ss/`
  - `gt_Pooncarie.csv` (ground truth)
  - `prediction_Pooncarie.csv` (model predictions)
- Lists available checkpoints if the specified file is not found

### Checkpoint Management

Training generates organized checkpoints:
```
checkpoint/
├── SQUARE_Mamba_2025.10.02_14.30.22-ubuntu.pkl     # Regular checkpoint
├── SQUARE_Mamba_BEST_2025.10.02_14.35.42-ubuntu.pkl # Best validation R²
├── SQUARE_Mamba_FINAL_2025.10.02_15.45.33-ubuntu.pkl # Training completion
└── SQUARE_Mamba_Not_Quantum_BEST_2025.10.02_16.22.15-ubuntu.pkl
```

## Demo & Plotting

The MATLAB demo workflow is reproduced in `demo.py` with automatic result detection:

```bash
# Automatically finds the most recent result folder and displays plot
python SQUARE_Mamba/demo.py --mode "SQUARE-Mamba"

# Skip test execution and use existing results
python SQUARE_Mamba/demo.py --skip-test --mode "SQUARE-Mamba" 

# Save the figure as an image (headless)
python SQUARE_Mamba/demo.py \
    --mode "SQUARE-Mamba" \
    --skip-test \
    --save-plot exports/quantum_plot_Montevideo-JA_9datos.png \
    --no-plot

# Classical "Not Quantum" variant
python SQUARE_Mamba/demo.py \
    --mode "SQUARE_Mamba_Not_Quantum" \
    --skip-test \
    --save-plot exports/quantum_plot.png \
    --no-plot
```

The demo automatically:
- Searches for result folders matching the selected mode (e.g., `SQUARE_Mamba_2025-10-02_14-30-22/`)
- Uses the most recent timestamped folder
- Provides clear error messages if no results are found
- Displays comprehensive metrics: MAE, RMSE, R², and Pearson correlation

The plotting routine now uses the built-in `DejaVu Serif` font so it works out of the box in slim Docker images. When `--save-plot` is provided, Matplotlib switches to the non-interactive Agg backend so plots can be generated in headless environments.

## Enhanced Features

### Flexible Data Loading
- Configurable data directories via `--data-dir` parameter
- Default: `CRU_data`, customizable to any directory
- Automatic validation with helpful error messages

### Advanced Checkpoint Management
- **Timestamped naming**: `<model>_yyyy.mm.dd_hh.mm.ss-<username>.pkl`
- **User identification**: Checkpoints tagged with system username
- **Multiple checkpoint types**:
  - Regular: Saved each epoch for progress tracking
  - BEST: Saved when validation metrics improve
  - FINAL: Saved upon training completion

### Organized Result Storage
- **Timestamped result folders**: `<MODE>_yyyy-mm-dd_hh-mm-ss/`
- **Automatic folder detection**: Demo finds most recent results
- **History preservation**: All experiments are retained chronologically

### Improved Error Handling
- Clear checkpoint validation with available file listings
- Data directory existence checking
- Comprehensive error messages for troubleshooting

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
| `Times New Roman` font warning while plotting | `demo.py` now defaults to `DejaVu Serif`. If you prefer the original font, install it inside your environment or adjust the font names in the script. |
| PennyLane + MPS crash on macOS | Ensure `TORCH_MPS_DISABLE=1` is set (already done in scripts) or run on CPU. The bundled QLTEM implementation is fully PyTorch-based to avoid this path. |
| Building Docker GPU image fails | The GPU image requires Linux with an NVIDIA GPU and the `nvidia/cuda` base image. On macOS/Windows use `Dockerfile.cpu` instead. |
| Checkpoint not found error | Use `--checkpoint` parameter with exact filename. Scripts will list available checkpoints if the specified file doesn't exist. |
| No result folders found in demo | Run the corresponding test scripts first to generate timestamped result folders in `SQUARE_Mamba/main/Result/`. |
| Custom data directory not found | Ensure the directory exists and contains required CSV files (cld.csv, tmn.csv, tmp.csv, tmx.csv, vap.csv, pet.csv, pre.csv, spei.csv). |

### Working with Custom Data

When using `--data-dir`, ensure your directory contains all required CSV files:
```
your_custom_data/
├── cld.csv      # Cloud cover
├── tmn.csv      # Temperature minimum  
├── tmp.csv      # Temperature average
├── tmx.csv      # Temperature maximum
├── vap.csv      # Vapor pressure
├── pet.csv      # Potential evapotranspiration
├── pre.csv      # Precipitation
└── spei.csv     # SPEI target values
```

## Contributing

Contributions are welcome! If you improve the model, add datasets, or enhance the tooling:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear explanations and, if possible, tests or sample outputs

## References

- Po-Wei Tang and Jian-Kai Huang, *SQUARE-Mamba User Guide*, Institute of Computer and Communication Engineering, National Cheng Kung University, Tainan, Taiwan, June 30, 2025. (Bundled as `SQUARE_Mamba/ReadMe.pdf`.)

## License

This project inherits the license of the original SQUARE-Mamba release. Please consult the upstream repository or publication for citation and usage guidelines.

Enjoy forecasting! If you run into issues, feel free to open an issue or start a discussion.
