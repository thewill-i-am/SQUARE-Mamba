# SQUARE-Mamba (Python Edition)

This repository hosts a pure-Python reimplementation of the **SQUARE-Mamba** drought forecasting pipeline. The original MATLAB demo has been ported to Python so that the model, evaluation utilities, and plotting workflow can be executed on laptops, workstations, or containers without depending on MATLAB.

The code base provides:
- A unified training/evaluation pipeline for the quantum-enhanced SQUARE-Mamba model and its classical ablation (`SQUARE_Mamba_Not_Quantum`).
- CPU-friendly fallbacks for the Mamba selective scan block and the quantum temporal encoder (QLTEM) so the project runs on macOS and standard CPUs. GPU execution is still supported via optional dependencies/Docker images.
- A Python demo (`SQUARE_Mamba/demo.py`) that replicates the MATLAB demo, prints evaluation metrics, and can render or save plots.
- Docker recipes for both CPU-only deployment and CUDA-enabled GPU training.

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

Two training scripts are available:

```bash
# Quantum-enhanced SQUARE-Mamba
python SQUARE_Mamba/main/train_SQUARE_Mamba.py

# Classical ablation without QLTEM
python SQUARE_Mamba/main/train_SQUARE_Mamba_Not_Quantum.py
```

Both scripts:
- Detect the best available device (`cuda`, `mps`, or `cpu`).
- Save checkpoints into `SQUARE_Mamba/main/checkpoint/` (files ending with `_cpu.pt`).
- Print training/validation losses every epoch.

### Quantum Noise Simulation (PennyLane)

The quantum encoder now supports configurable noise channels powered by PennyLane.

- Use `--noise` with a comma-separated list of `name=value` pairs to activate noise.
  - Supported channel names: `depolarizing`, `bit_flip`, `phase_flip`, `phase_damping`, `amplitude_damping`.
  - Example: `--noise depolarizing=0.02,amplitude_damping=0.05`.
- Optionally override the simulation backend with `--noise-device` (defaults to `default.qubit` when noiseless and `default.mixed` when noise is enabled).
- Set `--noise-shots` to sample measurement statistics instead of using analytic expectations.

The same flags are also available in `test_SQUARE_Mamba.py`, enabling side-by-side evaluation of noisy and noiseless models. Programmatic use is supported via `networks.SQUARE_Mamba.make_model(noise_config=...)` if you need finer control (e.g., passing a pre-built dictionary of channels).

## Testing / Inference

After training, generate CSV outputs with:

```bash
python SQUARE_Mamba/main/test_SQUARE_Mamba.py
python SQUARE_Mamba/main/test_SQUARE_Mamba_Not_Quantum.py
```

Each script loads the matching checkpoint (if present) and writes:
- `SQUARE_Mamba/main/Result/<MODE>/gt_Pooncarie.csv`
- `SQUARE_Mamba/main/Result/<MODE>/prediction_Pooncarie.csv`

If a checkpoint is missing, the script will emit a warning and proceed with randomly initialised weights.

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
| `Times New Roman` font warning while plotting | `demo.py` now defaults to `DejaVu Serif`. If you prefer the original font, install it inside your environment or adjust the font names in the script. |
| PennyLane + MPS crash on macOS | Ensure `TORCH_MPS_DISABLE=1` is set (already done in scripts) or run on CPU. The bundled QLTEM implementation is fully PyTorch-based to avoid this path. |
| Building Docker GPU image fails | The GPU image requires Linux with an NVIDIA GPU and the `nvidia/cuda` base image. On macOS/Windows use `Dockerfile.cpu` instead. |
| No checkpoints found | Run the corresponding `train_*.py` scripts first so checkpoint files are produced in `SQUARE_Mamba/main/checkpoint/`. |

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
