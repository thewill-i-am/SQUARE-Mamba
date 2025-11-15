# Full-Scale Experiment Guide (251 Epochs)

## Overview

This guide explains how to run a complete 251-epoch experiment comparing quantum and classical SQUARE-Mamba models.

## ⚠️ Important Notes

### Time Requirements
- **Quantum model**: ~50-60 hours (estimated)
- **Classical model**: ~40-50 hours (estimated)
- **Both models**: ~90-110 hours total (3.75-4.5 days)

### Resource Requirements
- **Disk space**: ~500MB per experiment (checkpoints + logs)
- **Memory**: ~4-8GB RAM
- **CPU**: Will use 100% of available cores

### Recommendations
- Run overnight or over a weekend
- Use `tmux` or `screen` to keep session alive
- Monitor progress via log files
- Ensure stable power supply

## 🚀 Quick Start

### Option 1: Compare Both Models (Recommended)

```bash
# Run both quantum and classical models
./run_test.sh experiment_full_comparison.py --model both --epochs 251
```

This will:
1. Train quantum model for 251 epochs
2. Train classical model for 251 epochs
3. Save checkpoints every 10 epochs
4. Generate comparison report

### Option 2: Single Model

```bash
# Quantum only
./run_test.sh experiment_full_comparison.py --model quantum --epochs 251

# Classical only
./run_test.sh experiment_full_comparison.py --model classical --epochs 251
```

### Option 3: With Quantum Noise

```bash
# Quantum with depolarizing noise
./run_test.sh experiment_full_comparison.py \
    --model quantum \
    --epochs 251 \
    --noise depolarizing=0.02
```

## 📊 Output Structure

Each experiment creates a directory in `experiments/`:

```
experiments/
├── quantum_20241114_120000/
│   ├── training.log              # Detailed training log
│   ├── metrics.json              # All metrics (losses, R², times)
│   ├── checkpoint_epoch_10.pt    # Checkpoint at epoch 10
│   ├── checkpoint_epoch_20.pt    # Checkpoint at epoch 20
│   ├── ...
│   ├── checkpoint_best.pt        # Best model (highest R²)
│   └── checkpoint_latest.pt      # Latest model
└── classical_20241114_130000/
    └── (same structure)
```

## 📈 Monitoring Progress

### Real-time Monitoring

```bash
# Watch the log file
tail -f experiments/quantum_*/training.log

# Check current epoch
grep "EPOCH" experiments/quantum_*/training.log | tail -1

# Check best R² so far
grep "NEW BEST" experiments/quantum_*/training.log | tail -1
```

### Check Metrics

```bash
# View metrics JSON
cat experiments/quantum_*/metrics.json | python -m json.tool

# Quick stats
python -c "import json; m=json.load(open('experiments/quantum_*/metrics.json')); print(f'Best R²: {m[\"best_r2\"]:.6f} at epoch {m[\"best_epoch\"]}')"
```

## 📊 Visualizing Results

### After Training Completes

```bash
# Plot single experiment
./run_test.sh plot_experiment_results.py \
    --single experiments/quantum_20241114_120000 \
    --output quantum_results.png

# Compare quantum vs classical
./run_test.sh plot_experiment_results.py \
    --quantum experiments/quantum_20241114_120000 \
    --classical experiments/classical_20241114_130000 \
    --output comparison.png
```

This generates plots showing:
- Training loss curves
- Validation loss curves
- Validation R² curves
- Training time per epoch
- Statistical comparison

## 🔄 Resuming Interrupted Training

If training is interrupted, you can resume from the latest checkpoint:

```python
# Add to experiment_full_comparison.py (future enhancement)
# Load checkpoint:
checkpoint = torch.load('experiments/quantum_*/checkpoint_latest.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## 📋 Expected Results

### Training Progression

**Early epochs (1-50)**:
- R² will be negative (normal with random weights)
- Loss will decrease rapidly
- R² will improve toward 0

**Mid epochs (51-150)**:
- R² should become positive
- Loss continues to decrease
- R² improves steadily

**Late epochs (151-251)**:
- R² should stabilize
- Loss may plateau
- Best R² likely achieved here

### Performance Targets

Based on similar models:
- **Good**: R² > 0.5
- **Very good**: R² > 0.7
- **Excellent**: R² > 0.8

### Quantum vs Classical

Expected differences:
- Quantum may achieve slightly higher R² (1-5% improvement)
- Classical trains faster (~20-30% faster per epoch)
- Quantum has more parameters (quantum circuit weights)

## 🐛 Troubleshooting

### Training is Very Slow

**Expected**: Quantum model backward pass takes ~75s per batch
- This is normal for quantum circuits
- Parameter-shift rule requires 2N circuit evaluations
- Cannot be significantly optimized

**Solutions**:
- Use classical model for faster results
- Reduce number of epochs for testing
- Run on faster hardware

### Out of Memory

**Symptoms**: Process killed, "Out of memory" error

**Solutions**:
```bash
# Reduce batch size in experiment_full_comparison.py
# Change line: batch_size = 315
# To: batch_size = 64  # or smaller
```

### Disk Space Full

**Symptoms**: "No space left on device"

**Solutions**:
```bash
# Remove old checkpoints (keep best and latest)
rm experiments/*/checkpoint_epoch_*.pt

# Or reduce checkpoint frequency
# In experiment_full_comparison.py, change:
# if epoch % 10 == 0:  # Save every 10 epochs
# To: if epoch % 50 == 0:  # Save every 50 epochs
```

### Process Killed Unexpectedly

**Solutions**:
```bash
# Use tmux to keep session alive
tmux new -s experiment
./run_test.sh experiment_full_comparison.py --model both --epochs 251

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t experiment

# Or use nohup
nohup ./run_test.sh experiment_full_comparison.py --model both --epochs 251 > experiment.log 2>&1 &
```

### R² Not Improving

**Possible causes**:
1. Learning rate too high/low
2. Model not converging
3. Data issues

**Solutions**:
- Check training log for NaN values
- Verify data loaded correctly
- Try different learning rate
- Check gradient clipping is working

## 📊 Analyzing Results

### Key Metrics to Compare

1. **Best R²**: Highest validation R² achieved
2. **Final R²**: R² at epoch 251
3. **Convergence**: How quickly R² improves
4. **Stability**: Variance in R² over last 50 epochs
5. **Training time**: Total time and time per epoch

### Statistical Significance

To determine if quantum improvement is significant:

```python
import numpy as np
from scipy import stats

# Load metrics
q_r2 = metrics_quantum['val_r2s'][-50:]  # Last 50 epochs
c_r2 = metrics_classical['val_r2s'][-50:]

# T-test
t_stat, p_value = stats.ttest_ind(q_r2, c_r2)
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Difference is statistically significant")
else:
    print("Difference is not statistically significant")
```

## 🎯 Experiment Checklist

Before starting:
- [ ] Verify system works: `./run_test.sh test_simple_quantum.py`
- [ ] Check disk space: `df -h`
- [ ] Check available memory: `free -h`
- [ ] Estimate time: ~4 days for both models
- [ ] Set up tmux/screen session
- [ ] Plan monitoring schedule

During training:
- [ ] Check progress every few hours
- [ ] Monitor disk space
- [ ] Verify no errors in log
- [ ] Check R² is improving

After training:
- [ ] Generate plots
- [ ] Compare metrics
- [ ] Save best checkpoints
- [ ] Document findings
- [ ] Clean up intermediate checkpoints

## 📝 Example Commands

### Full Workflow

```bash
# 1. Verify system
./run_test.sh test_simple_quantum.py

# 2. Start experiment in tmux
tmux new -s experiment
./run_test.sh experiment_full_comparison.py --model both --epochs 251

# 3. Detach (Ctrl+B, D) and let it run

# 4. Check progress periodically
tail -f experiments/quantum_*/training.log

# 5. After completion, generate plots
./run_test.sh plot_experiment_results.py \
    --quantum experiments/quantum_* \
    --classical experiments/classical_* \
    --output final_comparison.png

# 6. View results
open final_comparison.png  # macOS
# or
xdg-open final_comparison.png  # Linux
```

### Quick Test (10 Epochs)

Before running the full 251 epochs, test with 10:

```bash
./run_test.sh experiment_full_comparison.py --model both --epochs 10
```

This takes ~2-3 hours and verifies everything works.

## 🔬 Advanced Options

### Custom Learning Rate

Edit `experiment_full_comparison.py`:
```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,  # Change from 1e-3 to 1e-4
    weight_decay=0.0001,
)
```

### Different Noise Levels

```bash
# Low noise
./run_test.sh experiment_full_comparison.py \
    --model quantum --epochs 251 --noise depolarizing=0.01

# High noise
./run_test.sh experiment_full_comparison.py \
    --model quantum --epochs 251 --noise depolarizing=0.05

# Multiple channels
./run_test.sh experiment_full_comparison.py \
    --model quantum --epochs 251 \
    --noise depolarizing=0.02,amplitude_damping=0.05
```

### Early Stopping

Add to `experiment_full_comparison.py`:
```python
# After validation
patience = 50
if epoch - best_epoch > patience:
    logger.log(f"Early stopping: no improvement for {patience} epochs")
    break
```

## 📚 Additional Resources

- `README.md` - Main documentation
- `TESTING_GUIDE.md` - Setup and testing
- `NOISE_EXPERIMENT_RESULTS.md` - Noise validation results
- `QUICK_REFERENCE.md` - Fast commands

## ✅ Success Criteria

Your experiment is successful if:
- ✅ Training completes all 251 epochs
- ✅ No NaN or Inf values in metrics
- ✅ R² improves from negative to positive
- ✅ Best R² > 0.5 (good) or > 0.7 (very good)
- ✅ Checkpoints saved correctly
- ✅ Plots generated successfully

## 🎓 Interpreting Results

### If Quantum > Classical
- Quantum circuits provide useful feature extraction
- Noise level is appropriate
- Model architecture is well-designed

### If Classical > Quantum
- Quantum overhead not justified for this task
- May need different quantum circuit design
- Classical model may be sufficient

### If Similar Performance
- Quantum provides no significant advantage
- Consider: Is quantum necessary for this problem?
- May need more complex quantum circuits

## 📞 Support

If you encounter issues:
1. Check `TESTING_GUIDE.md` troubleshooting section
2. Review log files for error messages
3. Verify system with `test_simple_quantum.py`
4. Check disk space and memory
5. Try with fewer epochs first (10-20)

Good luck with your experiment! 🚀
