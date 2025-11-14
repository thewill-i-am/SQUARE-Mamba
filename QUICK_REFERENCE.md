# Quick Reference Card

## 🚀 Fast Commands

### Validation (Run First)
```bash
./run_test.sh test_simple_quantum.py          # ~10s - System check
./run_test.sh train_ultra_fast.py             # ~3min - Noise test
```

### Training
```bash
# Quick test (30 samples, 3 epochs, ~20min)
./run_test.sh train_simple_with_noise.py

# Full training - No noise
./run_test.sh SQUARE_Mamba/main/train_SQUARE_Mamba.py --epochs 3

# Full training - With noise
./run_test.sh SQUARE_Mamba/main/train_SQUARE_Mamba.py \
    --epochs 3 \
    --noise depolarizing=0.02
```

## 📊 Noise Channels

| Channel | Parameter | Example |
|---------|-----------|---------|
| Depolarizing | probability | `depolarizing=0.02` |
| Amplitude damping | gamma | `amplitude_damping=0.05` |
| Phase damping | lambda | `phase_damping=0.03` |
| Bit flip | probability | `bit_flip=0.01` |
| Phase flip | probability | `phase_flip=0.01` |

### Multiple Channels
```bash
--noise depolarizing=0.02,amplitude_damping=0.05,phase_damping=0.03
```

## 🔧 Environment

### Current Setup
- Python: 3.10.13 (`.venv-3.10/`)
- PyTorch: 2.2.1
- Qiskit: 1.4.3
- Helper: `./run_test.sh`

### Manual Execution
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
.venv-3.10/bin/python <script.py>
```

## ⏱️ Expected Times

| Task | Duration |
|------|----------|
| System validation | ~10 seconds |
| Quick noise test | ~3 minutes |
| Simple training (3 epochs) | ~20 minutes |
| Full training (10 epochs) | ~1-2 hours |

## ✅ Validation Checklist

- [ ] Run `./run_test.sh test_simple_quantum.py` → All checks pass
- [ ] Run `./run_test.sh train_ultra_fast.py` → ~10% noise effect
- [ ] Check output: Baseline vs Noisy shows difference
- [ ] Verify gradients: No NaN or Inf values

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| OMP Error #15 | Use `./run_test.sh` (sets KMP_DUPLICATE_LIB_OK) |
| Slow training | Expected (~75s/batch for backward pass) |
| Negative R² | Normal in early training |
| UserWarning | Fixed in `.venv-3.10/` |

## 📚 Documentation

| File | Purpose |
|------|---------|
| `README.md` | Main documentation |
| `TESTING_GUIDE.md` | Complete setup guide |
| `NOISE_EXPERIMENT_RESULTS.md` | Validated results |
| `.kiro/specs/noise-simulation-fix/` | Implementation spec |

## 🎯 Noise Effect Validation

**Expected Results** (from `train_ultra_fast.py`):

```
Baseline:     Loss=1.502, R²=-4.317
With noise:   Loss=1.354, R²=-3.792
Difference:   ~10% (noise is working)
```

If you see ~10% difference → ✅ System working correctly

## 💡 Tips

1. **Always use `./run_test.sh`** - Handles environment automatically
2. **Start with `train_ultra_fast.py`** - Quick validation
3. **Check gradients** - Should not be NaN/Inf
4. **Negative R² is OK** - Improves with training
5. **Backward pass is slow** - Expected (parameter-shift rule)

## 🔗 Quick Links

- System validation: `./run_test.sh test_simple_quantum.py`
- Quick noise test: `./run_test.sh train_ultra_fast.py`
- Full guide: `TESTING_GUIDE.md`
- Results: `NOISE_EXPERIMENT_RESULTS.md`
- Spec tasks: `.kiro/specs/noise-simulation-fix/tasks.md`
