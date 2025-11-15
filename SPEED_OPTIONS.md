# Speed Optimization Options

## 🚀 Available Experiments (Fastest to Slowest)

### 1. MAXIMUM SPEED ⚡⚡⚡ (RECOMMENDED)
**Time**: 4-6 minutes total (both models)

```bash
./run_test.sh experiment_maximum_speed.py --model both
```

**Optimizations**:
- ✅ 10 training samples, 3 validation
- ✅ Batch size: 2 (tiny batches = fast backward)
- ✅ 3 epochs only
- ✅ Shot-based quantum sampling (100 shots)
- ✅ Aggressive learning rate (5e-3)
- ✅ Strong gradient clipping (0.5)

**What you get**:
- Quick comparison quantum vs classical
- Real-time progress logging
- Sufficient to see performance differences

---

### 2. SUPER FAST ⚡⚡
**Time**: 6-10 minutes total

```bash
./run_test.sh experiment_super_fast.py --model both
```

**Config**:
- 15 training samples, 5 validation
- Batch size: 5
- 5 epochs

---

### 3. OPTIMIZED MINI ⚡
**Time**: 10-20 minutes total

```bash
./run_test.sh experiment_optimized.py --model both --epochs 5 --samples 20
```

**Config**:
- 20 training samples
- Batch size: 10
- 5 epochs

---

### 4. OPTIMIZED SMALL
**Time**: 30-40 minutes total

```bash
./run_test.sh experiment_optimized.py --model both --epochs 10 --samples 30
```

**Config**:
- 30 training samples
- Batch size: 10
- 10 epochs

---

### 5. OPTIMIZED MEDIUM
**Time**: 50-100 minutes total

```bash
./run_test.sh experiment_optimized.py --model both --epochs 10 --samples 50
```

**Config**:
- 50 training samples
- Batch size: 10
- 10 epochs

---

### 6. FULL SCALE
**Time**: 16+ hours

```bash
./run_test.sh experiment_full_comparison.py --model both --epochs 251
```

**Config**:
- 945 training samples
- Batch size: 315
- 251 epochs

---

## 🎯 Recommendations by Use Case

### "I want results NOW" (< 10 minutes)
```bash
./run_test.sh experiment_maximum_speed.py --model both
```

### "I want decent results" (< 30 minutes)
```bash
./run_test.sh experiment_optimized.py --model both --epochs 5 --samples 20
```

### "I want good results" (< 2 hours)
```bash
./run_test.sh experiment_optimized.py --model both --epochs 10 --samples 50
```

### "I want publication-quality results" (overnight)
```bash
./run_test.sh experiment_full_comparison.py --model both --epochs 251
```

---

## 🔧 Technical Optimizations Applied

### 1. Batch Size Reduction
**Problem**: Large batches (315) cause very slow backward pass
**Solution**: Small batches (2-10) process faster
**Speedup**: 10-20x faster per batch

### 2. Sample Reduction
**Problem**: 945 samples take forever
**Solution**: 10-50 samples sufficient for comparison
**Speedup**: 10-90x fewer computations

### 3. Shot-Based Sampling (Quantum Only)
**Problem**: Exact expectation values require many circuit evaluations
**Solution**: Use 100 shots for stochastic but faster gradients
**Speedup**: 2-3x faster quantum gradients

### 4. Aggressive Optimization
**Problem**: Conservative learning rate needs many epochs
**Solution**: Higher LR (5e-3) + strong clipping (0.5)
**Speedup**: Converges in 3 epochs instead of 10+

### 5. Reduced Epochs
**Problem**: 251 epochs is overkill for comparison
**Solution**: 3-10 epochs sufficient to see differences
**Speedup**: 25-80x fewer iterations

---

## 📊 Speed Comparison Table

| Experiment | Samples | Epochs | Batch Size | Time (Both) | Speedup |
|------------|---------|--------|------------|-------------|---------|
| Maximum Speed | 10 | 3 | 2 | 4-6 min | 200x |
| Super Fast | 15 | 5 | 5 | 6-10 min | 100x |
| Optimized Mini | 20 | 5 | 10 | 10-20 min | 50x |
| Optimized Small | 30 | 10 | 10 | 30-40 min | 25x |
| Optimized Medium | 50 | 10 | 10 | 50-100 min | 10x |
| Full Scale | 945 | 251 | 315 | 16+ hours | 1x |

---

## 🚦 How to Choose

**Start with Maximum Speed** to verify everything works:
```bash
./run_test.sh experiment_maximum_speed.py --model both
```

If results look good and you want more confidence:
```bash
./run_test.sh experiment_optimized.py --model both --epochs 10 --samples 50
```

If you need publication results, run overnight:
```bash
./run_test.sh experiment_full_comparison.py --model both --epochs 251
```

---

## 💡 Pro Tips

1. **Always start with Maximum Speed** - Verifies code works in minutes
2. **Use `--model quantum` or `--model classical`** - Test one at a time
3. **Monitor with `ps aux | grep experiment`** - Check it's running
4. **Classical is ~30% faster** - Test classical first
5. **Results are still meaningful** - Even with 10 samples you'll see differences

---

## ⚠️ Trade-offs

### Maximum Speed
- ✅ Fast results
- ✅ Sees quantum vs classical difference
- ⚠️ High variance (few samples)
- ⚠️ Not publication-quality

### Optimized Medium
- ✅ Good balance
- ✅ Reasonable confidence
- ✅ Completes in reasonable time
- ⚠️ Still some variance

### Full Scale
- ✅ Publication-quality
- ✅ Low variance
- ✅ Comprehensive
- ⚠️ Takes 16+ hours

---

## 🎓 Understanding the Results

Even with minimal samples, you should see:
- **Quantum R²**: -4 to +0.5 (improves with training)
- **Classical R²**: Similar range
- **Difference**: Usually within ±0.1

If quantum R² > classical R² by >0.05, quantum has advantage.
If classical R² > quantum R², classical is sufficient.

---

## 🔍 Monitoring Progress

While running:
```bash
# Check it's working
ps aux | grep experiment

# See CPU usage (should be ~100%)
top -l 1 | grep Python

# Watch for completion
watch -n 10 'ps aux | grep experiment'
```

---

## ✅ Next Steps After Results

1. **Review output** - Check R² values
2. **Compare models** - Quantum vs Classical
3. **If satisfied** - Document findings
4. **If need more** - Run longer experiment
5. **Generate plots** - Use plot_experiment_results.py

---

## 🚀 Quick Start Command

**Just run this:**
```bash
./run_test.sh experiment_maximum_speed.py --model both
```

**Wait 4-6 minutes, get results!**
