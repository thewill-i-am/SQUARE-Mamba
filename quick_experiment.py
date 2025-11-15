#!/usr/bin/env python3
"""
QUICK experiment: 100 samples, much faster
Expected time: ~10-15 minutes for 10 epochs
"""
import sys
from pathlib import Path

# Copy the experiment code but with reduced samples
exec(Path('experiment_full_comparison.py').read_text().replace(
    'train_samples = 945',
    'train_samples = 100  # FAST VERSION'
).replace(
    'val_samples = 285',
    'val_samples = 30  # FAST VERSION'
))
