#!/usr/bin/env python3
"""
SUPER FAST experiment with all optimizations:
- Minimal samples (15)
- Small batches (5)
- Fewer epochs (5)
- Progress logging
Expected time: 3-5 minutes per model
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent / "SQUARE_Mamba" / "main"))

from functions.util import Create_dataset, load_data, r_square

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def super_fast_experiment(model_type='quantum'):
    """Super fast experiment - results in minutes"""
    
    print(f"\n{'='*70}")
    print(f"SUPER FAST {model_type.upper()} EXPERIMENT")
    print(f"{'='*70}")
    print("Samples: 15 train, 5 val")
    print("Batch size: 5")
    print("Epochs: 5")
    print("Expected time: 3-5 minutes")
    print(f"{'='*70}\n")
    
    # Minimal setup
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load minimal data
    training_data, gt_training = load_data(0, 960)
    validation_data, gt_validation = load_data(960, 1260)
    
    training_dataset, train_gt = Create_dataset(training_data, gt_training, num_sample=15)
    validation_dataset, val_gt = Create_dataset(validation_data, gt_validation, num_sample=5)
    
    # Small batches
    trainloader = DataLoader(
        TensorDataset(training_dataset, train_gt[:, 4].reshape(-1, 1)),
        batch_size=5,
        shuffle=True,
    )
    valloader = DataLoader(
        TensorDataset(validation_dataset, val_gt[:, 4].reshape(-1, 1)),
        batch_size=5,
        shuffle=False,
    )
    
    print(f"Batches per epoch: {len(trainloader)}")
    
    # Create model
    if model_type == 'quantum':
        from networks.SQUARE_Mamba import make_model
        model = make_model(in_channel=105).to(device)
    else:
        from networks.SQUARE_Mamba_Not_Quantum import make_model as make_classical_model
        model = make_classical_model(in_channel=105).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # Training
    start_time = time.time()
    results = {'losses': [], 'r2s': [], 'times': []}
    
    for epoch in range(1, 6):  # 5 epochs
        epoch_start = time.time()
        print(f"\n[Epoch {epoch}/5]")
        
        # Train
        model.train()
        for batch_idx, (data, target) in enumerate(trainloader):
            print(f"  Training batch {batch_idx+1}/{len(trainloader)}...", end='', flush=True)
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(target, output)
            loss.backward()
            optimizer.step()
            
            print(f" Loss: {loss.item():.4f}")
        
        # Quick validation
        model.eval()
        with torch.no_grad():
            val_losses, val_r2s = [], []
            for data, target in valloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_losses.append(loss_fn(target, output).item())
                val_r2s.append(r_square(target, output).item())
        
        val_loss = np.mean(val_losses)
        val_r2 = np.mean(val_r2s)
        epoch_time = time.time() - epoch_start
        
        results['losses'].append(val_loss)
        results['r2s'].append(val_r2)
        results['times'].append(epoch_time)
        
        print(f"  Val Loss: {val_loss:.4f} | R²: {val_r2:.4f} | Time: {epoch_time:.1f}s")
        
        # Estimate remaining
        elapsed = time.time() - start_time
        remaining = (elapsed / epoch) * (5 - epoch)
        print(f"  Remaining: ~{remaining/60:.1f} min")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"{model_type.upper()} COMPLETE!")
    print(f"{'='*70}")
    print(f"Time: {total_time/60:.1f} minutes")
    print(f"Best R²: {max(results['r2s']):.4f}")
    print(f"Final R²: {results['r2s'][-1]:.4f}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['quantum', 'classical', 'both'],
                       default='both')
    args = parser.parse_args()
    
    results = {}
    
    if args.model in ['quantum', 'both']:
        results['quantum'] = super_fast_experiment('quantum')
    
    if args.model in ['classical', 'both']:
        results['classical'] = super_fast_experiment('classical')
    
    if args.model == 'both':
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        q_best = max(results['quantum']['r2s'])
        c_best = max(results['classical']['r2s'])
        print(f"Quantum Best R²:   {q_best:.4f}")
        print(f"Classical Best R²: {c_best:.4f}")
        print(f"Difference:        {q_best - c_best:+.4f}")
        print(f"{'='*70}")
