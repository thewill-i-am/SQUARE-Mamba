#!/usr/bin/env python3
"""
FAST BUT MEANINGFUL experiment:
- 30 samples (enough for valid R²)
- 5 epochs (quick but meaningful)
- Batch size 10
Expected time: 10-15 minutes total
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

device = torch.device("cpu")

def fast_meaningful_experiment(model_type='quantum'):
    """Fast but with enough samples for valid metrics"""
    
    print(f"\n{'='*70}")
    print(f"🚀 FAST & MEANINGFUL {model_type.upper()} EXPERIMENT 🚀")
    print(f"{'='*70}")
    print("Configuration:")
    print("  Samples: 30 train, 10 val (enough for valid R²)")
    print("  Batch size: 10")
    print("  Epochs: 5")
    print(f"  Expected time: ~{5 if model_type=='classical' else 15} minutes")
    print(f"{'='*70}\n")
    
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    print("Loading data...")
    training_data, gt_training = load_data(0, 960)
    validation_data, gt_validation = load_data(960, 1260)
    
    training_dataset, train_gt = Create_dataset(training_data, gt_training, num_sample=30)
    validation_dataset, val_gt = Create_dataset(validation_data, gt_validation, num_sample=10)
    
    # Dataloaders
    trainloader = DataLoader(
        TensorDataset(training_dataset, train_gt[:, 4].reshape(-1, 1)),
        batch_size=10,
        shuffle=True,
    )
    valloader = DataLoader(
        TensorDataset(validation_dataset, val_gt[:, 4].reshape(-1, 1)),
        batch_size=10,
        shuffle=False,
    )
    
    print(f"Batches per epoch: {len(trainloader)}")
    
    # Create model
    print("Creating model...")
    if model_type == 'quantum':
        from networks.SQUARE_Mamba import make_model
        model = make_model(in_channel=105).to(device)
    else:
        from networks.SQUARE_Mamba_Not_Quantum import make_model as make_classical_model
        model = make_classical_model().to(device)
    
    print("Model ready!\n")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0001)
    loss_fn = nn.MSELoss()
    
    # Training
    print(f"{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    results = {'train_losses': [], 'val_losses': [], 'val_r2s': [], 'batch_times': []}
    
    for epoch in range(1, 6):  # 5 epochs
        epoch_start = time.time()
        print(f"[Epoch {epoch}/5]")
        
        # Train
        model.train()
        train_losses = []
        
        for batch_idx, (data, target) in enumerate(trainloader):
            batch_start = time.time()
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(target, output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            batch_time = time.time() - batch_start
            results['batch_times'].append(batch_time)
            
            print(f"  Batch {batch_idx+1}/{len(trainloader)}: Loss={loss.item():.4f} Time={batch_time:.1f}s")
        
        train_loss = np.mean(train_losses)
        results['train_losses'].append(train_loss)
        
        # Validate
        model.eval()
        val_losses, val_r2s = [], []
        
        with torch.no_grad():
            for data, target in valloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_losses.append(loss_fn(target, output).item())
                val_r2s.append(r_square(target, output).item())
        
        val_loss = np.mean(val_losses)
        val_r2 = np.mean(val_r2s)
        results['val_losses'].append(val_loss)
        results['val_r2s'].append(val_r2)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n  Summary:")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Val Loss:   {val_loss:.4f}")
        print(f"    Val R²:     {val_r2:.4f}")
        print(f"    Epoch Time: {epoch_time/60:.1f} min")
        
        # Estimate remaining
        elapsed = time.time() - start_time
        remaining = (elapsed / epoch) * (5 - epoch)
        print(f"    Remaining:  ~{remaining/60:.1f} min\n")
    
    total_time = time.time() - start_time
    
    print(f"{'='*70}")
    print(f"✓ {model_type.upper()} COMPLETE!")
    print(f"{'='*70}")
    print(f"Total Time:     {total_time/60:.1f} minutes")
    print(f"Best R²:        {max(results['val_r2s']):.4f}")
    print(f"Final R²:       {results['val_r2s'][-1]:.4f}")
    print(f"Final Loss:     {results['val_losses'][-1]:.4f}")
    print(f"Avg Batch Time: {np.mean(results['batch_times']):.1f}s")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['quantum', 'classical', 'both'],
                       default='both')
    args = parser.parse_args()
    
    overall_start = time.time()
    results = {}
    
    if args.model in ['quantum', 'both']:
        results['quantum'] = fast_meaningful_experiment('quantum')
    
    if args.model in ['classical', 'both']:
        results['classical'] = fast_meaningful_experiment('classical')
    
    overall_time = time.time() - overall_start
    
    # Comparison
    if args.model == 'both':
        print(f"\n{'='*70}")
        print("🏆 FINAL COMPARISON 🏆")
        print(f"{'='*70}\n")
        
        q = results['quantum']
        c = results['classical']
        
        print("Quantum Model:")
        print(f"  Best R²:        {max(q['val_r2s']):.4f}")
        print(f"  Final R²:       {q['val_r2s'][-1]:.4f}")
        print(f"  Final Loss:     {q['val_losses'][-1]:.4f}")
        print(f"  Training Time:  {sum(q['batch_times'])/60:.1f} min")
        
        print("\nClassical Model:")
        print(f"  Best R²:        {max(c['val_r2s']):.4f}")
        print(f"  Final R²:       {c['val_r2s'][-1]:.4f}")
        print(f"  Final Loss:     {c['val_losses'][-1]:.4f}")
        print(f"  Training Time:  {sum(c['batch_times'])/60:.1f} min")
        
        q_best = max(q['val_r2s'])
        c_best = max(c['val_r2s'])
        diff = q_best - c_best
        
        print(f"\nDifference (Quantum - Classical):")
        print(f"  ΔR²: {diff:+.4f} ({diff/abs(c_best)*100:+.1f}%)")
        
        print(f"\n{'='*70}")
        if diff > 0.02:
            print("✓ Quantum model shows clear advantage!")
        elif diff < -0.02:
            print("✓ Classical model shows clear advantage!")
        else:
            print("≈ Models show similar performance")
        print(f"{'='*70}")
        
        print(f"\nTotal Experiment Time: {overall_time/60:.1f} minutes")
    
    print("\n✓ Experiment complete!")
