#!/usr/bin/env python3
"""
OPTIMIZED experiment: Small batches for faster training
Key optimization: batch_size=10 instead of 315
"""
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent / "SQUARE_Mamba" / "main"))

from functions.util import Create_dataset, load_data, r_square

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_optimized_experiment(model_type='quantum', epochs=10, samples=50):
    """Optimized experiment with small batches"""
    
    print("="*70)
    print(f"OPTIMIZED EXPERIMENT - {model_type.upper()}")
    print("="*70)
    print(f"Samples: {samples} training, {samples//3} validation")
    print(f"Batch size: 10 (OPTIMIZED)")
    print(f"Epochs: {epochs}")
    print("="*70)
    print()
    
    # Setup
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data
    print("Loading data...")
    training_data, gt_training = load_data(0, 960)
    validation_data, gt_validation = load_data(960, 1260)
    
    train_samples = samples
    val_samples = samples // 3
    
    training_dataset, train_gt = Create_dataset(training_data, gt_training, num_sample=train_samples)
    validation_dataset, val_gt = Create_dataset(validation_data, gt_validation, num_sample=val_samples)
    
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    
    # OPTIMIZED: Small batch size for faster backward pass
    BATCH_SIZE = 10
    trainloader = DataLoader(
        TensorDataset(training_dataset, train_gt[:, 4].reshape(-1, 1)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    valloader = DataLoader(
        TensorDataset(validation_dataset, val_gt[:, 4].reshape(-1, 1)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    
    print(f"Batches per epoch: {len(trainloader)} (batch_size={BATCH_SIZE})")
    
    # Create model
    print("\nCreating model...")
    if model_type == 'quantum':
        from networks.SQUARE_Mamba import make_model
        model = make_model(in_channel=105).to(device)
    else:
        from networks.SQUARE_Mamba_Not_Quantum import make_model as make_classical_model
        model = make_classical_model(in_channel=105).to(device)
    
    print("Model created!")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0001)
    loss_fn = nn.MSELoss()
    
    # Training
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70)
    
    start_time = time.time()
    results = {
        'train_losses': [],
        'val_losses': [],
        'val_r2s': [],
        'epoch_times': [],
        'batch_times': []
    }
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{epochs}")
        print(f"{'='*70}")
        
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
            
            # Print every batch
            print(f"  Batch {batch_idx+1}/{len(trainloader)}: "
                  f"Loss={loss.item():.6f} Time={batch_time:.1f}s")
        
        train_loss = np.mean(train_losses)
        results['train_losses'].append(float(train_loss))
        
        # Validate
        model.eval()
        val_losses = []
        val_r2s = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valloader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(target, output)
                r2 = r_square(target, output)
                
                val_losses.append(loss.item())
                val_r2s.append(r2.item())
        
        val_loss = np.mean(val_losses)
        val_r2 = np.mean(val_r2s)
        results['val_losses'].append(float(val_loss))
        results['val_r2s'].append(float(val_r2))
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        results['epoch_times'].append(float(epoch_time))
        
        avg_batch_time = np.mean([t for t in results['batch_times'][-len(trainloader):]])
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*70}")
        print(f"  Train Loss:     {train_loss:.6f}")
        print(f"  Val Loss:       {val_loss:.6f}")
        print(f"  Val R²:         {val_r2:.6f}")
        print(f"  Epoch Time:     {epoch_time:.1f}s")
        print(f"  Avg Batch Time: {avg_batch_time:.1f}s")
        
        # Progress estimate
        elapsed = time.time() - start_time
        avg_time = elapsed / epoch
        remaining = avg_time * (epochs - epoch)
        print(f"\n  Progress: {epoch}/{epochs} ({epoch/epochs*100:.0f}%)")
        print(f"  Elapsed: {elapsed/60:.1f}min | Remaining: {remaining/60:.1f}min")
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Avg time/epoch: {total_time/epochs:.1f}s")
    print(f"Avg time/batch: {np.mean(results['batch_times']):.1f}s")
    print(f"\nFinal Results:")
    print(f"  Final Train Loss: {results['train_losses'][-1]:.6f}")
    print(f"  Final Val Loss:   {results['val_losses'][-1]:.6f}")
    print(f"  Final Val R²:     {results['val_r2s'][-1]:.6f}")
    print(f"  Best R²:          {max(results['val_r2s']):.6f}")
    print("="*70)
    
    # Save results
    output_dir = Path("experiments") / f"{model_type}_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized experiment")
    parser.add_argument('--model', choices=['quantum', 'classical', 'both'],
                       default='both', help='Model type')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of training samples')
    
    args = parser.parse_args()
    
    all_results = {}
    
    if args.model in ['quantum', 'both']:
        print("\n" + "🔬"*35)
        print("QUANTUM MODEL")
        print("🔬"*35 + "\n")
        all_results['quantum'] = run_optimized_experiment('quantum', args.epochs, args.samples)
    
    if args.model in ['classical', 'both']:
        print("\n" + "⚙️"*35)
        print("CLASSICAL MODEL")
        print("⚙️"*35 + "\n")
        all_results['classical'] = run_optimized_experiment('classical', args.epochs, args.samples)
    
    # Comparison
    if args.model == 'both':
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        
        q = all_results['quantum']
        c = all_results['classical']
        
        print(f"\nQuantum:")
        print(f"  Best R²: {max(q['val_r2s']):.6f}")
        print(f"  Final R²: {q['val_r2s'][-1]:.6f}")
        print(f"  Avg time/epoch: {np.mean(q['epoch_times']):.1f}s")
        print(f"  Avg time/batch: {np.mean(q['batch_times']):.1f}s")
        
        print(f"\nClassical:")
        print(f"  Best R²: {max(c['val_r2s']):.6f}")
        print(f"  Final R²: {c['val_r2s'][-1]:.6f}")
        print(f"  Avg time/epoch: {np.mean(c['epoch_times']):.1f}s")
        print(f"  Avg time/batch: {np.mean(c['batch_times']):.1f}s")
        
        diff = max(q['val_r2s']) - max(c['val_r2s'])
        print(f"\nDifference (Q - C): {diff:+.6f}")
        
        if diff > 0:
            print("✓ Quantum performed better!")
        elif diff < 0:
            print("✓ Classical performed better!")
        else:
            print("= Similar performance")
        
        print("="*70)
    
    print("\n✓ Experiment complete!")
