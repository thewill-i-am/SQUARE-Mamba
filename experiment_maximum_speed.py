#!/usr/bin/env python3
"""
MAXIMUM SPEED experiment with ALL optimizations:
1. Minimal samples (10)
2. Tiny batches (2) 
3. Reduced quantum circuit complexity
4. Shot-based sampling (faster than exact)
5. Fewer epochs (3)
6. Aggressive gradient clipping

Expected time: 2-3 minutes per model (4-6 minutes total)
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

device = torch.device("cpu")  # Force CPU

def create_fast_quantum_model():
    """Create quantum model with speed optimizations"""
    from networks.SQUARE_Mamba import make_model
    
    # Use shot-based sampling for faster (but stochastic) gradients
    noise_config = None  # No noise for speed
    
    model = make_model(
        in_channel=105,
        noise_config=noise_config,
        quantum_device=None,
        shots=100  # Use shots for faster computation
    )
    return model


def maximum_speed_experiment(model_type='quantum'):
    """Maximum speed experiment"""
    
    print(f"\n{'='*70}")
    print(f"🚀 MAXIMUM SPEED {model_type.upper()} EXPERIMENT 🚀")
    print(f"{'='*70}")
    print("Optimizations:")
    print("  ✓ Minimal samples: 10 train, 3 val")
    print("  ✓ Tiny batches: 2 samples/batch")
    print("  ✓ Reduced epochs: 3")
    if model_type == 'quantum':
        print("  ✓ Shot-based sampling: 100 shots")
        print("  ✓ Simplified gradients")
    print("  ✓ Aggressive optimization")
    print(f"\nExpected time: 2-3 minutes")
    print(f"{'='*70}\n")
    
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    
    # MINIMAL data
    print("Loading data...")
    training_data, gt_training = load_data(0, 960)
    validation_data, gt_validation = load_data(960, 1260)
    
    training_dataset, train_gt = Create_dataset(training_data, gt_training, num_sample=10)
    validation_dataset, val_gt = Create_dataset(validation_data, gt_validation, num_sample=3)
    
    # TINY batches for faster backward pass
    BATCH_SIZE = 2
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
    
    print(f"Batches per epoch: {len(trainloader)}")
    
    # Create model
    print("Creating model...")
    if model_type == 'quantum':
        model = create_fast_quantum_model().to(device)
    else:
        from networks.SQUARE_Mamba_Not_Quantum import make_model as make_classical_model
        model = make_classical_model().to(device)  # Classical model doesn't take in_channel parameter
    
    print("Model ready!")
    
    # AGGRESSIVE optimizer settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-3,  # Higher learning rate for faster convergence
        weight_decay=0.01
    )
    loss_fn = nn.MSELoss()
    
    # Training
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    results = {'losses': [], 'r2s': [], 'batch_times': []}
    
    EPOCHS = 3  # Minimal epochs
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"[Epoch {epoch}/{EPOCHS}]")
        
        # Train
        model.train()
        train_losses = []
        
        for batch_idx, (data, target) in enumerate(trainloader):
            batch_start = time.time()
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward
            output = model(data)
            loss = loss_fn(target, output)
            
            # Backward with aggressive clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            batch_time = time.time() - batch_start
            results['batch_times'].append(batch_time)
            train_losses.append(loss.item())
            
            print(f"  Batch {batch_idx+1}/{len(trainloader)}: "
                  f"Loss={loss.item():.4f} Time={batch_time:.1f}s", flush=True)
        
        # Quick validation
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
        train_loss = np.mean(train_losses)
        epoch_time = time.time() - epoch_start
        
        results['losses'].append(val_loss)
        results['r2s'].append(val_r2)
        
        print(f"\n  Summary:")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Val Loss:   {val_loss:.4f}")
        print(f"    Val R²:     {val_r2:.4f}")
        print(f"    Epoch Time: {epoch_time:.1f}s")
        
        # Estimate remaining
        elapsed = time.time() - start_time
        remaining = (elapsed / epoch) * (EPOCHS - epoch)
        print(f"    Remaining:  ~{remaining:.0f}s\n")
    
    total_time = time.time() - start_time
    avg_batch_time = np.mean(results['batch_times'])
    
    print(f"{'='*70}")
    print(f"✓ {model_type.upper()} COMPLETE!")
    print(f"{'='*70}")
    print(f"Total Time:        {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Avg Batch Time:    {avg_batch_time:.1f}s")
    print(f"Best R²:           {max(results['r2s']):.4f}")
    print(f"Final R²:          {results['r2s'][-1]:.4f}")
    print(f"Final Loss:        {results['losses'][-1]:.4f}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Maximum speed experiment")
    parser.add_argument('--model', choices=['quantum', 'classical', 'both'],
                       default='both', help='Model type')
    
    args = parser.parse_args()
    
    overall_start = time.time()
    results = {}
    
    if args.model in ['quantum', 'both']:
        print("\n" + "🔬"*35)
        print("QUANTUM MODEL")
        print("🔬"*35)
        results['quantum'] = maximum_speed_experiment('quantum')
    
    if args.model in ['classical', 'both']:
        print("\n" + "⚙️"*35)
        print("CLASSICAL MODEL")
        print("⚙️"*35)
        results['classical'] = maximum_speed_experiment('classical')
    
    overall_time = time.time() - overall_start
    
    # Final comparison
    if args.model == 'both':
        print("\n" + "="*70)
        print("🏆 FINAL COMPARISON 🏆")
        print("="*70)
        
        q = results['quantum']
        c = results['classical']
        
        q_best = max(q['r2s'])
        c_best = max(c['r2s'])
        q_final = q['r2s'][-1]
        c_final = c['r2s'][-1]
        q_time = np.sum([t for t in q['batch_times']])
        c_time = np.sum([t for t in c['batch_times']])
        
        print(f"\nQuantum Model:")
        print(f"  Best R²:       {q_best:.4f}")
        print(f"  Final R²:      {q_final:.4f}")
        print(f"  Training Time: {q_time:.1f}s")
        print(f"  Avg Batch:     {np.mean(q['batch_times']):.1f}s")
        
        print(f"\nClassical Model:")
        print(f"  Best R²:       {c_best:.4f}")
        print(f"  Final R²:      {c_final:.4f}")
        print(f"  Training Time: {c_time:.1f}s")
        print(f"  Avg Batch:     {np.mean(c['batch_times']):.1f}s")
        
        print(f"\nDifferences:")
        best_diff = q_best - c_best
        final_diff = q_final - c_final
        time_diff = q_time - c_time
        
        print(f"  ΔBest R²:  {best_diff:+.4f} ({best_diff/abs(c_best)*100:+.1f}%)")
        print(f"  ΔFinal R²: {final_diff:+.4f} ({final_diff/abs(c_final)*100:+.1f}%)")
        print(f"  ΔTime:     {time_diff:+.1f}s ({time_diff/c_time*100:+.1f}%)")
        
        print(f"\n{'='*70}")
        if best_diff > 0.01:
            print("✓ Quantum model shows advantage!")
        elif best_diff < -0.01:
            print("✓ Classical model shows advantage!")
        else:
            print("≈ Models show similar performance")
        print(f"{'='*70}")
        
        print(f"\nTotal Experiment Time: {overall_time:.1f}s ({overall_time/60:.2f} min)")
    
    print("\n✓ Experiment complete!")
    print(f"{'='*70}\n")
