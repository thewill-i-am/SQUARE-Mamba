#!/usr/bin/env python3
"""
NOISE COMPARISON EXPERIMENT
Compares 3 scenarios:
1. Quantum without noise (ideal)
2. Quantum with noise (realistic)
3. Classical (baseline)

Configuration: 30 samples, 5 epochs, batch_size=10
Expected time: ~4.5 hours total (2h + 2h + 5min)
"""
import sys
import time
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent / "SQUARE_Mamba" / "main"))

from functions.util import Create_dataset, load_data, parse_noise_spec, r_square

device = torch.device("cpu")

def run_experiment(model_type='quantum', noise_config=None, experiment_name="experiment"):
    """Run single experiment with specified configuration"""
    
    print(f"\n{'='*70}")
    print(f"🔬 {experiment_name.upper()}")
    print(f"{'='*70}")
    print(f"Model: {model_type}")
    if noise_config:
        print(f"Noise: {noise_config}")
    else:
        print("Noise: None (ideal)")
    print(f"Samples: 30 train, 10 val")
    print(f"Epochs: 5")
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
        model = make_model(
            in_channel=105,
            noise_config=noise_config,
            quantum_device=None,
            shots=None
        ).to(device)
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
    results = {
        'experiment_name': experiment_name,
        'model_type': model_type,
        'noise_config': str(noise_config) if noise_config else None,
        'train_losses': [],
        'val_losses': [],
        'val_r2s': [],
        'epoch_times': []
    }
    
    for epoch in range(1, 6):
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
            
            print(f"  Batch {batch_idx+1}/{len(trainloader)}: Loss={loss.item():.4f} Time={batch_time:.1f}s")
        
        train_loss = np.mean(train_losses)
        results['train_losses'].append(float(train_loss))
        
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
        results['val_losses'].append(float(val_loss))
        results['val_r2s'].append(float(val_r2))
        
        epoch_time = time.time() - epoch_start
        results['epoch_times'].append(float(epoch_time))
        
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
    results['total_time'] = float(total_time)
    results['best_r2'] = float(max(results['val_r2s']))
    results['final_r2'] = float(results['val_r2s'][-1])
    results['final_loss'] = float(results['val_losses'][-1])
    
    print(f"{'='*70}")
    print(f"✓ {experiment_name.upper()} COMPLETE!")
    print(f"{'='*70}")
    print(f"Total Time:  {total_time/60:.1f} minutes")
    print(f"Best R²:     {results['best_r2']:.4f}")
    print(f"Final R²:    {results['final_r2']:.4f}")
    print(f"Final Loss:  {results['final_loss']:.4f}")
    print(f"{'='*70}\n")
    
    # Save results
    output_dir = Path("experiments") / f"noise_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{experiment_name.replace(' ', '_')}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 QUANTUM NOISE COMPARISON EXPERIMENT 🔬")
    print("="*70)
    print("\nThis experiment will compare 3 scenarios:")
    print("  1. Quantum without noise (ideal)")
    print("  2. Quantum with depolarizing noise (realistic)")
    print("  3. Classical (baseline)")
    print("\nTotal estimated time: ~4.5 hours")
    print("="*70 + "\n")
    
    overall_start = time.time()
    all_results = {}
    
    # Experiment 1: Quantum without noise
    print("\n" + "🔬"*35)
    print("EXPERIMENT 1/3: QUANTUM WITHOUT NOISE")
    print("🔬"*35)
    all_results['quantum_no_noise'] = run_experiment(
        model_type='quantum',
        noise_config=None,
        experiment_name="Quantum No Noise"
    )
    
    # Experiment 2: Quantum with noise
    print("\n" + "⚡"*35)
    print("EXPERIMENT 2/3: QUANTUM WITH NOISE")
    print("⚡"*35)
    noise_config = parse_noise_spec("depolarizing=0.02")
    all_results['quantum_with_noise'] = run_experiment(
        model_type='quantum',
        noise_config=noise_config,
        experiment_name="Quantum With Noise"
    )
    
    # Experiment 3: Classical
    print("\n" + "⚙️"*35)
    print("EXPERIMENT 3/3: CLASSICAL")
    print("⚙️"*35)
    all_results['classical'] = run_experiment(
        model_type='classical',
        noise_config=None,
        experiment_name="Classical"
    )
    
    overall_time = time.time() - overall_start
    
    # Final comparison
    print("\n" + "="*70)
    print("🏆 FINAL COMPARISON 🏆")
    print("="*70 + "\n")
    
    q_no_noise = all_results['quantum_no_noise']
    q_with_noise = all_results['quantum_with_noise']
    classical = all_results['classical']
    
    print("Results Summary:")
    print(f"\n1. Quantum (No Noise):")
    print(f"   Best R²:     {q_no_noise['best_r2']:.4f}")
    print(f"   Final R²:    {q_no_noise['final_r2']:.4f}")
    print(f"   Final Loss:  {q_no_noise['final_loss']:.4f}")
    print(f"   Time:        {q_no_noise['total_time']/60:.1f} min")
    
    print(f"\n2. Quantum (With Noise - depolarizing=0.02):")
    print(f"   Best R²:     {q_with_noise['best_r2']:.4f}")
    print(f"   Final R²:    {q_with_noise['final_r2']:.4f}")
    print(f"   Final Loss:  {q_with_noise['final_loss']:.4f}")
    print(f"   Time:        {q_with_noise['total_time']/60:.1f} min")
    
    print(f"\n3. Classical:")
    print(f"   Best R²:     {classical['best_r2']:.4f}")
    print(f"   Final R²:    {classical['final_r2']:.4f}")
    print(f"   Final Loss:  {classical['final_loss']:.4f}")
    print(f"   Time:        {classical['total_time']/60:.1f} min")
    
    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    noise_effect = q_with_noise['best_r2'] - q_no_noise['best_r2']
    print(f"\nNoise Effect:")
    print(f"  ΔR² (With Noise - No Noise): {noise_effect:+.4f}")
    if abs(noise_effect) > 0.01:
        print(f"  ✓ Noise has measurable effect ({noise_effect/abs(q_no_noise['best_r2'])*100:+.1f}%)")
    else:
        print(f"  ≈ Noise effect is minimal")
    
    q_vs_c_no_noise = q_no_noise['best_r2'] - classical['best_r2']
    q_vs_c_with_noise = q_with_noise['best_r2'] - classical['best_r2']
    
    print(f"\nQuantum vs Classical:")
    print(f"  Without Noise: {q_vs_c_no_noise:+.4f}")
    print(f"  With Noise:    {q_vs_c_with_noise:+.4f}")
    
    if classical['best_r2'] > q_no_noise['best_r2']:
        print(f"\n  ✓ Classical outperforms Quantum (even without noise)")
    elif classical['best_r2'] > q_with_noise['best_r2'] and classical['best_r2'] < q_no_noise['best_r2']:
        print(f"\n  ✓ Quantum without noise > Classical > Quantum with noise")
    
    print(f"\nSpeed Comparison:")
    print(f"  Classical is {q_no_noise['total_time']/classical['total_time']:.0f}x faster than Quantum")
    
    print(f"\n{'='*70}")
    print(f"Total Experiment Time: {overall_time/3600:.2f} hours")
    print(f"{'='*70}")
    
    # Save combined results
    output_dir = Path("experiments") / f"noise_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "combined_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print("\n✓ All experiments complete!")
