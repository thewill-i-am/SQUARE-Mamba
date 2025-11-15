#!/usr/bin/env python3
"""
Full-scale experiment: 251 epochs comparing quantum vs classical SQUARE_Mamba.
Includes detailed logging, checkpointing, and metric tracking.
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

# Add path
sys.path.insert(0, str(Path(__file__).parent / "SQUARE_Mamba" / "main"))

from functions.util import Create_dataset, load_data, parse_noise_spec, r_square

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ExperimentLogger:
    """Logs experiment progress and saves results"""
    
    def __init__(self, experiment_name, output_dir="experiments"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        self.exp_dir.mkdir(exist_ok=True)
        
        self.log_file = self.exp_dir / "training.log"
        self.metrics_file = self.exp_dir / "metrics.json"
        
        self.metrics = {
            'train_losses': [],
            'val_losses': [],
            'val_r2s': [],
            'epoch_times': [],
            'best_r2': -999,
            'best_epoch': 0,
        }
        
        self.log(f"Experiment: {experiment_name}")
        self.log(f"Output directory: {self.exp_dir}")
        self.log(f"Device: {device}")
        self.log("="*70)
    
    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def save_metrics(self):
        """Save metrics to JSON"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def save_checkpoint(self, model, epoch, optimizer=None):
        """Save model checkpoint"""
        checkpoint_path = self.exp_dir / f"checkpoint_epoch_{epoch}.pt"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': self.metrics,
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as "latest"
        latest_path = self.exp_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        return checkpoint_path


def train_epoch(model, dataloader, optimizer, loss_fn, epoch, logger):
    """Train one epoch"""
    model.train()
    losses = []
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(target, output)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
            logger.log(f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.6f} | Avg: {np.mean(losses):.6f}")
    
    return np.mean(losses)


def validate_epoch(model, dataloader, loss_fn, epoch, logger):
    """Validate one epoch"""
    model.eval()
    losses = []
    r2_scores = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = loss_fn(target, output)
            r2 = r_square(target, output)
            
            losses.append(loss.item())
            r2_scores.append(r2.item())
    
    avg_loss = np.mean(losses)
    avg_r2 = np.mean(r2_scores)
    
    logger.log(f"  Validation - Loss: {avg_loss:.6f} | R²: {avg_r2:.6f}")
    
    return avg_loss, avg_r2


def run_experiment(model_type, epochs=251, noise_config=None):
    """
    Run full experiment for specified model type.
    
    Args:
        model_type: 'quantum' or 'classical'
        epochs: Number of training epochs
        noise_config: Noise configuration (only for quantum)
    """
    # Create experiment name
    exp_name = f"{model_type}"
    if noise_config:
        exp_name += "_with_noise"
    
    logger = ExperimentLogger(exp_name)
    logger.log(f"Model type: {model_type}")
    logger.log(f"Epochs: {epochs}")
    if noise_config:
        logger.log(f"Noise config: {noise_config}")
    
    # Set seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.log(f"Seed: {seed}")
    
    # Load data
    logger.log("\nLoading data...")
    training_data, gt_training = load_data(0, 960)
    validation_data, gt_validation = load_data(960, 1260)
    
    # Use full dataset (as in original training)
    train_samples = 945
    val_samples = 285
    
    training_dataset, train_gt = Create_dataset(training_data, gt_training, num_sample=train_samples)
    validation_dataset, val_gt = Create_dataset(validation_data, gt_validation, num_sample=val_samples)
    
    logger.log(f"Training samples: {train_samples}")
    logger.log(f"Validation samples: {val_samples}")
    
    # Create dataloaders
    batch_size = 315  # As in improved training script
    trainloader = DataLoader(
        TensorDataset(training_dataset, train_gt[:, 4].reshape(-1, 1)),
        batch_size=batch_size,
        shuffle=True,
    )
    valloader = DataLoader(
        TensorDataset(validation_dataset, val_gt[:, 4].reshape(-1, 1)),
        batch_size=285,
        shuffle=False,
    )
    
    # Create model
    logger.log("\nCreating model...")
    if model_type == 'quantum':
        from networks.SQUARE_Mamba import make_model
        model = make_model(
            in_channel=105,
            noise_config=noise_config,
            quantum_device=None,
            shots=None,
        ).to(device)
    elif model_type == 'classical':
        from networks.SQUARE_Mamba_Not_Quantum import make_model as make_classical_model
        model = make_classical_model(in_channel=105).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.log("Model created successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Total parameters: {total_params:,}")
    logger.log(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0001,
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * 3,
        eta_min=1e-7
    )
    
    loss_fn = nn.MSELoss()
    
    # Training loop
    logger.log("\n" + "="*70)
    logger.log("STARTING TRAINING")
    logger.log("="*70)
    
    start_time = time.time()
    best_r2 = -999
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        logger.log(f"\n{'='*70}")
        logger.log(f"EPOCH {epoch}/{epochs}")
        logger.log(f"{'='*70}")
        logger.log(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train
        logger.log("\nTraining...")
        train_loss = train_epoch(model, trainloader, optimizer, loss_fn, epoch, logger)
        logger.metrics['train_losses'].append(float(train_loss))
        
        # Validate
        logger.log("\nValidating...")
        val_loss, val_r2 = validate_epoch(model, valloader, loss_fn, epoch, logger)
        logger.metrics['val_losses'].append(float(val_loss))
        logger.metrics['val_r2s'].append(float(val_r2))
        
        # Update scheduler
        scheduler.step()
        
        # Track best model
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_epoch = epoch
            logger.metrics['best_r2'] = float(best_r2)
            logger.metrics['best_epoch'] = best_epoch
            logger.log(f"\n🎯 NEW BEST R²: {best_r2:.6f} at epoch {epoch}")
            
            # Save best model
            best_path = logger.exp_dir / "checkpoint_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': val_r2,
                'val_loss': val_loss,
            }, best_path)
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        logger.metrics['epoch_times'].append(float(epoch_time))
        
        logger.log(f"\nEpoch {epoch} Summary:")
        logger.log(f"  Train Loss: {train_loss:.6f}")
        logger.log(f"  Val Loss:   {val_loss:.6f}")
        logger.log(f"  Val R²:     {val_r2:.6f}")
        logger.log(f"  Best R²:    {best_r2:.6f} (epoch {best_epoch})")
        logger.log(f"  Time:       {epoch_time:.2f}s")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = logger.save_checkpoint(model, epoch, optimizer)
            logger.log(f"  Checkpoint saved: {checkpoint_path.name}")
        
        # Save metrics
        logger.save_metrics()
        
        # Estimate remaining time
        elapsed = time.time() - start_time
        avg_epoch_time = elapsed / epoch
        remaining_epochs = epochs - epoch
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        logger.log(f"\nProgress: {epoch}/{epochs} ({epoch/epochs*100:.1f}%)")
        logger.log(f"Elapsed: {elapsed/3600:.2f}h | "
                  f"Estimated remaining: {estimated_remaining/3600:.2f}h | "
                  f"Total estimated: {(elapsed + estimated_remaining)/3600:.2f}h")
    
    # Final summary
    total_time = time.time() - start_time
    
    logger.log("\n" + "="*70)
    logger.log("TRAINING COMPLETE")
    logger.log("="*70)
    logger.log(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)")
    logger.log(f"Average time per epoch: {total_time/epochs:.2f}s")
    logger.log(f"\nFinal Results:")
    logger.log(f"  Best R²: {best_r2:.6f} at epoch {best_epoch}")
    logger.log(f"  Final Train Loss: {logger.metrics['train_losses'][-1]:.6f}")
    logger.log(f"  Final Val Loss: {logger.metrics['val_losses'][-1]:.6f}")
    logger.log(f"  Final Val R²: {logger.metrics['val_r2s'][-1]:.6f}")
    
    # Save final checkpoint
    final_path = logger.save_checkpoint(model, epochs, optimizer)
    logger.log(f"\nFinal checkpoint: {final_path}")
    logger.log(f"Best checkpoint: {logger.exp_dir / 'checkpoint_best.pt'}")
    logger.log(f"Metrics file: {logger.metrics_file}")
    
    return logger.exp_dir, logger.metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Full-scale SQUARE_Mamba experiment")
    parser.add_argument('--model', choices=['quantum', 'classical', 'both'],
                       default='both', help='Model type to train')
    parser.add_argument('--epochs', type=int, default=251,
                       help='Number of epochs')
    parser.add_argument('--noise', type=str, default=None,
                       help='Noise specification (only for quantum model)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("FULL-SCALE SQUARE_MAMBA EXPERIMENT")
    print("="*70)
    print(f"Model(s): {args.model}")
    print(f"Epochs: {args.epochs}")
    if args.noise:
        print(f"Noise: {args.noise}")
    print("="*70)
    print()
    
    results = {}
    
    if args.model in ['quantum', 'both']:
        print("\n" + "🔬"*35)
        print("EXPERIMENT 1: QUANTUM MODEL")
        print("🔬"*35 + "\n")
        
        noise_config = parse_noise_spec(args.noise) if args.noise else None
        exp_dir, metrics = run_experiment('quantum', args.epochs, noise_config)
        results['quantum'] = {'dir': exp_dir, 'metrics': metrics}
    
    if args.model in ['classical', 'both']:
        print("\n" + "⚙️"*35)
        print("EXPERIMENT 2: CLASSICAL MODEL")
        print("⚙️"*35 + "\n")
        
        exp_dir, metrics = run_experiment('classical', args.epochs, None)
        results['classical'] = {'dir': exp_dir, 'metrics': metrics}
    
    # Final comparison
    if args.model == 'both':
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        
        q_metrics = results['quantum']['metrics']
        c_metrics = results['classical']['metrics']
        
        print(f"\nQuantum Model:")
        print(f"  Best R²: {q_metrics['best_r2']:.6f} at epoch {q_metrics['best_epoch']}")
        print(f"  Final R²: {q_metrics['val_r2s'][-1]:.6f}")
        print(f"  Output: {results['quantum']['dir']}")
        
        print(f"\nClassical Model:")
        print(f"  Best R²: {c_metrics['best_r2']:.6f} at epoch {c_metrics['best_epoch']}")
        print(f"  Final R²: {c_metrics['val_r2s'][-1]:.6f}")
        print(f"  Output: {results['classical']['dir']}")
        
        print(f"\nDifference (Quantum - Classical):")
        best_diff = q_metrics['best_r2'] - c_metrics['best_r2']
        final_diff = q_metrics['val_r2s'][-1] - c_metrics['val_r2s'][-1]
        print(f"  Best R²: {best_diff:+.6f}")
        print(f"  Final R²: {final_diff:+.6f}")
        
        if best_diff > 0:
            print(f"\n✓ Quantum model achieved better performance!")
        elif best_diff < 0:
            print(f"\n✓ Classical model achieved better performance!")
        else:
            print(f"\n= Models achieved similar performance")
        
        print("="*70)
    
    print("\n✓ All experiments complete!")
