#!/usr/bin/env python3
"""
Plot and analyze results from full-scale experiments.
"""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np


def load_metrics(experiment_dir):
    """Load metrics from experiment directory"""
    metrics_file = Path(experiment_dir) / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    with open(metrics_file, 'r') as f:
        return json.load(f)


def plot_comparison(quantum_dir, classical_dir, output_file="comparison.png"):
    """Plot comparison between quantum and classical models"""
    
    # Load metrics
    q_metrics = load_metrics(quantum_dir)
    c_metrics = load_metrics(classical_dir)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Quantum vs Classical SQUARE-Mamba Comparison', fontsize=16, fontweight='bold')
    
    epochs_q = range(1, len(q_metrics['train_losses']) + 1)
    epochs_c = range(1, len(c_metrics['train_losses']) + 1)
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(epochs_q, q_metrics['train_losses'], label='Quantum', linewidth=2, alpha=0.8)
    ax.plot(epochs_c, c_metrics['train_losses'], label='Classical', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax = axes[0, 1]
    ax.plot(epochs_q, q_metrics['val_losses'], label='Quantum', linewidth=2, alpha=0.8)
    ax.plot(epochs_c, c_metrics['val_losses'], label='Classical', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation R²
    ax = axes[1, 0]
    ax.plot(epochs_q, q_metrics['val_r2s'], label='Quantum', linewidth=2, alpha=0.8)
    ax.plot(epochs_c, c_metrics['val_r2s'], label='Classical', linewidth=2, alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation R²')
    ax.set_title('Validation R² Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Epoch Time
    ax = axes[1, 1]
    if q_metrics['epoch_times'] and c_metrics['epoch_times']:
        ax.plot(epochs_q, q_metrics['epoch_times'], label='Quantum', linewidth=2, alpha=0.8)
        ax.plot(epochs_c, c_metrics['epoch_times'], label='Classical', linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time per Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Print statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    
    print("\nQuantum Model:")
    print(f"  Best R²: {q_metrics['best_r2']:.6f} at epoch {q_metrics['best_epoch']}")
    print(f"  Final Train Loss: {q_metrics['train_losses'][-1]:.6f}")
    print(f"  Final Val Loss: {q_metrics['val_losses'][-1]:.6f}")
    print(f"  Final Val R²: {q_metrics['val_r2s'][-1]:.6f}")
    if q_metrics['epoch_times']:
        print(f"  Avg time/epoch: {np.mean(q_metrics['epoch_times']):.2f}s")
    
    print("\nClassical Model:")
    print(f"  Best R²: {c_metrics['best_r2']:.6f} at epoch {c_metrics['best_epoch']}")
    print(f"  Final Train Loss: {c_metrics['train_losses'][-1]:.6f}")
    print(f"  Final Val Loss: {c_metrics['val_losses'][-1]:.6f}")
    print(f"  Final Val R²: {c_metrics['val_r2s'][-1]:.6f}")
    if c_metrics['epoch_times']:
        print(f"  Avg time/epoch: {np.mean(c_metrics['epoch_times']):.2f}s")
    
    print("\nDifference (Quantum - Classical):")
    best_diff = q_metrics['best_r2'] - c_metrics['best_r2']
    final_diff = q_metrics['val_r2s'][-1] - c_metrics['val_r2s'][-1]
    print(f"  Best R²: {best_diff:+.6f} ({best_diff/abs(c_metrics['best_r2'])*100:+.2f}%)")
    print(f"  Final R²: {final_diff:+.6f} ({final_diff/abs(c_metrics['val_r2s'][-1])*100:+.2f}%)")
    
    if q_metrics['epoch_times'] and c_metrics['epoch_times']:
        time_diff = np.mean(q_metrics['epoch_times']) - np.mean(c_metrics['epoch_times'])
        print(f"  Avg time/epoch: {time_diff:+.2f}s ({time_diff/np.mean(c_metrics['epoch_times'])*100:+.2f}%)")


def plot_single(experiment_dir, output_file="training_curves.png"):
    """Plot training curves for a single experiment"""
    
    metrics = load_metrics(experiment_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    exp_name = Path(experiment_dir).name
    fig.suptitle(f'Training Curves: {exp_name}', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(epochs, metrics['train_losses'], linewidth=2, color='blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax = axes[0, 1]
    ax.plot(epochs, metrics['val_losses'], linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation R²
    ax = axes[1, 0]
    ax.plot(epochs, metrics['val_r2s'], linewidth=2, color='green')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=metrics['best_r2'], color='red', linestyle='--', alpha=0.5, 
               label=f"Best: {metrics['best_r2']:.4f}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation R²')
    ax.set_title('Validation R²')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Epoch Time
    ax = axes[1, 1]
    if metrics['epoch_times']:
        ax.plot(epochs, metrics['epoch_times'], linewidth=2, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time per Epoch')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Print statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Best R²: {metrics['best_r2']:.6f} at epoch {metrics['best_epoch']}")
    print(f"Final Train Loss: {metrics['train_losses'][-1]:.6f}")
    print(f"Final Val Loss: {metrics['val_losses'][-1]:.6f}")
    print(f"Final Val R²: {metrics['val_r2s'][-1]:.6f}")
    if metrics['epoch_times']:
        print(f"Avg time/epoch: {np.mean(metrics['epoch_times']):.2f}s")
        print(f"Total time: {sum(metrics['epoch_times'])/3600:.2f}h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument('--quantum', type=str, help='Path to quantum experiment directory')
    parser.add_argument('--classical', type=str, help='Path to classical experiment directory')
    parser.add_argument('--single', type=str, help='Path to single experiment directory')
    parser.add_argument('--output', type=str, default='experiment_plot.png',
                       help='Output file name')
    
    args = parser.parse_args()
    
    if args.single:
        plot_single(args.single, args.output)
    elif args.quantum and args.classical:
        plot_comparison(args.quantum, args.classical, args.output)
    else:
        print("Error: Provide either --single or both --quantum and --classical")
        print("\nExamples:")
        print("  # Plot single experiment")
        print("  python plot_experiment_results.py --single experiments/quantum_20241114_120000")
        print("\n  # Compare quantum vs classical")
        print("  python plot_experiment_results.py \\")
        print("      --quantum experiments/quantum_20241114_120000 \\")
        print("      --classical experiments/classical_20241114_130000")
