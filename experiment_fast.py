#!/usr/bin/env python3
"""
FAST version: Reduced samples for quicker experiments
Uses 100 samples instead of 945 for ~10x speedup
"""
import sys
from pathlib import Path

# Reuse the main experiment code but with modified parameters
sys.path.insert(0, str(Path(__file__).parent))

# Import and modify
import experiment_full_comparison as exp

# Override the sample counts
original_run = exp.run_experiment

def fast_run_experiment(model_type, epochs=10, noise_config=None):
    """Fast version with reduced samples"""
    # Temporarily patch the function
    import types
    
    def patched_run(model_type, epochs, noise_config):
        result = original_run(model_type, epochs, noise_config)
        return result
    
    # Modify sample counts in the original function
    exp.train_samples = 100  # Instead of 945
    exp.val_samples = 30     # Instead of 285
    
    return original_run(model_type, epochs, noise_config)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FAST SQUARE_Mamba experiment")
    parser.add_argument('--model', choices=['quantum', 'classical', 'both'],
                       default='both', help='Model type to train')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    
    args = parser.parse_args()
    
    print("="*70)
    print("FAST EXPERIMENT (100 samples)")
    print("="*70)
    print(f"Expected time: ~10-15 minutes for {args.epochs} epochs")
    print("="*70)
    print()
    
    # Modify the run_experiment function to use fewer samples
    original_code = exp.run_experiment.__code__
    
    # Just run with modified globals
    if args.model in ['quantum', 'both']:
        print("\n🔬 QUANTUM MODEL (FAST)")
        # Monkey patch the sample counts
        import experiment_full_comparison
        original_train = 945
        original_val = 285
        
        # Replace in the source
        exp_source = Path('experiment_full_comparison.py').read_text()
        exp_source = exp_source.replace('train_samples = 945', 'train_samples = 100')
        exp_source = exp_source.replace('val_samples = 285', 'val_samples = 30')
        
        # Write temporary file
        temp_file = Path('experiment_temp_fast.py')
        temp_file.write_text(exp_source)
        
        # Import and run
        import importlib.util
        spec = importlib.util.spec_from_file_location("exp_fast", temp_file)
        exp_fast = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp_fast)
        
        exp_fast.run_experiment('quantum', args.epochs, None)
        
        # Cleanup
        temp_file.unlink()
