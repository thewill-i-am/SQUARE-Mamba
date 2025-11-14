#!/usr/bin/env python3
"""
Quick test script to verify the quantum circuit example works.
"""
import sys
from pathlib import Path

# Add SQUARE_Mamba/main to path
sys.path.insert(0, str(Path(__file__).parent / "SQUARE_Mamba" / "main"))

from networks.SQUARE_Mamba import run_two_qubit_qnn_example

if __name__ == "__main__":
    print("Testing run_two_qubit_qnn_example()...")
    print("-" * 50)
    
    try:
        output = run_two_qubit_qnn_example()
        print(f"Success! Output shape: {output.shape}")
        print(f"Output values:\n{output}")
        print("-" * 50)
        print("✓ Quantum circuit example is working!")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
