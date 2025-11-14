#!/usr/bin/env python3
"""
Entrenamiento ULTRA RÁPIDO para pruebas de ruido.
Usa datos mínimos y solo 1 época para ver el efecto del ruido rápidamente.
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Agregar path del modelo
sys.path.insert(0, str(Path(__file__).parent / "SQUARE_Mamba" / "main"))

from functions.util import Create_dataset, load_data, parse_noise_spec, r_square
from networks.SQUARE_Mamba import make_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quick_train_test(noise_config=None, experiment_name="test"):
    """Entrenamiento ultra rápido para verificar que el ruido funciona"""
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO: {experiment_name}")
    print(f"{'='*60}")
    
    if noise_config:
        print(f"Ruido: {noise_config}")
    else:
        print("Sin ruido")
    
    # Seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Datos MÍNIMOS
    print("\nCargando datos mínimos...")
    training_data, gt_training = load_data(0, 960)
    
    # Solo 5 muestras para ser super rápido
    train_samples = 5
    training_dataset, train_gt = Create_dataset(training_data, gt_training, num_sample=train_samples)
    
    # Batch de todas las muestras a la vez
    trainloader = DataLoader(
        TensorDataset(training_dataset, train_gt[:, 4].reshape(-1, 1)),
        batch_size=train_samples,
        shuffle=False,
    )
    
    # Modelo
    print("Creando modelo...")
    start_model = time.time()
    model = make_model(
        in_channel=105,
        noise_config=noise_config,
        quantum_device=None,
        shots=None,
    ).to(device)
    model_time = time.time() - start_model
    print(f"  Tiempo de creación: {model_time:.2f}s")
    
    # Optimizador
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # UNA SOLA iteración de entrenamiento
    print("\nEjecutando forward + backward pass...")
    model.train()
    
    data, target = next(iter(trainloader))
    data, target = data.to(device), target.to(device)
    
    # Forward
    start_forward = time.time()
    optimizer.zero_grad()
    output = model(data)
    forward_time = time.time() - start_forward
    
    # Loss
    loss = loss_fn(target, output)
    r2 = r_square(target, output)
    
    # Backward
    start_backward = time.time()
    loss.backward()
    backward_time = time.time() - start_backward
    
    # Optimizer step
    start_step = time.time()
    optimizer.step()
    step_time = time.time() - start_step
    
    print(f"\n{'='*60}")
    print("RESULTADOS:")
    print(f"{'='*60}")
    print(f"Loss:           {loss.item():.6f}")
    print(f"R²:             {r2.item():.6f}")
    print(f"Forward time:   {forward_time:.2f}s")
    print(f"Backward time:  {backward_time:.2f}s")
    print(f"Step time:      {step_time:.2f}s")
    print(f"Total:          {forward_time + backward_time + step_time:.2f}s")
    print(f"{'='*60}")
    
    return {
        'experiment': experiment_name,
        'loss': loss.item(),
        'r2': r2.item(),
        'forward_time': forward_time,
        'backward_time': backward_time,
        'total_time': forward_time + backward_time + step_time,
        'noise_config': noise_config,
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PRUEBA ULTRA RÁPIDA DE RUIDO CUÁNTICO")
    print("="*60)
    print("\nEsto ejecutará 2 experimentos con solo 5 muestras")
    print("para verificar rápidamente que el ruido funciona.\n")
    
    results = []
    
    # Experimento 1: Sin ruido
    print("\n" + "▶"*30)
    print("EXPERIMENTO 1/2: Baseline (sin ruido)")
    print("▶"*30)
    result1 = quick_train_test(
        noise_config=None,
        experiment_name="Baseline"
    )
    results.append(result1)
    
    # Experimento 2: Con ruido
    print("\n" + "▶"*30)
    print("EXPERIMENTO 2/2: Con ruido depolarizing")
    print("▶"*30)
    noise_config = parse_noise_spec("depolarizing=0.02")
    result2 = quick_train_test(
        noise_config=noise_config,
        experiment_name="Depolarizing 0.02"
    )
    results.append(result2)
    
    # Comparación
    print("\n\n" + "="*60)
    print("COMPARACIÓN FINAL")
    print("="*60)
    
    baseline = results[0]
    noisy = results[1]
    
    loss_diff = noisy['loss'] - baseline['loss']
    r2_diff = noisy['r2'] - baseline['r2']
    time_diff = noisy['total_time'] - baseline['total_time']
    
    print(f"\nBaseline:")
    print(f"  Loss: {baseline['loss']:.6f}")
    print(f"  R²:   {baseline['r2']:.6f}")
    print(f"  Time: {baseline['total_time']:.2f}s")
    
    print(f"\nCon ruido (depolarizing=0.02):")
    print(f"  Loss: {noisy['loss']:.6f}")
    print(f"  R²:   {noisy['r2']:.6f}")
    print(f"  Time: {noisy['total_time']:.2f}s")
    
    print(f"\nDiferencias (Ruido - Baseline):")
    print(f"  ΔLoss: {loss_diff:+.6f} ({loss_diff/baseline['loss']*100:+.2f}%)")
    print(f"  ΔR²:   {r2_diff:+.6f} ({r2_diff/baseline['r2']*100:+.2f}%)")
    print(f"  ΔTime: {time_diff:+.2f}s ({time_diff/baseline['total_time']*100:+.2f}%)")
    
    print("\n" + "="*60)
    
    # Verificación
    if abs(loss_diff) > 0.0001 or abs(r2_diff) > 0.0001:
        print("✓ EL RUIDO TIENE EFECTO MEDIBLE")
        print("  El modelo produce valores diferentes con ruido.")
    else:
        print("⚠ El efecto del ruido es muy pequeño")
        print("  Puede necesitar más iteraciones o mayor nivel de ruido.")
    
    print("="*60)
    print("\n✓ Prueba completada!")
