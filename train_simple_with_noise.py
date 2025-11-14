#!/usr/bin/env python3
"""
Entrenamiento simple del modelo SQUARE_Mamba con y sin ruido cuántico.
Usa pocos epochs y datos reducidos para pruebas rápidas.
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
print(f"Usando dispositivo: {device}")


def train_epoch(model, dataloader, optimizer, loss_fn, epoch):
    """Entrena una época"""
    model.train()
    losses = []
    
    print(f"\n[Época {epoch}] Entrenando...")
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(target, output)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Batch {batch_idx+1}/{len(dataloader)}: loss={loss.item():.6f}")
    
    avg_loss = np.mean(losses)
    print(f"[Época {epoch}] Loss promedio: {avg_loss:.6f}")
    return avg_loss


def validate_epoch(model, dataloader, loss_fn, epoch):
    """Valida una época"""
    model.eval()
    losses = []
    r2_scores = []
    
    print(f"\n[Época {epoch}] Validando...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = loss_fn(target, output)
            r2 = r_square(target, output)
            
            losses.append(loss.item())
            r2_scores.append(r2.item())
            print(f"  Batch {batch_idx+1}/{len(dataloader)}: loss={loss.item():.6f}, R²={r2.item():.6f}")
    
    avg_loss = np.mean(losses)
    avg_r2 = np.mean(r2_scores)
    print(f"[Época {epoch}] Validación - Loss: {avg_loss:.6f}, R²: {avg_r2:.6f}")
    return avg_loss, avg_r2


def train_model(noise_config=None, epochs=3, experiment_name="baseline"):
    """
    Entrena el modelo con la configuración especificada.
    
    Args:
        noise_config: Configuración de ruido (None para sin ruido)
        epochs: Número de épocas
        experiment_name: Nombre del experimento para logging
    """
    print("\n" + "="*70)
    print(f"EXPERIMENTO: {experiment_name}")
    print("="*70)
    
    if noise_config:
        print(f"Configuración de ruido: {noise_config}")
    else:
        print("Sin ruido (baseline)")
    
    # Configurar seed para reproducibilidad
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")
    
    # Cargar datos (reducidos para pruebas rápidas)
    print("\nCargando datos...")
    training_data, gt_training = load_data(0, 960)
    validation_data, gt_validation = load_data(960, 1260)
    
    # Usar menos muestras para entrenamiento rápido
    train_samples = 30  # Reducido de 60
    val_samples = 15    # Reducido de 60
    
    training_dataset, train_gt = Create_dataset(training_data, gt_training, num_sample=train_samples)
    validation_dataset, val_gt = Create_dataset(validation_data, gt_validation, num_sample=val_samples)
    
    print(f"Muestras de entrenamiento: {train_samples}")
    print(f"Muestras de validación: {val_samples}")
    
    # Crear dataloaders
    batch_size = 10  # Batch pequeño para ver progreso
    trainloader = DataLoader(
        TensorDataset(training_dataset, train_gt[:, 4].reshape(-1, 1)),
        batch_size=batch_size,
        shuffle=True,
    )
    valloader = DataLoader(
        TensorDataset(validation_dataset, val_gt[:, 4].reshape(-1, 1)),
        batch_size=batch_size,
        shuffle=False,
    )
    
    # Crear modelo
    print("\nCreando modelo...")
    model = make_model(
        in_channel=105,
        noise_config=noise_config,
        quantum_device=None,
        shots=None,
    ).to(device)
    
    # Configurar optimizador y loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0001)
    loss_fn = nn.MSELoss()
    
    # Entrenar
    print(f"\nIniciando entrenamiento por {epochs} épocas...")
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    val_r2s = []
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Entrenar
        train_loss = train_epoch(model, trainloader, optimizer, loss_fn, epoch)
        train_losses.append(train_loss)
        
        # Validar
        val_loss, val_r2 = validate_epoch(model, valloader, loss_fn, epoch)
        val_losses.append(val_loss)
        val_r2s.append(val_r2)
        
        epoch_time = time.time() - epoch_start
        print(f"\n[Época {epoch}] Tiempo: {epoch_time:.2f}s")
        print("-" * 70)
    
    total_time = time.time() - start_time
    
    # Resumen
    print("\n" + "="*70)
    print(f"RESUMEN - {experiment_name}")
    print("="*70)
    print(f"Tiempo total: {total_time:.2f}s ({total_time/epochs:.2f}s por época)")
    print(f"\nLoss de entrenamiento:")
    print(f"  Inicial: {train_losses[0]:.6f}")
    print(f"  Final:   {train_losses[-1]:.6f}")
    print(f"  Cambio:  {train_losses[-1] - train_losses[0]:.6f}")
    print(f"\nLoss de validación:")
    print(f"  Inicial: {val_losses[0]:.6f}")
    print(f"  Final:   {val_losses[-1]:.6f}")
    print(f"  Cambio:  {val_losses[-1] - val_losses[0]:.6f}")
    print(f"\nR² de validación:")
    print(f"  Inicial: {val_r2s[0]:.6f}")
    print(f"  Final:   {val_r2s[-1]:.6f}")
    print(f"  Cambio:  {val_r2s[-1] - val_r2s[0]:.6f}")
    print("="*70)
    
    return {
        'experiment': experiment_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_r2s': val_r2s,
        'total_time': total_time,
        'noise_config': noise_config,
    }


def compare_experiments(results_list):
    """Compara los resultados de múltiples experimentos"""
    print("\n" + "="*70)
    print("COMPARACIÓN DE EXPERIMENTOS")
    print("="*70)
    
    for result in results_list:
        name = result['experiment']
        final_loss = result['val_losses'][-1]
        final_r2 = result['val_r2s'][-1]
        time_per_epoch = result['total_time'] / len(result['val_losses'])
        
        print(f"\n{name}:")
        print(f"  Loss final:     {final_loss:.6f}")
        print(f"  R² final:       {final_r2:.6f}")
        print(f"  Tiempo/época:   {time_per_epoch:.2f}s")
        if result['noise_config']:
            print(f"  Ruido:          {result['noise_config']}")
    
    # Comparar baseline vs ruido
    if len(results_list) >= 2:
        baseline = results_list[0]
        noisy = results_list[1]
        
        loss_diff = noisy['val_losses'][-1] - baseline['val_losses'][-1]
        r2_diff = noisy['val_r2s'][-1] - baseline['val_r2s'][-1]
        
        print(f"\nDiferencia (Ruido - Baseline):")
        print(f"  ΔLoss: {loss_diff:+.6f} ({loss_diff/baseline['val_losses'][-1]*100:+.2f}%)")
        print(f"  ΔR²:   {r2_diff:+.6f} ({r2_diff/baseline['val_r2s'][-1]*100:+.2f}%)")
        
        if abs(loss_diff) > 0.001:
            print(f"\n✓ El ruido tiene un efecto medible en el entrenamiento")
        else:
            print(f"\n⚠ El efecto del ruido es muy pequeño (puede necesitar más épocas)")
    
    print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("ENTRENAMIENTO SIMPLE CON RUIDO CUÁNTICO")
    print("="*70)
    
    epochs = 3
    results = []
    
    # Experimento 1: Sin ruido (baseline)
    print("\n\n")
    result_baseline = train_model(
        noise_config=None,
        epochs=epochs,
        experiment_name="Baseline (sin ruido)"
    )
    results.append(result_baseline)
    
    # Experimento 2: Con ruido depolarizing
    print("\n\n")
    noise_spec = "depolarizing=0.02"
    noise_config = parse_noise_spec(noise_spec)
    result_noisy = train_model(
        noise_config=noise_config,
        epochs=epochs,
        experiment_name=f"Con ruido ({noise_spec})"
    )
    results.append(result_noisy)
    
    # Comparar resultados
    compare_experiments(results)
    
    print("\n✓ Entrenamiento completado!")
