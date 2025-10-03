#!/bin/bash

# Script para ejecutar benchmarks con diferentes configuraciones de ruido cuántico

for noise in "" "depolarizing=0.02" "depolarizing=0.02,phase_damping=0.05"; do
  # Generar etiqueta para el archivo de log
  if [ -z "$noise" ]; then
    label="none"
  else
    # Reemplazar caracteres especiales por guiones bajos
    label=$(echo "$noise" | sed 's/[^a-z0-9]/_/g')
  fi
  
  echo "=== Benchmark noise=${noise:-none} ==="
  
  # Construir comando según si hay ruido o no
  if [ -z "$noise" ]; then
    python3 SQUARE_Mamba/main/test_SQUARE_Mamba.py \
      2>&1 | tee "exports/benchmark_${label}.log"
  else
    python3 SQUARE_Mamba/main/test_SQUARE_Mamba.py \
      --noise "$noise" \
      --noise-device default.mixed \
      --noise-shots 4096 \
      2>&1 | tee "exports/benchmark_${label}.log"
  fi
  
  echo "=== Benchmark completado: ${label} ==="
  echo ""
done

echo "🎉 Todos los benchmarks completados!"
echo "📁 Logs guardados en: exports/benchmark_*.log"