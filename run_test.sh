#!/bin/bash
# Script helper para correr pruebas con el entorno correcto

# Configurar variable de entorno para evitar conflicto de OpenMP
export KMP_DUPLICATE_LIB_OK=TRUE

# Usar el venv correcto
PYTHON=".venv-3.10/bin/python"

# Verificar que existe
if [ ! -f "$PYTHON" ]; then
    echo "Error: No se encuentra $PYTHON"
    exit 1
fi

echo "Usando Python: $($PYTHON --version)"
echo ""

# Ejecutar el script que se pase como argumento
if [ $# -eq 0 ]; then
    echo "Uso: $0 <script.py> [argumentos...]"
    echo ""
    echo "Ejemplos de pruebas:"
    echo "  $0 test_simple_quantum.py          # Prueba completa del sistema"
    echo "  $0 test_qnn_example.py             # Ejemplo del modelo"
    echo "  $0 train_ultra_fast.py             # Prueba rápida de ruido (~3 min)"
    echo "  $0 train_simple_with_noise.py      # Entrenamiento completo (~20 min)"
    echo ""
    echo "Ejemplos de entrenamiento:"
    echo "  $0 SQUARE_Mamba/main/train_SQUARE_Mamba.py --epochs 3"
    echo "  $0 SQUARE_Mamba/main/train_SQUARE_Mamba.py --epochs 3 --noise depolarizing=0.02"
    exit 1
fi

$PYTHON "$@"
