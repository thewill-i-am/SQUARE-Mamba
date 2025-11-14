# Guía de Configuración y Pruebas

## Estado Actual

✅ **El código cuántico funciona correctamente**

El sistema está operativo con las siguientes configuraciones:

### Entorno de Trabajo
- **Python**: 3.10.13 (`.venv-3.10/`)
- **PyTorch**: 2.2.1
- **Qiskit**: 1.4.3
- **Qiskit Aer**: 0.17.2+
- **Qiskit Machine Learning**: 0.8.4
- **NumPy**: 1.26.4

### Conflictos de Dependencias Identificados

1. **NumPy Version Conflict**:
   - PennyLane requiere: `numpy<2.0`
   - Qiskit ML requiere: `numpy>=2.0`
   - **Solución actual**: Usar numpy 1.26.4 (funciona con warnings)

2. **OpenMP Duplicate Library**:
   - Múltiples librerías cargan OpenMP
   - **Solución**: Configurar `KMP_DUPLICATE_LIB_OK=TRUE`

## Cómo Ejecutar las Pruebas

### Opción 1: Usar el script helper (Recomendado)

```bash
./run_test.sh test_simple_quantum.py
```

Este script automáticamente:
- Configura las variables de entorno necesarias
- Usa el Python correcto (`.venv-3.10/bin/python`)
- Ejecuta el test

### Opción 2: Comando directo

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv-3.10/bin/python test_simple_quantum.py
```

## Pruebas Disponibles

### 1. `test_simple_quantum.py` - Prueba Completa del Sistema
Verifica paso a paso:
- ✓ Importación de Qiskit
- ✓ Importación de Qiskit Aer
- ✓ Importación de Qiskit Machine Learning
- ✓ Importación de PyTorch
- ✓ Creación de circuito cuántico
- ✓ Creación de observables
- ✓ Creación de Estimator
- ✓ Creación de EstimatorQNN
- ✓ Creación de TorchConnector
- ✓ Forward pass
- ✓ Backward pass (gradientes)

**Ejecutar**:
```bash
./run_test.sh test_simple_quantum.py
```

### 2. `test_qnn_example.py` - Ejemplo del Modelo
Ejecuta la función `run_two_qubit_qnn_example()` del modelo SQUARE_Mamba.

**Ejecutar**:
```bash
./run_test.sh test_qnn_example.py
```

## Resultados de las Pruebas

### Test Simple Quantum ✅
```
============================================================
✓ TODAS LAS PRUEBAS PASARON EXITOSAMENTE
El sistema está listo para entrenar modelos cuánticos!
============================================================
```

**Output del forward pass**:
```
tensor([[0.9727, 0.9980],
        [0.7246, 0.8145]], grad_fn=<_TorchNNFunctionBackward>)
```

**Gradientes calculados**: ✅
```
Gradient shape: torch.Size([2])
```

### Test QNN Example ✅
```
Success! Output shape: torch.Size([2, 1])
Output values:
tensor([[-0.9064],
        [-0.6531]], grad_fn=<AddmmBackward0>)
✓ Quantum circuit example is working!
```

## Warnings Conocidos (No Críticos)

1. **DeprecationWarning: Estimator V1**
   - Qiskit Aer recomienda usar EstimatorV2
   - El código actual funciona pero debería actualizarse en el futuro

2. **DeprecationWarning: V1 Primitives**
   - Qiskit ML recomienda usar V2 primitives
   - Funcional pero deprecado

3. **UserWarning: torch.tensor() construction**
   - Recomendación de usar `.clone().detach()`
   - No afecta funcionalidad

## Próximos Pasos para el Spec

Ahora que confirmamos que el código cuántico funciona, podemos proceder con las tareas del spec:

### Tareas Prioritarias (del spec)

1. **Tarea 1**: Arreglar parámetros duplicados
   - Buscar `noise_config=noise_config,` duplicado
   - Limpiar código

2. **Tarea 2**: Agregar soporte para seed
   - Implementar reproducibilidad
   - Agregar logging

3. **Tarea 8**: Validar inyección de ruido
   - Ejecutar entrenamientos con/sin ruido
   - Comparar métricas

### Comandos para Entrenar (Cuando esté listo)

```bash
# Sin ruido (baseline)
./run_test.sh SQUARE_Mamba/main/train_SQUARE_Mamba.py --epochs 3

# Con ruido depolarizing
./run_test.sh SQUARE_Mamba/main/train_SQUARE_Mamba.py --epochs 3 --noise depolarizing=0.02

# Con múltiples canales de ruido
./run_test.sh SQUARE_Mamba/main/train_SQUARE_Mamba.py --epochs 3 --noise depolarizing=0.02,amplitude_damping=0.05
```

## Recomendaciones de Dependencias

Para evitar conflictos futuros, considera:

### Opción A: Mantener configuración actual
- Funciona con warnings
- Requiere `KMP_DUPLICATE_LIB_OK=TRUE`

### Opción B: Actualizar a Qiskit V2 Primitives
- Elimina deprecation warnings
- Requiere refactorizar código del modelo
- Más trabajo pero más sostenible

### Opción C: Separar entornos
- Un venv para PennyLane (numpy<2.0)
- Otro venv para Qiskit (numpy>=2.0)
- Más complejo de mantener

## Archivo de Dependencias Recomendado

Ver `requirements-fixed.txt` para una versión limpia con rangos de versiones compatibles.

## Troubleshooting

### Error: "No module named 'torch'"
```bash
# Verificar que estás usando el venv correcto
.venv-3.10/bin/python --version
```

### Error: "OMP: Error #15"
```bash
# Agregar variable de entorno
export KMP_DUPLICATE_LIB_OK=TRUE
# O usar el script helper
./run_test.sh <tu_script.py>
```

### Error: "numpy version conflict"
```bash
# Instalar versión compatible
.venv-3.10/bin/python -m pip install "numpy>=1.24,<2.0"
```
