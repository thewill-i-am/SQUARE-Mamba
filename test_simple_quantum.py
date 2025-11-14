#!/usr/bin/env python3
"""
Ejemplo simple para probar que Qiskit funciona correctamente.
No requiere el modelo completo de SQUARE_Mamba.
"""

def test_qiskit_basic():
    """Prueba básica de Qiskit"""
    print("1. Probando importación de Qiskit...")
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp
        print("   ✓ Qiskit importado correctamente")
    except ImportError as e:
        print(f"   ✗ Error importando Qiskit: {e}")
        return False
    
    print("\n2. Probando Qiskit Aer...")
    try:
        from qiskit_aer.primitives import Estimator as AerEstimator
        print("   ✓ Qiskit Aer importado correctamente")
    except ImportError as e:
        print(f"   ✗ Error importando Qiskit Aer: {e}")
        return False
    
    print("\n3. Probando Qiskit Machine Learning...")
    try:
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.connectors import TorchConnector
        print("   ✓ Qiskit Machine Learning importado correctamente")
    except ImportError as e:
        print(f"   ✗ Error importando Qiskit ML: {e}")
        return False
    
    print("\n4. Probando PyTorch...")
    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__} importado correctamente")
    except ImportError as e:
        print(f"   ✗ Error importando PyTorch: {e}")
        return False
    
    print("\n5. Creando circuito cuántico simple...")
    try:
        # Crear un circuito simple de 2 qubits
        data_params = ParameterVector("x", 2)
        weight_params = ParameterVector("θ", 2)
        
        circuit = QuantumCircuit(2)
        
        # Codificar datos
        for idx, param in enumerate(data_params):
            circuit.ry(param, idx)
        
        # Aplicar puerta entrelazada
        circuit.cz(0, 1)
        
        # Aplicar pesos entrenables
        for idx, param in enumerate(weight_params):
            circuit.rx(param, idx)
        
        print(f"   ✓ Circuito creado con {circuit.num_qubits} qubits")
        print(f"   ✓ Parámetros de entrada: {len(data_params)}")
        print(f"   ✓ Parámetros entrenables: {len(weight_params)}")
    except Exception as e:
        print(f"   ✗ Error creando circuito: {e}")
        return False
    
    print("\n6. Creando observables...")
    try:
        observables = [
            SparsePauliOp.from_list([("ZI", 1.0)]),
            SparsePauliOp.from_list([("IZ", 1.0)]),
        ]
        print(f"   ✓ {len(observables)} observables creados")
    except Exception as e:
        print(f"   ✗ Error creando observables: {e}")
        return False
    
    print("\n7. Creando Estimator...")
    try:
        estimator = AerEstimator()
        print("   ✓ Estimator creado correctamente")
    except Exception as e:
        print(f"   ✗ Error creando Estimator: {e}")
        return False
    
    print("\n8. Creando EstimatorQNN...")
    try:
        qnn = EstimatorQNN(
            circuit=circuit,
            observables=observables,
            input_params=list(data_params),
            weight_params=list(weight_params),
            estimator=estimator,
            input_gradients=True,
        )
        print("   ✓ EstimatorQNN creado correctamente")
    except Exception as e:
        print(f"   ✗ Error creando EstimatorQNN: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n9. Creando TorchConnector...")
    try:
        initial_weights = torch.zeros(len(weight_params), dtype=torch.float32)
        torch_layer = TorchConnector(qnn, initial_weights=initial_weights)
        print("   ✓ TorchConnector creado correctamente")
    except Exception as e:
        print(f"   ✗ Error creando TorchConnector: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n10. Probando forward pass...")
    try:
        # Crear datos de entrada de prueba
        test_input = torch.tensor([[0.1, -0.2], [0.6, 0.8]], dtype=torch.float32)
        print(f"   Input shape: {test_input.shape}")
        
        # Ejecutar forward pass
        output = torch_layer(test_input)
        print(f"   ✓ Forward pass exitoso!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output values:\n{output}")
    except Exception as e:
        print(f"   ✗ Error en forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n11. Probando backward pass (gradientes)...")
    try:
        # Calcular una pérdida simple
        loss = output.sum()
        loss.backward()
        
        print(f"   ✓ Backward pass exitoso!")
        print(f"   Loss: {loss.item():.6f}")
        
        # Verificar que hay gradientes
        if torch_layer.weight.grad is not None:
            print(f"   ✓ Gradientes calculados correctamente")
            print(f"   Gradient shape: {torch_layer.weight.grad.shape}")
        else:
            print(f"   ⚠ No se calcularon gradientes")
    except Exception as e:
        print(f"   ✗ Error en backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBA SIMPLE DE QUANTUM NEURAL NETWORK")
    print("=" * 60)
    print()
    
    success = test_qiskit_basic()
    
    print()
    print("=" * 60)
    if success:
        print("✓ TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
        print("El sistema está listo para entrenar modelos cuánticos!")
    else:
        print("✗ ALGUNAS PRUEBAS FALLARON")
        print("Revisa los errores arriba para más detalles.")
    print("=" * 60)
