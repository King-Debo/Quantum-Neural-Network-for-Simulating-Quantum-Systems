# utils.py
# This file contains the utility functions, which are used by the QNN class and the main function.

# Import the libraries
import numpy as np
import pennylane as qml
import torch
import qiskit

# Import the other files
from config import *

# Define the quantum gates function
def quantum_gates(gate, *args, **kwargs):
    # Apply the quantum gate depending on the gate name and the arguments
    if gate == "Hadamard":
        qml.Hadamard(*args, **kwargs)
    elif gate == "PauliX":
        qml.PauliX(*args, **kwargs)
    elif gate == "PauliY":
        qml.PauliY(*args, **kwargs)
    elif gate == "PauliZ":
        qml.PauliZ(*args, **kwargs)
    elif gate == "CNOT":
        qml.CNOT(*args, **kwargs)
    elif gate == "SWAP":
        qml.SWAP(*args, **kwargs)
    elif gate == "ControlledPhaseShift":
        qml.ControlledPhaseShift(*args, **kwargs)
    elif gate == "CRX":
        qml.CRX(*args, **kwargs)
    elif gate == "CRY":
        qml.CRY(*args, **kwargs)
    else:
        raise ValueError(f"Invalid gate name: {gate}")

# Define the quantum operators function
def quantum_operators(operator, *args, **kwargs):
    # Return the quantum operator depending on the operator name and the arguments
    if operator == "Hamiltonian":
        return qml.Hamiltonian(*args, **kwargs)
    elif operator == "UnitaryOperator":
        return qml.UnitaryOperator(*args, **kwargs)
    else:
        raise ValueError(f"Invalid operator name: {operator}")

# Define the quantum circuits function
def quantum_circuits(circuit, *args, **kwargs):
    # Return the quantum circuit depending on the circuit name and the arguments
    if circuit == "UCCSD":
        return qml.UCCSD(*args, **kwargs)
    elif circuit == "QAOAEmbedding":
        return qml.QAOAEmbedding(*args, **kwargs)
    else:
        raise ValueError(f"Invalid circuit name: {circuit}")

# Define the quantum gradients function
def quantum_gradients(gradient, *args, **kwargs):
    # Return the quantum gradient depending on the gradient name and the arguments
    if gradient == "param_shift":
        return qml.gradients.param_shift(*args, **kwargs)
    elif gradient == "natural":
        return qml.gradients.natural(*args, **kwargs)
    else:
        raise ValueError(f"Invalid gradient name: {gradient}")

# Define the quantum metrics function
def quantum_metrics(metric, *args, **kwargs):
    # Return the quantum metric depending on the metric name and the arguments
    if metric == "energy":
        return qml.metric_tensor(*args, **kwargs)
    elif metric == "phase":
        return qml.phase(*args, **kwargs)
    elif metric == "fidelity":
        return qml.fidelity(*args, **kwargs)
    elif metric == "precision":
        return qml.precision(*args, **kwargs)
    elif metric == "entanglement":
        return qml.entanglement(*args, **kwargs)
    else:
        raise ValueError(f"Invalid metric name: {metric}")

# Define the quantum noise function
def quantum_noise(noise, *args, **kwargs):
    # Apply the quantum noise depending on the noise name and the arguments
    if noise == "decoherence":
        qml.transforms.apply_decoherence(*args, **kwargs)
    elif noise == "depolarization":
        qml.transforms.apply_depolarization(*args, **kwargs)
    elif noise == "dephasing":
        qml.transforms.apply_dephasing(*args, **kwargs)
    elif noise == "bit_flip":
        qml.transforms.apply_bit_flip(*args, **kwargs)
    else:
        raise ValueError(f"Invalid noise name: {noise}")

# Define the h2_hamiltonian function
def h2_hamiltonian():
    # Return the Hamiltonian of the hydrogen molecule
    # The Hamiltonian is obtained from https://pennylane.ai/qml/demos/tutorial_vqe.html
    h2_hamiltonian_coeffs = np.array([-0.24274280513140462, 0.18093119978423156, -0.24274280513140462, 0.18093119978423156, -0.04475014401535161, -0.04475014401535161, -0.04475014401535161, -0.04475014401535161, 0.21362531027403572, 0.21362531027403572, 0.21362531027403572, 0.21362531027403572, 0.03277244519963702, 0.03277244519963702, 0.03277244519963702, 0.03277244519963702])
    h2_hamiltonian_ops = [qml.Identity(wires=[0]), qml.PauliZ(wires=[0]), qml.Identity(wires=[1]), qml.PauliZ(wires=[1]), qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]), qml.PauliY(wires=[0]) @ qml.PauliX(wires=[1]), qml.PauliX(wires=[0]) @ qml.PauliY(wires=[1]), qml.PauliX(wires=[0]) @ qml.PauliX(wires=[1]), qml.PauliY(wires=[0]) @ qml.PauliY(wires=[1]), qml.Identity(wires=[0]) @ qml.PauliZ(wires=[1]), qml.PauliZ(wires=[0]) @ qml.Identity(wires=[1]), qml.Identity(wires=[0]) @ qml.PauliX(wires=[1]), qml.PauliX(wires=[0]) @ qml.Identity(wires=[1]), qml.Identity(wires=[0]) @ qml.PauliY(wires=[1]), qml.PauliY(wires=[0]) @ qml.Identity(wires=[1]), qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1])]
    return h2_hamiltonian_coeffs, h2_hamiltonian_ops

# Define the phase_shift_operator function
def phase_shift_operator():
    # Return the unitary operator that generates the phase shift
    # The unitary operator is obtained from https://pennylane.ai/qml/demos/tutorial_quantum_metrology.html
    phase_shift_operator_matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    phase_shift_operator_wires = [1]
    return phase_shift_operator_matrix, phase_shift_operator_wires
