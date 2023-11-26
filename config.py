# config.py
# This file contains the configuration parameters, which are used by the QNN class and the utility functions.

# Define the quantum backend
quantum_backend = "qasm_simulator" # The name of the quantum backend or the quantum simulator
# Define the quantum device
quantum_device = "ibmq_athens" # The name of the quantum device or the quantum hardware
# Define the quantum shots
quantum_shots = 1000 # The number of times the quantum circuit is executed
# Define the quantum wires
quantum_wires = 2 # The number of qubits in the quantum circuit
# Define the quantum layers
quantum_layers = 1 # The number of layers in the quantum ansatz
# Define the quantum parameters
quantum_parameters = 2 # The number of parameters in the quantum ansatz
# Define the classical layers
classical_layers = 4 # The number of nodes in the hidden layer of the classical neural network
# Define the classical parameters
classical_parameters = 2 # The number of nodes in the output layer of the classical neural network
# Define the learning rate
learning_rate = 0.01 # The learning rate of the classical optimizer
# Define the batch size
batch_size = 32 # The batch size of the input data
# Define the epochs
epochs = 100 # The number of epochs for the training process
# Define the seed
seed = 42 # The seed for the random number generator
