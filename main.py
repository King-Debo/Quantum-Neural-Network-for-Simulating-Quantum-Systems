# main.py
# This is the entry point of the program, which imports the other files and executes the main function.

# Import the libraries
import numpy as np
import pennylane as qml
import torch
import qiskit
import matplotlib.pyplot as plt

# Import the other files
from qnn import QNN
from utils import *
from config import *

# Define the main function
def main():
    # Take the user input, such as the quantum system, the input data, and the output result
    quantum_system = input("Please enter the quantum system you want to simulate: ")
    input_data = input("Please enter the input data for the quantum system: ")
    output_result = input("Please enter the output result you want to obtain from the quantum system: ")

    # Call the corresponding functions from the qnn.py file to construct, train, and evaluate the QNN model
    qnn = QNN(quantum_system) # Create a QNN object for the quantum system
    qnn.encode(input_data) # Encode the input data into quantum states using quantum gates
    qnn.train(output_result) # Train the QNN model using gradient-based methods
    qnn.predict() # Predict the output result using the classical neural network

    # Print the output result and plot the relevant graphs using the matplotlib library
    print(f"The output result for the {quantum_system} is: {qnn.output_result}")
    plt.plot(qnn.loss_history) # Plot the loss history
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch for the {quantum_system}")
    plt.show()
    # Plot other graphs depending on the quantum system
    if quantum_system == "hydrogen molecule":
        plt.plot(qnn.energy_history) # Plot the energy history
        plt.xlabel("Epoch")
        plt.ylabel("Energy")
        plt.title(f"Energy vs Epoch for the {quantum_system}")
        plt.show()
    elif quantum_system == "quantum phase estimation":
        plt.plot(qnn.phase_history) # Plot the phase history
        plt.xlabel("Epoch")
        plt.ylabel("Phase")
        plt.title(f"Phase vs Epoch for the {quantum_system}")
        plt.show()

# Execute the main function
if __name__ == "__main__":
    main()
