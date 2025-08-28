#Importing libraries
#import matplotlib.pyplot as plt
# NEW --- initialise the MPI runtime early -----------------
try:
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init_thread()
except ImportError:
    raise RuntimeError("mpi4py not available; load it with the module system")
# ----------------------------------------------------------

from qulacs import QuantumState, QuantumCircuit
# ... rest of your original code ...
import numpy as np
import time
import random
from qulacs import QuantumState , QuantumStateGpu
from qulacs.state import inner_product
from qulacs import QuantumCircuit
from qulacs.gate import to_matrix_gate
from qulacs import QuantumState
from qulacs.gate import Identity, X,Y,Z #Pauli operator
from qulacs.gate import H
from qulacs.gate import RX,RY,RZ #Rotation operations on Pauli operators
#from qulacsvis import circuit_drawer
from qulacs.circuit import QuantumCircuitOptimizer as QCO

from argparse import ArgumentParser
# ---- Args ----
parser = ArgumentParser()
parser.add_argument("--n_qubits", type=int, default=1, help="Number of qubits")
args = parser.parse_args()

# ---- Parameters ----
n_qubits = args.n_qubits


def show_distribution(state,nqubits):
    plt.bar([i for i in range(pow(2,nqubits))], abs(state.get_vector()))
    plt.show()

def Oracle(nqubits, target_state=None):
    U_w = QuantumCircuit(nqubits)

    for i in range(nqubits):
        if target_state[i] == 0:
            U_w.add_gate(X(i))  # Flip qubit to control state

    CnZ = to_matrix_gate(Z(nqubits-1))
    # apply gate only if i-th qubits are all 1's
    for i in range(nqubits-1):
        control_index = i
        control_with_value = 1
        CnZ.add_control_qubit(control_index, control_with_value)

    U_w.add_gate(CnZ)

    for i in range(nqubits):
        if target_state[i] == 0:

            U_w.add_gate(X(i))  # Flip qubit to control state

    return U_w

def Diffuser(nqubits):
    U_s = QuantumCircuit(nqubits)
    for i in range(nqubits):
        U_s.add_gate(H(i))

    ## 2|0><0| - I implementation
    ## First, phase (-1) is given to all states. The gate matrix is arrary([[-1,0],[0,-1]])
    U_s.add_gate(to_matrix_gate(RZ(nqubits-1, 2*np.pi)))
    U_s.add_gate( X(nqubits-1) )
    ## apply the Z-gate only if all i-th qubits are 0
    CnZ = to_matrix_gate(Z(nqubits-1))
    for i in range(nqubits-1):
        control_index = i
        control_with_value = 0
        CnZ.add_control_qubit(control_index, control_with_value)
    U_s.add_gate( CnZ )
    U_s.add_gate( X(nqubits-1) )

    for i in range(nqubits):
        U_s.add_gate(H(i))

    return U_s


#n_qubits = 10

#target_state = QuantumState(n_qubits, use_multi_cpu=True)
#target_state.set_computational_basis(2**n_qubits-1) ## 2**n_qubits-1 is a binary number 1.... .1

## Run Grover's algorithm
state = QuantumStateGpu(n_qubits)
state.set_zero_state()

def make_Hadamard(nqubits):
    Hadamard = QuantumCircuit(nqubits)
    for i in range(nqubits):
        Hadamard.add_gate(H(i))
    return Hadamard

Hadamard = make_Hadamard(n_qubits)
Hadamard.update_quantum_state(state)

marked_state = [1] * n_qubits
U_w = Oracle(n_qubits,target_state=marked_state)
U_s = Diffuser(n_qubits)

elements = 1
optimal_iterations = int(np.floor(np.pi/4 * np.sqrt(2**n_qubits/elements)))

#for i in range(optimal_iterations):
#just 10 grover iterations
for i in range(10):
    U_w.update_quantum_state(state)
    U_s.update_quantum_state(state)


circuit = QuantumCircuit(n_qubits)

circuit.update_quantum_state(state)
QCO().optimize(circuit,1,0)

#show_distribution(state,n_qubits)

probs = abs(state.get_vector())**2
print("Probabilities: ", probs)

#test Oracle with an empty circuit

#circuit = QuantumCircuit(n_qubits)
#circuit.merge_circuit(U_w)

#circuit_drawer(circuit)
