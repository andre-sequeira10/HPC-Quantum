from qulacs import QuantumCircuit, QuantumState
import time 
from argparse import ArgumentParser
# ---- Args ----
parser = ArgumentParser()
parser.add_argument("--n_qubits", type=int, default=1, help="Number of qubits")
args = parser.parse_args()

# ---- Parameters ----
n = args.n_qubits

# apply hadamard gate to first qubit
circuit = QuantumCircuit(n)
circuit.add_H_gate(0)
for i in range(1, n):
    # apply CNOT gate to all other qubits
    circuit.add_CNOT_gate(0, i)


#Update state and save time
time_start = time.time()
state = QuantumState(n)
circuit.update_quantum_state(state)
time_end = time.time()

print(f"statevector: {state.get_vector()}")
print(f"Time taken to update state: {time_end - time_start:.6f} seconds")




