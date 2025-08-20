# Built-in modules
import math

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMTGate, ZGate
from qiskit.visualization import plot_distribution
#import matplotlib.pyplot as plt
from qiskit_aer import StatevectorSimulator, AerSimulator

from argparse import ArgumentParser
# ---- Args ----
parser = ArgumentParser()
parser.add_argument("--n_qubits", type=int, default=1, help="Number of qubits")
args = parser.parse_args()

# ---- Parameters ----
n_qubits = args.n_qubits

def grover_oracle(marked_states):
    """Build a Grover oracle for multiple marked states

    Here we assume all input marked states have the same number of bits

    Parameters:
        marked_states (str or list): Marked states of oracle

    Returns:
        QuantumCircuit: Quantum circuit representing Grover oracle
    """
    if not isinstance(marked_states, list):
        marked_states = [marked_states]
    # Compute the number of qubits in circuit
    num_qubits = len(marked_states[0])

    qc = QuantumCircuit(num_qubits)
    # Mark each target state in the input list
    for target in marked_states:
        # Flip target bit-string to match Qiskit bit-ordering
        rev_target = target[::-1]
        # Find the indices of all the '0' elements in bit-string
        zero_inds = [ind for ind in range(num_qubits) if rev_target.startswith("0", ind)]
        # Add a multi-controlled Z-gate with pre- and post-applied X-gates (open-controls)
        # where the target bit-string has a '0' entry
        if zero_inds != []:
            qc.x(zero_inds)
        qc.compose(MCMTGate(ZGate(), num_qubits - 1, 1), inplace=True)
        if zero_inds != []:
            qc.x(zero_inds)    
        
    return qc


marked_states = ["1"*n_qubits]

oracle = grover_oracle(marked_states)

grover_op = GroverOperator(oracle)

optimal_num_iterations = math.floor(
    math.pi / (4 * math.asin(math.sqrt(len(marked_states) / 2**grover_op.num_qubits)))
)
qc = QuantumCircuit(grover_op.num_qubits)
# Create even superposition of all basis states
qc.h(range(grover_op.num_qubits))
# Apply Grover operator the optimal number of times
#apply just for 10 steps
qc.compose(grover_op.power(100), inplace=True)
# Measure all qubits
qc.measure_all()

# Simulate the circuit with "statevector_simulator" backend

#qc = qc.decompose(reps=10)

#qc.save_statevector(label="final_state")

backend = StatevectorSimulator()
#backend = AerSimulator(method = 'statevector')

from qiskit import transpile
tr_circuits = transpile(qc, backend = backend)

backend.set_options(
    max_parallel_threads = 0,
    max_parallel_experiments = 0,
    max_parallel_shots = 1,
    statevector_parallel_threshold = 16
)

result = backend.run(tr_circuits).result().get_statevector()
#result = backend.run(qc).result()

print("amplitudes: ", result)
#print("statevector: ", result)