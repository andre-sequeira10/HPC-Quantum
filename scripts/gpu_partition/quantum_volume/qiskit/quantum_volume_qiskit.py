# Built-in modules
import math

# Imports from Qiskit
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMTGate, ZGate
from qiskit.visualization import plot_distribution
#import matplotlib.pyplot as plt
from qiskit_aer import StatevectorSimulator, AerSimulator

import time
from argparse import ArgumentParser
# ---- Args ----
parser = ArgumentParser()
parser.add_argument("--n_qubits", type=int, default=1, help="Number of qubits")
args = parser.parse_args()

# ---- Parameters ----
n_qubits = args.n_qubits


#save time 
time_start = time.time()
qc = qiskit.circuit.library.QuantumVolume(n_qubits, depth=10, seed=None, classical_permutation=True,flatten=False)

qc.save_statevector()

#qc.save_statevector(label="final_state")

#backend = StatevectorSimulator()
backend = AerSimulator(method = 'statevector', device="GPU")

backend.set_options(    
    max_parallel_threads = 0,
    max_parallel_experiments = 0,
    max_parallel_shots = 1,
    statevector_parallel_threshold = 16,
    blocking_enable=True, 
    blocking_qubits=21
    )

from qiskit import transpile


qc_decomposed = qc.decompose()

tr_circuits = transpile(qc_decomposed)#, backend = backend)


#tr_circuits.decompose().draw(output='mpl')

#tr_circuits.measure_all()

result = backend.run(tr_circuits).result()

#result = execute(tr_circuits, backend, blocking_enable=True, blocking_qubits=26).result()
#result = backend.run(qc).result()

time_end = time.time()
print("Time taken: ", time_end - time_start, " seconds")
print("QV: ", result.get_statevector())
#print("statevector: ", result)