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
from qulacs import QuantumState
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

"""
build a QuantumVolume Circuit with parameter
"""

import numpy as np

def simple_swap(p, q, array):
    tmp = array[p]
    array[p] = array[q]
    array[q] = tmp


def local_swap(p, q, done_ug, qubit_table):
    simple_swap(p, q, done_ug)
    simple_swap(p, q, qubit_table)
    # simple_swap(p, q, master_table)


def block_swap(p, q, bs, done_ug, qubit_table):
    for t in range(bs):
        simple_swap(p + t, q + t, qubit_table)
        # simple_swap(p + t, q + t, master_table)


def build_circuit(nqubits, global_nqubits=0, depth=10, verbose=False, random_gen=""):
    use_fusedswap = True if global_nqubits > 0 else False
    local_nqubits = nqubits - global_nqubits
    if random_gen == "":
        rng = np.random.default_rng()
    else:
        rng = random_gen

    circuit = QuantumCircuit(nqubits)#, use_multi_cpu=True)
    perm_0 = list(range(nqubits))
    seed = rng.integers(10000)

    for d in range(depth):
        qubit_table = list(range(nqubits))
        perm = rng.permutation(perm_0)
        pend_pair = []
        done_ug = [0] * nqubits

        # add random_unitary_gate for local_nqubits, first
        for w in range(nqubits // 2):
            physical_qubits = [int(perm[2 * w]), int(perm[2 * w + 1])]
            if (
                physical_qubits[0] < local_nqubits
                and physical_qubits[1] < local_nqubits
            ):
                if verbose:
                    print("#1: circuit.add_random_unitary_gate(", physical_qubits, ")")
                circuit.add_random_unitary_gate(physical_qubits, seed)
                seed += 1
                done_ug[physical_qubits[0]] = 1
                done_ug[physical_qubits[1]] = 1
            else:
                pend_pair.append(physical_qubits)

        # add SWAP gate for FusedSWAP
        work_qubit = local_nqubits - global_nqubits
        for s in range(global_nqubits):
            if done_ug[work_qubit + s] == 0:
                for t in range(work_qubit):
                    if done_ug[work_qubit - t - 1] == 1:
                        p = work_qubit + s
                        q = work_qubit - t - 1
                        local_swap(p, q, done_ug, qubit_table)
                        if verbose:
                            print("#2: circuit.add_SWAP_gate(", p, ", ", q, ")")
                        circuit.add_SWAP_gate(p, q)
                        break

        if verbose:
            print(
                "#3 block_swap(",
                work_qubit,
                ", ",
                local_nqubits,
                ", ",
                global_nqubits,
                ")",
            )
        if global_nqubits > 0:
            block_swap(work_qubit, local_nqubits, global_nqubits, done_ug, qubit_table)
            circuit.add_FusedSWAP_gate(work_qubit, local_nqubits, global_nqubits)
        if verbose:
            print("#: qubit_table=", qubit_table)

        # add random_unitary_gate for qubits that were originally outside.
        for pair in pend_pair:
            unitary_pair = [qubit_table.index(pair[0]), qubit_table.index(pair[1])]
            if verbose:
                print("#4: circuit.add_random_unitary_gate(", unitary_pair, ")")
            circuit.add_random_unitary_gate(unitary_pair, seed)
            seed += 1
            done_ug[unitary_pair[0]] = 1
            done_ug[unitary_pair[1]] = 1

    if verbose:
        print("circuit=", circuit)

    return circuit

start_time = time.time()
circuit = build_circuit(n_qubits , global_nqubits=2, depth=10, verbose=False)

state = QuantumState(n_qubits, use_multi_cpu=True)
#QCO().optimize(circuit,2) --> does not with memory, also explodes


circuit.update_quantum_state(state)


index = 3

#save time of get_zero_probability\
#zero_probability = state.get_zero_probability(index) --> does not work multi node 
#p = abs(state.get_vector()) # --> explodes memory 
samples = state.sampling(1) # --> explodes memory

#run with 29 qubits. 

end_time = time.time()
#calculate time taken
time_taken = end_time - start_time
#print("samples: ", samples)
print("Time taken to get zero probability: ", time_taken)
#test Oracle with an empty circuit

#circuit = QuantumCircuit(n_qubits)
#circuit.merge_circuit(U_w)

#circuit_drawer(circuit)
