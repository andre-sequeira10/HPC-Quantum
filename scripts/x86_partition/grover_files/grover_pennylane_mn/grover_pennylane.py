from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import pennylane as qml
import numpy as np

'''
import os 
os.environ["OMP_NUM_THREADS"] = "48"
os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"
'''

from argparse import ArgumentParser
# ---- Args ----
parser = ArgumentParser()
parser.add_argument("--n_qubits", type=int, default=1, help="Number of qubits")
args = parser.parse_args()

# ---- Parameters ----
NUM_QUBITS = args.n_qubits

omega = np.array([np.ones(NUM_QUBITS)])

M = len(omega)
N = 2**NUM_QUBITS
wires = list(range(NUM_QUBITS))

dev = qml.device("lightning.kokkos", wires=NUM_QUBITS, mpi=True)#,shots = 10**6)
#dev = qml.device("lightning.gpu", wires=NUM_QUBITS)#, shots = 10**9)

@qml.qnode(dev)
def circuit():
    #iterations = int(np.round(np.sqrt(N / M) * np.pi / 4))
    iterations = 10
    # Initial state preparation
    for w in wires:
        qml.Hadamard(wires=w)

    # Grover's iterator
    for _ in range(iterations):
        for omg in omega:
            qml.FlipSign(omg, wires=wires)

        qml.templates.GroverOperator(wires)

    return qml.state()
    #return qml.sample(wires=wires)


samples = circuit()
print(samples)
#print(f"Probabilities for {NUM_QUBITS} qubits: {probs}")

#state_vector = comm.gather(probs, root=0)
#if rank == 0:
    #print(state_vector)