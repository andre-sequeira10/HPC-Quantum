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

from qulacs import QuantumState, QuantumCircuit, Observable, PauliOperator
from qulacs.gate import H, CNOT, RX, RZ
from scipy.optimize import minimize
import numpy as np

import networkx as nx
import random



from argparse import ArgumentParser
# ---- Args ----
parser = ArgumentParser()
parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits")
args = parser.parse_args()

# ---- Parameters ----
n = args.n_qubits

# ---- 1. make the graph -------------------------------------------------
n_qubits = n             
G = nx.erdos_renyi_graph(n_qubits, p=0.3, seed=123)

# ---- 2. give every edge a positive integer weight ----------------------
for (u, v) in G.edges():
    G.edges[u, v]["weight"] = random.randint(1, 10)

# ---- 3. visual sanity check -------------------------------------------
#pos = nx.spring_layout(G, seed=1)  # layout only for plotting
#labels = nx.get_edge_attributes(G, "weight")
#nx.draw(G, pos, node_color="lightblue", with_labels=True)
#nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#plt.show()

## define C(Z) as qulacs.Observable
cost_observable = Observable(n)
for i in range(n):
    cost_observable.add_operator( PauliOperator("Z {:} Z {:}".format(i, (i+1)%n), 0.5) )

# a function to add U_C(gamma) to a circuit
def add_U_C(circuit, gamma):
    # for edges in the graph G, add CNOT and RZ gates
    for (i,j) in G.edges():
        circuit.add_CNOT_gate(i, j)
        circuit.add_gate(RZ(j, -2*G.edges[i,j]["weight"]*gamma)) ## With qulacs, RZ(theta)=e^{i*theta/2*Z}
        circuit.add_CNOT_gate(i, j)
    return circuit

# A function to add U_X(beta) to a circuit
def add_U_X(circuit, beta):
    for i in range(n):
        circuit.add_gate(RX(i, -2*beta))
    return circuit

# a function to |beta, gamma> in the case of p=2  and return  <beta, gamma| C(Z) |beta, gamma>
# x = [beta0, beta1, gamma0, gamma1]
def QAOA_output_twolayer(x):
    beta0, beta1, beta2, gamma0, gamma1, gamma2 = x

    circuit = QuantumCircuit(n)
    ## to create superposition, apply Hadamard gate
    for i in range(n):
        circuit.add_H_gate(i)
    ##apply  U_C, U_X
    circuit =  add_U_C(circuit, gamma0)
    circuit =  add_U_X(circuit, beta0)
    circuit =  add_U_C(circuit, gamma1)
    circuit =  add_U_X(circuit, beta1)
    circuit =  add_U_C(circuit, gamma2)
    circuit =  add_U_X(circuit, beta2)

    ## prepare |beta, gamma>
    state = QuantumState(n,use_multi_cpu=True) 
    state.set_zero_state()
    circuit.update_quantum_state(state)
    return cost_observable.get_expectation_value(state)

## initial value
x0 = np.array( [0.1] * 6)

## minimize with scipy.minimize
result = minimize(QAOA_output_twolayer, x0, options={'maxiter':500}, method='powell')
#print(result.fun) # value after optimization
#print(result.x) # [beta0, beta1, gamma0, gamma1] after optimization

## Check the probability distribution when measuring the state after optimization
beta0, beta1, beta2 , gamma0, gamma1, gamma2 = result.x
 
circuit = QuantumCircuit(n)
 ## to create superposition, apply Hadamard gate
for i in range(n):
    circuit.add_H_gate(i)
## apply U_C, U_X
circuit =  add_U_C(circuit, gamma0)
circuit =  add_U_X(circuit, beta0)
circuit =  add_U_C(circuit, gamma1)
circuit =  add_U_X(circuit, beta1)
circuit =  add_U_C(circuit, gamma2)
circuit =  add_U_X(circuit, beta2)

## prepare |beta, gamma>
state = QuantumState(n,use_multi_cpu=True)
state.set_zero_state()
circuit.update_quantum_state(state)

##  Square of the absolute value observation probability
probs = np.abs(state.get_vector())**2
print(probs)

## a bit string which can be acquired from z axis projective measurement
z_basis = [format(i,"b").zfill(n) for i in range(probs.size)]

'''
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.xlabel("states")
plt.ylabel("probability(%)")
plt.bar(z_basis, probs*100)
plt.show()
'''