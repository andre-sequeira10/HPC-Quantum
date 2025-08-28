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
parser.add_argument("--n_layers", type=int, default=2, help="Number of QAOA layers")
args = parser.parse_args()

# ---- Parameters ----
n = args.n_qubits       # number of qubits / graph vertices
p = args.n_layers       # QAOA depth (number of layers)


#!/usr/bin/env python3
"""
QAOA for weighted Max-Cut in Qulacs (general depth p)
No command-line arguments; edit the variables below.
"""


# =========================
# User-configurable settings that can be made as arguments as well !!! 
# =========================
edge_prob = 0.3          # Erdos–Renyi edge probability
seed = 123               # RNG seed for graph & weights
maxiter = 300            # optimizer iterations
use_multi_cpu = False    # set True if running with MPI-enabled Qulacs (multi-rank)
weights_low, weights_high = 1, 10  # integer edge weights in [low, high]
# =========================

# ---- Build random weighted graph ----
rng = np.random.default_rng(seed)
G = nx.erdos_renyi_graph(n, p=edge_prob, seed=seed)
for (i, j) in G.edges():
    G.edges[i, j]["weight"] = int(rng.integers(weights_low, weights_high + 1))

# ---- Cost observable: C(Z) = 0.5 * sum_{(i,j) in E} w_ij Z_i Z_j ----
def cost_observable_from_graph(G, n):
    obs = Observable(n)
    for (i, j, data) in G.edges(data=True):
        w = float(data["weight"])
        obs.add_operator(PauliOperator(f"Z {i} Z {j}", 0.5 * w))
    return obs

cost_observable = cost_observable_from_graph(G, n)

# ---- QAOA layers ----
# U_C(gamma): for each edge (i,j) with weight w,
#   CNOT(i->j) ; RZ(j, -2*w*gamma) ; CNOT(i->j)
def add_U_C(circuit: QuantumCircuit, gamma: float, G) -> QuantumCircuit:
    for (i, j, data) in G.edges(data=True):
        w = float(data["weight"])
        circuit.add_CNOT_gate(i, j)
        circuit.add_gate(RZ(j, -2.0 * w * gamma))  # RZ(theta) = exp(-i*theta/2 Z)
        circuit.add_CNOT_gate(i, j)
    return circuit

# U_X(beta): RX(q, -2*beta) on every qubit
def add_U_X(circuit: QuantumCircuit, beta: float, n: int) -> QuantumCircuit:
    for q in range(n):
        circuit.add_gate(RX(q, -2.0 * beta))       # RX(theta) = exp(-i*theta/2 X)
    return circuit

# ---- Build p-layer QAOA circuit ----
def build_qaoa_circuit(n: int, G, betas, gammas) -> QuantumCircuit:
    assert len(betas) == len(gammas) == p
    circuit = QuantumCircuit(n)
    # Prepare |s> = |+>^{⊗n}
    for q in range(n):
        circuit.add_H_gate(q)
    # Apply layers: U_C(gamma_l) then U_X(beta_l)
    for l in range(p):
        add_U_C(circuit, gammas[l], G)
        add_U_X(circuit, betas[l], n)
    return circuit

# ---- Expectation of C(Z) for parameters x = [betas..., gammas...] ----
def qaoa_expectation(x: np.ndarray) -> float:
    betas = x[:p]
    gammas = x[p:]
    circ = build_qaoa_circuit(n, G, betas, gammas)
    state = QuantumState(n, use_multi_cpu=use_multi_cpu)
    state.set_zero_state()
    circ.update_quantum_state(state)
    return cost_observable.get_expectation_value(state)

# ---- Optimize parameters ----
x0 = np.concatenate([0.1 * np.ones(p), 0.1 * np.ones(p)])
res = minimize(qaoa_expectation, x0, method="powell", options={"maxiter": maxiter})

print("Optimized cost  ⟨C(Z)⟩:", res.fun)
betas_opt = res.x[:p]
gammas_opt = res.x[p:]
print("Optimized betas:", betas_opt)
print("Optimized gammas:", gammas_opt)

# ---- Rebuild circuit with optimal params and inspect ----
circ_opt = build_qaoa_circuit(n, G, betas_opt, gammas_opt)
state = QuantumState(n, use_multi_cpu=use_multi_cpu)
state.set_zero_state()
circ_opt.update_quantum_state(state)

probs = np.abs(state.get_vector()) ** 2
print("Probabilities:", probs)

# (Optional) report the best bitstring observed in amplitudes (for small n)
best_idx = int(np.argmax(probs))
best_bitstring = format(best_idx, "b").zfill(n)

def maxcut_value(bitstring: str) -> float:
    # Max-Cut value = sum_{(i,j) in E} w_ij * (1 - s_i s_j)/2, with s_i = (-1)^{z_i}
    z = np.array([int(b) for b in bitstring], dtype=int)
    s = 1 - 2*z  # 0->+1, 1->-1
    val = 0.0
    for (i, j, data) in G.edges(data=True):
        w = float(data["weight"])
        val += 0.5 * w * (1.0 - s[i] * s[j])
    return val

print(f"Most probable bitstring: {best_bitstring}")
print(f"Estimated Max-Cut value for it: {maxcut_value(best_bitstring)}")