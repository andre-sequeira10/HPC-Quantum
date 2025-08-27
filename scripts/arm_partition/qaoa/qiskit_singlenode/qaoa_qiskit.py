#!/usr/bin/env python3
"""
QAOA for Max-Cut on a weighted Erdős–Rényi graph (Qiskit ≥ 1.2)
"""

import argparse, random, networkx as nx, numpy as np
from qiskit.primitives import Sampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.algorithms import MinimumEigenOptimizer


def build_graph(n, p=0.3, seed=123):
    g = nx.erdos_renyi_graph(n, p, seed=seed)
    for u, v in g.edges():
        g.edges[u, v]['weight'] = random.randint(1, 10)
    return g


def graph_to_w(g):
    n = g.number_of_nodes()
    w = np.zeros((n, n))
    for u, v, d in g.edges(data=True):
        w[u, v] = w[v, u] = d['weight']
    return w


def cut_value(g, bits):
    return sum(d['weight'] for u, v, d in g.edges(data=True) if bits[u] != bits[v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_qubits', type=int, default=4)
    ap.add_argument('--p_layers', type=int, default=3)
    ap.add_argument('--maxiter',  type=int, default=500)
    args = ap.parse_args()

    g       = build_graph(args.n_qubits)
    w       = graph_to_w(g)
    qp      = Maxcut(w).to_quadratic_program()

    sampler   = Sampler()                      # state-vector backend by default
    optimizer = COBYLA(maxiter=args.maxiter)
    qaoa      = QAOA(sampler=sampler,
                     optimizer=optimizer,
                     reps=args.p_layers)

    result = MinimumEigenOptimizer(qaoa).solve(qp)

    print("\n======= Results =======")
    print("best bitstring :", result.x)
    print("cut weight     :", cut_value(g, result.x))
    print("=======================\n")


if __name__ == '__main__':
    main()