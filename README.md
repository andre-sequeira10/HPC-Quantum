# HPC-Quantum: Quantum Statevector Simulation on Deucalion

A hands-on, performance-oriented tutorial series that takes you from
**quantum computing foundations** → **HPC statevector theory** → **real jobs on the Deucalion supercomputer**.

- **Language & stack:** Python + [Qulacs] (primary), with examples also for Qiskit Aer and PennyLane.
- **Target cluster:** [Deucalion] (ARM, x86 and A100 GPU partitions).
- **Focus:** Scaling **statevector simulation**; practical memory sizing; node/thread/GPU mapping; Slurm job scripts that “just work”.

---

## 📚 What’s inside

1. **Part 1 — Quantum computing & classical simulation**  
   Quantum computing through linear algebra and Dirac notation; single- & two-qubit gates; what’s classically easy/hard and statevector simulation.  
   👉 Read: **[part1_quantum_computing_and_classical_simulation.md](part1_quantum_computing_and_classical_simulation.md)**  

2. **Part 2 — Statevector simulation on HPC**  
   Parallelization and distributed statevectors; Deucalion sizing tables for ARM/x86/GPU.  
   👉 Read: **[part2_statevector_simulation_on_hpc.md](part2_statevector_simulation_on_hpc.md)**  

3. **Part 3 — Statevector simulation on Deucalion**  
   Slurm basics; Deucalion's partitions overview; safe memory budgets; **working job scripts** for **GHZ**, **Grover**, and **QAOA** on ARM nodes.  
   👉 Read: **[part3_statevector_simulation_on_deucalion.md](part3_statevector_simulation_on_deucalion.md)**  

> 💡 Tip: Skim Part 3 first if you just want to run jobs now; circle back to Parts 1–2 for theory and why the scripts are shaped this way.

---

## 🔎 Why this repo

- For quantum computing enthusiasts that lack the resources for large-scale simulations on their local machines and lack the expertise to set up HPC environments.
- **Qulacs** is one of the fastest quantum statevector simulators in practice and it is highly optimized to run on Deucalion's ARM architecture. However, there is little documentation available for users unfamiliar with HPC concepts and large scale Qulacs usage.
- You get **copy-pasteable** Slurm scripts + **sane defaults** for `OMP_*`, MPI ranks, and GPU allocations.
- Clear, cluster-specific **memory tables** up to 40+ qubits to ease the process of requesting the right resources and simplify quantum simulation on Deucalion.

---

## 🗂️ Repository layout