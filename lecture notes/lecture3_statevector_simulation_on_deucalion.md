# Lecture 3 - Quantum statevector simulation on Deucalion

- [Lecture 3 - Quantum statevector simulation on Deucalion](#lecture-3---quantum-statevector-simulation-on-deucalion)
  - [1. Slurm Basics ](#1-slurm-basics-)
    - [1.1 Create a batch script](#11-create-a-batch-script)
    - [1.2 Submit a batch job](#12-submit-a-batch-job)
    - [1.3 Examine the queue](#13-examine-the-queue)
    - [1.4 Cancel a job](#14-cancel-a-job)
    - [1.5 CPU Jobs](#15-cpu-jobs)
    - [1.6 GPU jobs](#16-gpu-jobs)
  - [2. Distributed statevector simulation on Deucalion's partitions](#2-distributed-statevector-simulation-on-deucalions-partitions)
    - [2.1 ARM partitions ](#21-arm-partitions-)
    - [2.2 x86 partitions ](#22-x86-partitions-)
    - [2.3 GPU partitions ](#23-gpu-partitions-)
  - [3. Examples](#3-examples)
    - [3.1 Hello quantum world](#31-hello-quantum-world)
    - [3.2 Grover's algorithm](#32-grovers-algorithm)
    - [3.3 Quantum Approximate Optimization Algorithm](#33-quantum-approximate-optimization-algorithm)
  - [4. References](#4-references)





## 1. Slurm Basics <a id="2-slurm-basics-"></a>

An HPC cluster consists of multiple compute nodes, each equipped with processors, memory, and GPUs. Users access these resources by submitting jobs, which specify the required resources and how to execute their applications. On Deucalion, resource allocation and job scheduling is managed by [Slurm](https://slurm.schedmd.com/). To learn more about Slurm in a hands-on way, you can explore the interactive [Slurm Learning tutorial](http://slurmlearning.deic.dk/).

The main commands for using Slurm are summarized in the table below.

<div align="center">

| Command   | Description                                               |
|-----------|-----------------------------------------------------------|
| `sbatch`  | Submit a batch script                                     |
| `squeue`  | View information about jobs in the scheduling queue       |
| `scancel` | Signal or cancel jobs, job arrays or job steps            |
| `sinfo`   | View information about nodes and partitions               |

<p><em>Table 1: Common Slurm commands</em></p>
</div>

### 1.1 Create a batch script

The most common type of jobs are batch jobs which are submitted to the
scheduler using a batch job script and the `sbatch` command.

A batch job script is a text file containing information about the job
to be run: the amount of computing resource and the tasks that must be executed.

A batch script is summarized by the following steps:

- the interpreter to use for the execution of the script: bash, python, ...
- directives that define the job options: resources, run time, ...
- setting up the environment: prepare input, environment variables, ...
- run the application(s)

Below is an example batch job script.

```bash
#!/bin/bash
#SBATCH --account=<your account>
#SBATCH --partition=<partition name>
#SBATCH --job-name=<your job name>
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

module load MyApp

MyApp -i input -o output
```

In the previous example, the first line `#!/bin/bash` indicates that this is a bash script.

Lines starting with `#SBATCH` are directives for the workload manager. 

The first directive


```bash
#SBATCH --account=<your account>
```

sets the billed account. To check your available accounts you can run the command

```bash
sacctmgr show Association where User=<username> format=Cluster,Account%30,User
```

where `<username>` is your Deucalion username.

```bash
[johnDoe@ln01 ~]$ sacctmgr show Association where User=JohnDoe format=Cluster,Account%30,User

   Cluster   Account                        User
   --------  ------------------------------ --------
   deucalion i2024000a                     JohnDoe
   deucalion i2024000g                     JohnDoe
   deucalion i2024000x                     JohnDoe
```

Above there is an example for the output of the account listing command. It shows three user accounts along with their associated clusters differing in a single character. These are responsible for indicating which Deucalion partitions the account has access - "a" (ARM) , "g" (GPU) and "x" (x86). 

Provided your account has access to a given partition you can set it in your batch script using the `#SBATCH --partition` directive, by setting the partition name as in Table 2. 

The directive 
```bash
#SBATCH --job-name=<your job name>
```

sets the name of the job so that it can be easily identified in the job queue
and other listings.


The next lines in the script specify the resources required for your job. The most important is the **maximum wall time** your job is allowed to run. If your job exceeds this limit, it will be terminated by the scheduler.

```bash
#SBATCH --time=02:00:00
```

The time is given in `hh:mm:ss` format (or `d-hh:mm:ss` for days). In this example, the job is limited to 2 hours.

The following directives define the compute resources:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
```

Here, you are requesting one task (process) on one node, with one CPU core allocated to that task. In most cases, a task corresponds to an MPI rank.

After specifying resources, you typically set up the environment. This may include copying input files to scratch storage or exporting environment variables.

```bash
module load MyApp
```

This command loads the required software module so that the `MyApp` application is available in your environment.

Finally, you launch your application:

```bash
MyApp -i input -o output
```

### 1.2 Submit a batch job

To submit the job script we just created we use the `sbatch` command. The
general syntax can be condensed as

```bash
$ sbatch [options] job_script [job_script_arguments ...]
```

Assuming the script is saved as a shell script named `myjob.sh`, you can submit it with

```bash
[johnDoe@ln01 ~]$ sbatch myjob.sh
Submitted batch job 123456
```

You can also pass arguments in the command line. For instance `sbatch --nodes=2` and `#SBATCH --nodes=2` in a batch script are equivalent. However, command line value takes precedence if the same
option is present both on the command line and as a directive in a script.

### 1.3 Examine the queue

Once you have submitted your batch script it won't necessarily run immediately.
It may wait in the queue of pending jobs for some time before its required
resources become available. In order to view your jobs in the queue, use the
`squeue` command.

```bash
$ squeue
  JOBID PARTITION     NAME     USER  ST       TIME  NODES NODELIST(REASON)
 123456   normal-arm exampleJ johnDoe  PD       0:00      1 (Priority)
```

The output shows the state of your job in the `ST` column. In our case, the job
is pending (`PD`). The last column indicates the reason why the job isn't
running: `Priority`. This indicates that your job is queued behind a higher
priority job. One other possible reason can be that your job is waiting for
resources to become available. In such a case, the value in the `REASON` column
will be `Resources`.

Let's look at the information that will be shown if your job is running. In order
to see only the jobs that belong to you use the `squeue` command with the
`--me` flag.


```bash
$ squeue -me
  JOBID PARTITION     NAME     USER  ST       TIME  NODES NODELIST(REASON)
 123456   normal-arm exampleJ johnDoe  R      35:00      1 node-0123
```

The `ST` column will now display a `R` value (for `RUNNING`). The `TIME` column
will represent the time your job has been running. The list of nodes on which
your job is executing is given in the last column of the output.

### 1.4 Cancel a job

You may need to cancel your job. This can be achieved using the `scancel`
command which takes the job ID of the job to cancel.

```bash
$ scancel <jobid>
```

The job ID can be obtained from the output of the `sbatch` command when
submitting your job or by using `squeue`. 


For more advanced options, please refer to the [Slurm documentation](https://slurm.schedmd.com/documentation.html).

### 1.5 CPU Jobs

CPU-only jobs run on the cluster’s CPUs without GPUs. For multithreaded programs, it’s important to request the right number of cores and bind your threads. The example below runs one task on a single node, gives that task 48 CPU cores, and sets OpenMP variables so threads are pinned and not oversubscribed. On Deucalion's there are several CPU partitions available, check Table 2 for details.

```bash
#!/bin/bash
#SBATCH --account=<account name>
#SBATCH --partition=<cpu partition name, e.g., normal-arm>
#SBATCH --job-name=cpu-example
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48

module load MyCPUApp  

# --- OpenMP/Threading best practices ---
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # match threads to allocated cores
export OMP_PLACES=cores                        # pin to physical cores
export OMP_PROC_BIND=spread                    # spread threads across cores

srun MyCPUApp -i input -o output
```

- `OMP_NUM_THREADS` ensures the program uses exactly the cores you requested—no more, no less.

- **`OMP_PROC_BIND=spread` vs `close`**
  - `spread` places threads **as far apart as possible** across the cores in your allocation. This improves **memory bandwidth** and reduces contention—often best for **bandwidth-bound** codes (e.g., state-vector simulation, large BLAS, big arrays).
  - `close` packs threads **near the master thread** (same core/socket first). This increases **cache locality**—often best for **cache-heavy** kernels with small working sets or lots of **producer→consumer** reuse.
  - Keep `OMP_PLACES=cores` so binding is at the **physical core** granularity (not SMT siblings). On NUMA systems, `spread` also tends to balance threads across sockets.

- **SLURM `--hint=nomultithread`**
  - Asks SLURM to allocate only **one logical CPU per physical core** and to **avoid SMT/Hyper-Threading siblings**. This prevents two OpenMP threads from landing on the same core, which can otherwise cause cache and pipeline contention.
  - Use `--hint=nomultithread` when you want **one thread per core** (the common case for HPC). If your code benefits from SMT, drop it or use `--hint=multithread`.

- **Other tips**
  - Set `OMP_DISPLAY_ENV=true` and (if supported) `OMP_DISPLAY_AFFINITY=true` to print the OpenMP runtime’s binding at startup.
  - Compare `spread` vs `close` and a few `--cpus-per-task` values; pick the fastest on your node architecture.
  - SLURM `--hint=nomultithread` asks SLURM to allocate only **one logical CPU per physical core** and to **avoid SMT/Hyper-Threading siblings**. This prevents two OpenMP threads from landing on the same core, which can otherwise cause cache and pipeline contention. Use `--hint=nomultithread` when you want **one thread per core** (the common case for HPC). 
  
### 1.6 GPU jobs

<h3>Single-GPU</h3>

All GPU nodes in Deucalion are non-exclusive, meaning that you can allocate any number of GPUs you ask for. You can allocate them with `--gpus`:

```bash
#!/bin/bash
#SBATCH --account=<account name>
#SBATCH --partition=<partition name>
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00

module load MyModule

srun ./my_gpu_application
```

For every GPU you ask, you must ask for 32 CPUs for instance through `--cpus-per-task`. 

<h3>Multi-GPU in Single Node</h3>
Each node on every GPU partition in Deucalion (see Table 2) is equipped with up to 4 GPUs, all of which can be used simultaneously within a single job. The following job script requests 4 GPUs, fully utilizing a single node:

```bash
#!/bin/bash

#SBATCH --account <account name>
#SBATCH --partition=<partition name>
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=results/%j.out
#SBATCH --error=results/%j.err

module load  MyModule

srun  ./my_gpu_application
```

`--tasks-per-node=4` ensures that exactly four independent tasks (processes) are launched on each node, matching the number of GPUs available. It is a good practice to check the output of `nvidia-smi` to ensure that you are running at least one process per GPU. Keep in mind that if the number of nodes is not manually specified, the script may still run but SLURM can allocate your GPUs across different nodes which may lead to performance degradation. `--output` and `--error` are optional parameters that specify the file where the standard output and error of the job will be written for easy debugging.

---

<h3>Multi-node</h3>

For multi-node jobs, you should ensure that **every GPU in every node** is being used — otherwise you are underutilizing the resources. It is crucial to ensure one task per GPU (via `--tasks-per-node=4`) and to match CPU allocations accordingly (e.g., `--cpus-per-task=32`) to fully saturate each GPU. This configuration is ideal for large-scale distributed training or simulation workloads using MPI, PyTorch, etc. The example below illustrates how to configure a SLURM job across 4 nodes with 16 GPUs total, aligned with Table 2 specifications for `normal-a100-40`:

```bash
#!/bin/bash

#SBATCH --account=<account name>
#SBATCH --time=00:30:00
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=4
#SBATCH --gpus=16
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=results/%j.out

module load MyModule

srun ./my_gpu_application
```
In this setup, `--tasks-per-node=4` ensures that exactly four independent tasks (processes) are launched on each node, matching the number of GPUs available. The total number of tasks across all nodes will therefore be `--ntasks=16` implicitly (4 tasks/node × 4 nodes), which is what `srun` will use to coordinate parallel execution. If needed, you can also explicitly specify `--ntasks=16`, but it’s not required when using `--tasks-per-node` in conjunction with `--nodes`. Ensuring the correct mapping between GPUs and tasks is key for performance, especially in distributed workloads.


## 2. Distributed statevector simulation on Deucalion's partitions

We size by **memory first**. With double precision, the state size is $S(n)=16\cdot 2^n\ \text{bytes}$. We use conservative “safe” per-node budgets (leave room for work buffers/OS):

- **ARM nodes (A64FX, 32 GiB HBM2):** **16 GiB** per node for the state  
- **x86 nodes (2× EPYC 7742, 256 GiB DRAM):** **128 GiB** per node for the state  
- **GPU nodes:** 4× A100 per node with 2× AMD EPYC 7742 (128 CPU cores total)

  - A100-40: **~32 GiB** per GPU for the state (of 40 GB) → 4 GPUs/node ⇒ **128 GiB/node**  
  - A100-80: **~64 GiB** per GPU for the state (of 80 GB) → 4 GPUs/node ⇒ **256 GiB/node**

Assume **1 MPI rank per CPU node**, or **1 rank per GPU** on GPU nodes. Choose the smallest **power-of-two** ranks so that the per-rank slice fits the safe budget.

<div align="center">
  <table>
    <thead>
      <tr>
        <th>Partition name</th>
        <th>Max nodes</th>
        <th>Cores per node</th>
        <th>RAM per node</th>
        <th>Total RAM for statevector (16 bytes/amp)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>normal-arm</td>
        <td>128</td>
        <td>48</td>
        <td>32 GB</td>
        <td>2 TiB</td>
      </tr>
      <tr>
        <td>large-arm</td>
        <td>512</td>
        <td>48</td>
        <td>32 GB</td>
        <td>16 TiB</td>
      </tr>
      <tr style="border-top: 2px solid #fcfbfbff;">
        <td>normal-x86</td>
        <td>64</td>
        <td>128</td>
        <td>256 GB</td>
        <td>16 TiB</td>
      </tr>
      <tr>
        <td>large-x86</td>
        <td>128</td>
        <td>128</td>
        <td>256 GB</td>
        <td>32 TiB</td>
      </tr>
      <tr style="border-top: 2px solid #fcfbfbff;">
        <td>normal-a100-40</td>
        <td>4</td>
        <td>128</td>
        <td>512 GB</td>
        <td>2 TiB</td>
      </tr>
      <tr>
        <td>normal-a100-80</td>
        <td>4</td>
        <td>128</td>
        <td>512 GB</td>
        <td>2 TiB</td>
      </tr>
    </tbody>
  </table>
  <p><em>Table 2: Standard publicly available partitions on Deucalion, with total RAM available for statevector simulation (assuming 16 bytes per amplitude)</em></p>
</div>

---

### 2.1 ARM partitions <a id="11-arm-partitions"></a>

- `normal-arm`: **max 128 nodes**  
- `large-arm` : **max 512 nodes**

Per-node safe = **16 GiB**

<div align="center">

| Qubits | State size | Nodes | Fits in `normal-arm` (≤128) | Fits in `large-arm` (≤512) |
|---:|---:|---:|:--:|:--:|
| 30 | 16 GiB  | 1    | ✅ | ✅ |
| 31 | 32 GiB  | 2    | ✅ | ✅ |
| 32 | 64 GiB  | 4    | ✅ | ✅ |
| 33 | 128 GiB | 8    | ✅ | ✅ |
| 34 | 256 GiB | 16   | ✅ | ✅ |
| 35 | 512 GiB | 32   | ✅ | ✅ |
| 36 | 1 TiB   | 64   | ✅ | ✅ |
| 37 | 2 TiB   | 128  | ✅ | ✅ |
| 38 | 4 TiB   | 256  | ❌ | ✅ |
| 39 | 8 TiB   | 512  | ❌ | ✅ |
| 40 | 16 TiB  | 1024 | ❌ | ❌ |

<p><em>Table 3: Maximum qubits and required nodes for ARM partitions</em></p>
</div>

*Max inside partition:* `normal-arm` → **37 qubits**; `large-arm` → **39 qubits**.

---

### 2.2 x86 partitions <a id="12-x86-partitions"></a>

- `normal-x86`: **max 64 nodes**  
- `large-x86` : **max 128 nodes**

Per-node safe = **128 GiB**

<div align="center">

| Qubits | State size | Nodes | Fits in `normal-x86` (≤64) | Fits in `large-x86` (≤128) |
|---:|---:|---:|:--:|:--:|
| 33 | 128 GiB | 1   | ✅ | ✅ |
| 34 | 256 GiB | 2   | ✅ | ✅ |
| 35 | 512 GiB | 4   | ✅ | ✅ |
| 36 | 1 TiB   | 8   | ✅ | ✅ |
| 37 | 2 TiB   | 16  | ✅ | ✅ |
| 38 | 4 TiB   | 32  | ✅ | ✅ |
| 39 | 8 TiB   | 64  | ✅ | ✅ |
| 40 | 16 TiB  | 128 | ❌ | ✅ |
| 41 | 32 TiB  | 256 | ❌ | ❌ |
| 42 | 64 TiB  | 512 | ❌ | ❌ |

<p><em>Table 4: Maximum qubits and required nodes for x86 partitions</em></p>
</div>

*Max inside partition:* `normal-x86` → **39 qubits**; `large-x86` → **40 qubits**.

---

### 2.3 GPU partitions <a id="13-gpu-partitions"></a>
- CPU: 2× AMD EPYC 7742 (128 CPU cores total)
- `normal-a100-40`: **max 4 nodes**, each **4× A100 40 GB** → **16 GPUs** total in this partition  
    Safe VRAM for state: **~32 GiB/GPU** ⇒ **512 GiB** total across the partition
- `normal-a100-80`: **max 4 nodes**, each **4× A100 80 GB** → **16 GPUs** total  
    Safe VRAM for state: **~64 GiB/GPU** ⇒ **1 TiB** total across the partition



<div align="center">

| Qubits $n$ | State size | GPUs (min) | Nodes (min) | Fits in partition (≤16 GPUs) |
|---:|---:|---:|---:|:--:|
| 31 | 32 GiB  | 1   | 1 | ✅ |
| 32 | 64 GiB  | 2   | 1 | ✅ |
| 33 | 128 GiB | 4   | 1 | ✅ |
| 34 | 256 GiB | 8   | 2 | ✅ |
| 35 | 512 GiB | 16  | 4 | ✅ |
| 36 | 1 TiB   | 32  | 8 | ❌ |

<p><em>Table 5: Maximum qubits and required GPUs/nodes for `normal-a100-40` GPU partition</em></p>
</div>

*Max inside partition:* **35 qubits** (DP) on `normal-a100-40`.

- `normal-a100-80` (≤4 nodes = 16 GPUs total)

<div align="center">

| Qubits $n$ | State size | GPUs (min) | Nodes (min) | Fits in partition (≤16 GPUs) |
|---:|---:|---:|---:|:--:|
| 32 | 64 GiB  | 1   | 1 | ✅ |
| 33 | 128 GiB | 2   | 1 | ✅ |
| 34 | 256 GiB | 4   | 1 | ✅ |
| 35 | 512 GiB | 8   | 2 | ✅ |
| 36 | 1 TiB   | 16  | 4 | ✅ |
| 37 | 2 TiB   | 32  | 8 | ❌ |

<p><em>Table 6: Maximum qubits and required GPUs/nodes for `normal-a100-80` GPU partition</em></p>
</div>

## 3. Examples

### 3.1 Hello quantum world
As a first example on Deucalion, we’ll build an n-qubit GHZ state using Qulacs (a fast state-vector simulator). On Deucalion, several quantum frameworks are preinstalled as modules; you load them like any other software. In this example we assume Python and Qulacs, which you can load inside the jobscript with `module load qulacs`. ( You can check quantum based modules on deucalion using the command `module avail`). Qulacs is a high-performance, versatile quantum circuit simulator developed for quantum computing research. Written primarily in C++ with optimized backend implementations (including CPU parallelization via OpenMP, SIMD optimizations) and user-friendly Python bindings, Qulacs aims to provide researchers with one of the fastest simulation environments available.

On the Deucalion supercomputer, Qulacs has been installed and tested, leveraging the system's large-scale distributed computing capabilities to simulate quantum systems beyond the reach of typical workstations and up to 40 qubits at the ARM partition.

Qulacs is a high-performance CPU-based statevector simulator optimized with OpenMP and vectorized kernels, making it well-suited for Deucalion’s ARM nodes (see Table 2 for core and memory details). For example, simulating a 28-qubit GHZ state requires a statevector with $2^{28}$ complex amplitudes (about 4 GiB for `complex128`), which fits easily within a single ARM node’s memory. This allows you to run single-node jobs efficiently. On Deucalion, quantum simulation frameworks like Qulacs are available as modules—simply load Qulacs with `module load qulacs` in your job script. You can try this example yourself using the script in `scripts/ghz`.


```python
from qulacs import QuantumCircuit, QuantumState
import time 
from argparse import ArgumentParser
# ---- Args ----
parser = ArgumentParser()
parser.add_argument("--n_qubits", type=int, default=2, help="Number of qubits")
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

#get probability of all zero state
prob_zero = state.get_probability([0] * n)
#get probability of all one state
prob_one = state.get_probability([1] * n)

print(f"Time taken to update state {n} qubits: {time_end - time_start:.6f} seconds")
print(f"Probability of all zero state: {prob_zero:.6f}")
print(f"Probability of all one state: {prob_one:.6f}")
```
What the Python script does:
1.	Parses `--n_qubits` (default 2).
2.	Builds a circuit: H on qubit 0, then CNOT(0→i) for i=1..n-1 → prepares GHZ_n.
3.	Creates `QuantumState(n)` (starts in $|0\rangle^n$), applies the circuit, and saves the time it takes to get the amplitudes.
4.	Prints the probability of the all zero and all ones states and the elapsed time.

⚠️ For large n printing the full vector is huge; it’s common to instead print only selected amplitudes or summary stats.

Then we create the following jobscript:
```bash
#!/bin/bash
#SBATCH --job-name=qulacs-ghz
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --time=00:30:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --exclusive

#SBATCH -o ghz_%j.out
#SBATCH -e ghz_%j.err

#SBATCH --array=4-30

ml qulacs

# Set OpenMP environment variables
export OMP_NUM_THREADS=48              # or: ${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread            # try 'close' vs 'spread' and benchmark

srun python ghz.py --n_qubits=${SLURM_ARRAY_TASK_ID}
```

What the jobscript does (step-by-step):

1.	Chooses account, normal-arm partition, wall-time, and a single node.
2.	Runs 1 task `--ntasks=1` with 48 CPU cores `--cpus-per-task=48`, no SMT `--hint=nomultithread`, and exclusive node use.
3.	Uses a job array 4-30 → launches one run per $n \in [4 \dots 30]$ (value passed via `SLURM_ARRAY_TASK_ID`).
4.	Loads Qulacs via the module system (ml qulacs).
5.	Sets OpenMP controls so Qulacs threads match the cores reserved and are pinned to cores (PLACES=cores, PROC_BIND=spread).
6.	Launches the program with srun, passing `--n_qubits=${SLURM_ARRAY_TASK_ID}`.

How to run and fetch results

1.	Save the python script as `ghz.py` and jobscript as `jobscript_ghz.sh` in the same directory.
2.	Submit the array job with `sbatch jobscript_ghz.sh`.
3.	Monitor progress with `squeue --me`.
4.	Outputs appear as ghz_<jobid>.out and ghz_<jobid>.err. For timings: `grep "Time taken to update state" ghz_*.out`
5. To inspect a specific task `less ghz_<jobid>.out`

### 3.2 Grover's algorithm

### 3.3 Quantum Approximate Optimization Algorithm


## 4. References 
- [Y.Suzuki et.al Qulacs: a fast and versatile quantum circuit simulator for research purpose](https://arxiv.org/pdf/2011.13524)