## PennyLane distributed computation on Deucalion's ARM — From source installation 

Pennylane supports distributed statevector simulation via MPI through its Lightning-Kokkos backend. However, this feature is not available as a module on Deucalion yet. To utilize this capability, users need to build PennyLane Lightning-Kokkos from source in their own Python virtual environment - follow the steps below to build from source in your own Python virtual environment.

- Target: CPU (A64FX) with OpenMP + MPI (no GPU in this guide, but installation is similar, see [lightning.kokkos install page](https://docs.pennylane.ai/projects/lightning/en/stable/lightning_kokkos/installation.html).
- You’ll get: `qml.device("lightning.kokkos", wires=n_qubits)` with MPI-enabled distributed simulation.

Official docs (for reference):
https://docs.pennylane.ai/projects/lightning/en/stable/lightning_kokkos/installation.html


#### 1. Load build toolchain modules (ARM node)

```bash
ml CMake/3.29.3-GCCcore-13.3.0 \
   OpenMPI/5.0.3-GCC-13.3.0 \
   Ninja/1.12.1-GCCcore-13.3.0 \
   Python/3.12.3-GCCcore-13.3.0
```

Tip: do the build on an ARM login node or an interactive salloc on the ARM partition.


#### 2. Install Kokkos (A64FX, OpenMP) — required for best performance

The Lightning docs list Kokkos as “optional,” but on Deucalion we recommend a dedicated A64FX Kokkos build so Lightning-Kokkos can link against it cleanly.

###### Get Kokkos source

```bash
# Replace x, y, and z by the correct version
wget https://github.com/kokkos/kokkos/archive/refs/tags/4.6.02.tar.gz
tar -xvf 4.6.02.tar.gz
cd kokkos-4.6.02

# Choose an install path (per-user)
export KOKKOS_INSTALL_PATH=$HOME/kokkos-install/4.6.02/A64FX
mkdir -p ${KOKKOS_INSTALL_PATH}

# Configure & build (A64FX + OpenMP)
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebugInfo -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_PATH} -DBUILD_SHARED_LIBS:BOOL=ON -DBUILD_TESTING:BOOL=OFF -DKokkos_ENABLE_SERIAL:BOOL=ON -DKokkos_ARCH_A64FX:BOOL=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_COMPLEX_ALIGN=OFF


cmake --build build && cmake --install build
export CMAKE_PREFIX_PATH=:"${KOKKOS_INSTALL_PATH}":$CMAKE_PREFIX_PATH
```

#### 3. Create & activate your Python virtual environment (ARM)

```bash
# Create a venv (pick any path; example uses home dir)
python -m venv $HOME/venvs/pl-kokkos-mpi
source $HOME/venvs/pl-kokkos-mpi/bin/activate

# Basic prerequisites
python -m pip install --upgrade pip wheel setuptools
pip install mpi4py
```

#### 4. Build and install PennyLane + Lightning-Kokkos with MPI 

```bash
# Get Lightning monorepo
git clone https://github.com/PennyLaneAI/pennylane-lightning.git
cd pennylane-lightning

# Requirements for building Python wheels
pip install -r requirements.txt

# Install PennyLane (latest master). Pin to a release if you prefer.
pip install git+https://github.com/PennyLaneAI/pennylane.git@master

# 4.1 Configure and 'install' Lightning-QUbit (no compilation needed)
PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
SKIP_COMPILATION=True pip install -e . --config-settings editable_mode=compat

# 4.2 Configure and build Lightning-Kokkos with OpenMP + MPI on A64FX
PL_BACKEND="lightning_kokkos" python scripts/configure_pyproject_toml.py


CMAKE_ARGS="-DKokkos_ENABLE_OPENMP=ON -DENABLE_MPI=ON -DKokkos_ARCH_A64FX=ON" \
  python -m pip install -e . --config-settings editable_mode=compat -vv

```

#### 5. Test your installation in a jobscript 

Once the installation is done, you can use the lightning.kokkos device in your pennylane script as: 

```python

dev = qml.device("lightning.kokkos", wires=NUM_QUBITS, mpi=True)
```

If you want to run on multiple nodes you need to set `mpi=True` otherwise set it to `False` for single-node runs - see [lightning.kokkos docs](https://docs.pennylane.ai/projects/lightning/en/stable/lightning_kokkos/device.html).

Below is a ready-to-run template. It assumes you completed the install in
`$HOME/venvs/pl-kokkos-mpi`. Adjust `--nodes`, `--ntasks-per-node`, and your script name as needed.

```bash 
#!/bin/bash
#SBATCH --job-name=PL-Kokkos-MPI
#SBATCH --account=<your ARM account>       # e.g., i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=16                          # total nodes
#SBATCH --ntasks-per-node=8                 # 8 MPI ranks per node
#SBATCH --cpus-per-task=6                   # 6 OpenMP threads per rank -> 8*6=48 cores/node
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH -o plkk_%a_%j.out
#SBATCH -e plkk_%a_%j.err

# ---------- Modules ----------
ml CMake/3.29.3-GCCcore-13.3.0 \
   OpenMPI/5.0.3-GCC-13.3.0 \
   Ninja/1.12.1-GCCcore-13.3.0

# ---------- Activate your environment ----------
source $HOME/venvs/pl-kokkos-mpi/bin/activate

# ---------- Kokkos runtime paths ----------
export KOKKOS_INSTALL_PATH=$HOME/kokkos-install/4.6.02/A64FX
export CMAKE_PREFIX_PATH="${KOKKOS_INSTALL_PATH}:${CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="${KOKKOS_INSTALL_PATH}/lib:${LD_LIBRARY_PATH}"

# ---------- OpenMP pinning per rank ----------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}   # 6 threads per rank
export OMP_PLACES=cores
export OMP_PROC_BIND=close                      # try 'spread' vs 'close' and benchmark

# ---------- Run ----------
srun python <your_python_script>.py --n_qubits 30
```

Follow `HPC-Quantum/scripts/` for several example scripts using PennyLane Lightning-Kokkos with MPI. Update your environment path and Kokkos paths accordingly.