#!/bin/bash
#SBATCH --job-name=QKGPU
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=16                        # 2 nodes (each with 4 GPUs)
#SBATCH --ntasks-per-node=8          # 4 MPI ranks per node
##SBATCH --tasks-per-node=2          # 4 MPI ranks per node
#SBATCH --cpus-per-task=6            # 32 cpus per MPI rank since node has 128 cores
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --array=33
#SBATCH --exclusive 
#SBATCH -o grover_state_%a_%j.out
#SBATCH -e grover_state_%a_%j.err

# Load modules
ml CMake/3.29.3-GCCcore-13.3.0 \
   OpenMPI/5.0.3-GCC-13.3.0 \
   Ninja/1.12.1-GCCcore-13.3.0


# Activate your OWN environment
source /projects/I20240010/qsim/kokkos-4.6.02/venv_kokkos_mpi/bin/activate

# Set backend and threads
export OMP_NUM_THREADS=48
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static

export KOKKOS_INSTALL_PATH=$HOME/kokkos-install/4.6.02/A64FX
export CMAKE_PREFIX_PATH=:"${KOKKOS_INSTALL_PATH}":$CMAKE_PREFIX_PATH
# Run
srun python grover_pennylane.py --n_qubits ${SLURM_ARRAY_TASK_ID} 
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_state_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt \
                srun python grover_pennylane.py --n_qubits ${SLURM_ARRAY_TASK_ID}


