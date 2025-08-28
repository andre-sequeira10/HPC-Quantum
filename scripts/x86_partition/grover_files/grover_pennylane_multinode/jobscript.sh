#!/bin/bash
#SBATCH --job-name=QKGPU
#SBATCH --account=i20240010x
#SBATCH --partition=normal-x86
#SBATCH --nodes=2                        # 2 nodes (each with 4 GPUs)
##SBATCH --tasks-per-node=4             # 4 MPI ranks per node
#SBATCH --ntasks=2          # 4 MPI ranks per node
#SBATCH --cpus-per-task=128             # 32 cpus per MPI rank since node has 128 cores
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --array=32
#SBATCH --exclusive 
#SBATCH -o grover_state_%a_%j.out
#SBATCH -e grover_state_%a_%j.err

# Load modules
ml NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1 \
   cuQuantum/24.08.0.5-CUDA-12.1.1 \
   OpenMPI/4.1.5-GCC-12.3.0 \
   CMake/3.26.3-GCCcore-12.3.0 \
   Ninja/1.11.1-GCCcore-12.3.0 

# Activate your environment
source /projects/I20240010/qsim/venv_kokkos_gpu_mpi/kokkos-4.6.02/venv_kokkos_gpu_omp_mpi/bin/activate

# Set backend and threads
#export UCX_TLS=shm,rc_x,cuda_copy,cuda_ipc,^cma   # <-- key line
#export UCX_TLS=^cma
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#export KOKKOS_INSTALL_PATH=$HOME/kokkos-install/4.5.0/AMPERE80
#export CMAKE_PREFIX_PATH=:"${KOKKOS_INSTALL_PATH}":$CMAKE_PREFIX_PATH

# Run
srun python grover_pennylane.py --n_qubits ${SLURM_ARRAY_TASK_ID} 
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_state_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt \
                srun python grover_pennylane.py --n_qubits ${SLURM_ARRAY_TASK_ID}


