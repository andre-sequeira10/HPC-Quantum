#!/bin/bash
#SBATCH --job-name=QKGPU
#SBATCH --account=i20240010g
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --array=28
#SBATCH --exclusive 
#SBATCH -o grover_state_%a_%j.out
#SBATCH -e grover_state_%a_%j.err


# Load modules
ml NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1 \
   cuQuantum/24.08.0.5-CUDA-12.1.1 \
   OpenMPI/4.1.5-GCC-12.3.0 \
   CMake/3.26.3-GCCcore-12.3.0 \
   Ninja/1.11.1-GCCcore-12.3.0 

# Activate the penynlane environment
source /projects/I20240010/qsim/venv_kokkos_gpu_mpi/kokkos-4.6.02/venv_kokkos_gpu_omp_mpi/bin/activate

#export path to kokkos to run pennylane-kokkos device enabling multi gpu distributed simulation 
export KOKKOS_INSTALL_PATH=$HOME/kokkos-install/4.5.0/AMPERE80
export CMAKE_PREFIX_PATH=:"${KOKKOS_INSTALL_PATH}":$CMAKE_PREFIX_PATH

# Run
srun python grover_pennylane.py --n_qubits ${SLURM_ARRAY_TASK_ID} 
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_state_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt \
                srun python grover_pennylane.py --n_qubits ${SLURM_ARRAY_TASK_ID}


