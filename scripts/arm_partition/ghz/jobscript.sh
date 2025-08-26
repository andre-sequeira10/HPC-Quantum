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

#SBATCH --array=4-28

ml qulacs

# Set OpenMP environment variables
export OMP_NUM_THREADS=48
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

srun python ghz.py --n_qubits=${SLURM_ARRAY_TASK_ID}
