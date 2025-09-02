#!/bin/bash
#SBATCH --job-name=QV
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=2
#SBATCH --ntasks=2
##--ntasks-per-node=48
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --array=33 # independent array task -> 30 qubits 
#SBATCH -o qv_2nodes_%a_%j.out          # %a -> array index
#SBATCH -e qv_2nodes_%a_%j.err

# Load environment
ml Qiskit


# Set OpenMP environment variables
export OMP_NUM_THREADS=48
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static
export GOMP_CPU_AFFINITY="0-47"
export OMP_DISPLAY_ENV=TRUE

# ---- EXECUTE ------------------------------   ----------------------------

srun python quantum_volume_qiskit.py --n_qubits ${SLURM_ARRAY_TASK_ID}