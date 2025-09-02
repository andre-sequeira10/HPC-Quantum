#!/bin/bash
#SBATCH --job-name=QKQAOA
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --array=20,22,24,26,28,30 # six independent array tasks
#SBATCH -o qaoa_qiskit_%a.out          # %a -> array index
#SBATCH -e qaoa_qiskit_%a.err


# Load environment
ml Qiskit/2.0.2-foss-2023a-opt


# Set OpenMP environment variables
export OMP_NUM_THREADS=48
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static
export GOMP_CPU_AFFINITY="0-47"
export OMP_DISPLAY_ENV=TRUE


# ---- EXECUTE ----------------------------------------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_test_qaoa_qiskit_${SLURM_ARRAY_TASK_ID}.txt \
    python qaoa_qiskit.py --n_qubits ${SLURM_ARRAY_TASK_ID}
