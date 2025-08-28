#!/bin/bash
#SBATCH --job-name=QK_G35
#SBATCH --account=i20240010a
#SBATCH --partition=normal-x86
#SBATCH --nodes=64
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --array=35 # six independent array tasks
#SBATCH -o grover_qiskit_sv%a.out          # %a = array index (= n_qubits here)
#SBATCH -e grover_qiskit_sv%a.err


# Load environment
ml Qiskit

# Set OpenMP environment variables
export OMP_NUM_THREADS=48
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static
export GOMP_CPU_AFFINITY="0-47"
#export OMP_DISPLAY_ENV=TRUE


# ---- EXECUTE ----------------------------------------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_grover_qiskit_mn${SLURM_ARRAY_TASK_ID}.txt \
    srun python grover_qiskit_mn.py --n_qubits ${SLURM_ARRAY_TASK_ID}
