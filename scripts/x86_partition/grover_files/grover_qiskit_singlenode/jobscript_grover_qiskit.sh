#!/bin/bash
#SBATCH --job-name=Qiskit_Grover
#SBATCH --account=i20240010x
#SBATCH --partition=normal-x86
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --array=33 # six independent array tasks
#SBATCH -o grover_%a_%j.out          # %a = array index (= n_qubits here)
#SBATCH -e grover_%a_%j.err

 
# Load qiskit x86 environment
ml Qiskit/2.0.2-foss-2023a

# Set OpenMP environment variables

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export SLURM_CPU_BIND=threads  
export OMP_DISPLAY_ENV=TRUE


srun python grover_qiskit.py --n_qubits ${SLURM_ARRAY_TASK_ID}
srun python grover_qiskit.py --n_qubits ${SLURM_ARRAY_TASK_ID}

# ---- EXECUTE ----------------------------------------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt \
    srun python grover_qiskit.py --n_qubits ${SLURM_ARRAY_TASK_ID}
