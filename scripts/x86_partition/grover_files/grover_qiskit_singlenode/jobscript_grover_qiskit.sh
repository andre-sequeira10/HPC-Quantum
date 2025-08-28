#!/bin/bash
#SBATCH --job-name=G32_48t
#SBATCH --account=i20240010x
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --array=20,22,24,26,28,30,31,32,33 # six independent array tasks
#SBATCH -o grover_sn_100its_48t_%a_%j.out          # %a = array index (= n_qubits here)
#SBATCH -e grover_sn_100its_48t_%a_%j.err

 
# Load qiskit x86 environment
ml Qiskit/2.0.2-foss-2023a

# Set OpenMP environment variables

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export SLURM_CPU_BIND=threads  
export OMP_DISPLAY_ENV=TRUE


# ---- EXECUTE ----------------------------------------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_100its_48t_${SLURM_ARRAY_TASK_ID}.txt \
    srun --cpu-bind=threads python grover_qiskit.py --n_qubits ${SLURM_ARRAY_TASK_ID}
