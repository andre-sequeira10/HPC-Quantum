#!/bin/bash
#SBATCH --job-name=Q100its
#SBATCH --account=i20240010x
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --array=30,31,32,33 # six independent array tasks
#SBATCH -o grover_%a_%j.out          # %a = array index (= n_qubits here)
#SBATCH -e grover_%a_%j.err
#SBATCH --exclusive



# Load environment
#ml qulacs/0.6.11-foss-2024a
ml qulacs

# Set OpenMP environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
#export OMP_PROC_BIND=spread


# ---- EXECUTE ----------------------------------------------------------
srun python grover_example.py --n_qubits ${SLURM_ARRAY_TASK_ID}
srun python grover_example.py --n_qubits ${SLURM_ARRAY_TASK_ID}

/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt \
    srun python grover_example.py --n_qubits ${SLURM_ARRAY_TASK_ID}
