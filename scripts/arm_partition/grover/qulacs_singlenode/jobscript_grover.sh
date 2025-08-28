#!/bin/bash
#SBATCH --job-name=grover_qulacs_scaling
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --array=20 # six independent array tasks
#SBATCH -o grover_sn_300its_%a.out          # %a = array index (= n_qubits here)
#SBATCH -e grover_sn_300its_%a.err


# Load environment
ml qulacs/0.6.11-foss-2024a-mem

# Set OpenMP environment variables
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# ---- EXECUTE ----------------------------------------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_sn_qulacs_300its_${SLURM_ARRAY_TASK_ID}.txt \
    srun python grover_example.py --n_qubits ${SLURM_ARRAY_TASK_ID}
