#!/bin/bash
#SBATCH --job-name=QAOA
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --array=20,22,24,26,28,30 
#SBATCH -o qaoa_%a.out          # %a -> array index 
#SBATCH -e qaoa_%a.err


# Load environment
ml qulacs
module load networkx/3.1-foss-2024a

# Set OpenMP environment variables
export OMP_NUM_THREADS=48

# ---- EXECUTE ----------------------------------------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_qaoa_${SLURM_ARRAY_TASK_ID}.txt \
    srun python qaoa_qulacs.py --n_qubits ${SLURM_ARRAY_TASK_ID}
