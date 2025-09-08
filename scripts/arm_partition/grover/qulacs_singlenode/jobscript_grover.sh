#!/bin/bash
#SBATCH --job-name=Grover
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --array=20,22,24,26,28 # independent array task
#SBATCH -o grover_spread_%a_%j.out          # %a -> array index %j -> job ID
#SBATCH -e grover_spread_%a_%j.err


# Load environment
ml qulacs

# Set OpenMP environment variables
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# ---- EXECUTE ----------------------------------------------------------
srun python grover_example.py --n_qubits ${SLURM_ARRAY_TASK_ID}
srun python grover_example.py --n_qubits ${SLURM_ARRAY_TASK_ID}

/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_spread_${SLURM_ARRAY_TASK_ID}.txt \
    srun python grover_example.py --n_qubits ${SLURM_ARRAY_TASK_ID}
