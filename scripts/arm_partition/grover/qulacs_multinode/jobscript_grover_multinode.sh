#!/bin/bash
#SBATCH --job-name=Grover33
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=8
#SBATCH --ntasks=8

#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -o grover_33_%j.out         
#SBATCH -e grover_33_%j.err


# Load environment
ml qulacs
# Set OpenMP environment variables
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# ---- EXECUTE ----------------------------------------------------------
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_33_${SLURM_JOB_ID}.txt \
    srun python grover_example.py --n_qubits 33