#!/bin/bash
#SBATCH --job-name=QAOA
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --array=31 
#SBATCH -o qaoa_%a_%j.out    
#SBATCH -e qaoa_%a_%j.err


# Load environment
ml qulacs
# If SciPy and networkx aren't in your Python env
module load networkx/3.1-foss-2024a
module load SciPy-bundle/2024.05-gfbf-2024a

# Set OpenMP environment variables
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# ---- EXECUTE ----------------------------------------------------------
srun python qaoa_qulacs.py --n_qubits ${SLURM_ARRAY_TASK_ID} --n_layers 2
