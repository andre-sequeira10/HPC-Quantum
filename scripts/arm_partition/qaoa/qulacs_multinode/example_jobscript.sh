#!/bin/bash
#SBATCH --job-name=QAOA
#SBATCH --account=i20240010a
#SBATCH --partition=large-arm
#SBATCH --nodes=512
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH -o qaoa.out      
#SBATCH -e qaoa.err

# Load environment
ml qulacs
ml networkx/3.1-foss-2024a

# Set OpenMP environment variables
export OMP_NUM_THREADS=48

# Execute
srun python qaoa_qulacs.py 




