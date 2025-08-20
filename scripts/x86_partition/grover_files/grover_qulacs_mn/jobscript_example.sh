#!/bin/bash
#SBATCH --account=i20240010a
#SBATCH --partition=large-arm 
#SBATCH --nodes=512
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH -o grover_39.out         
#SBATCH -e grover_39.err

# Load environment
ml qulacs

# Set OpenMP environment variables
export OMP_NUM_THREADS=48

# Execute 
srun python grover_example.py



