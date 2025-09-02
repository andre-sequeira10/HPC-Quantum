#!/bin/bash
#SBATCH --job-name=QV30
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -o qv_30_%j.out    
#SBATCH -e qv_30_%j.err


# Load environment
ml qulacs

#export CUDA_VISIBLE_DEVICES=0
# Set OpenMP environment variables
export OMP_NUM_THREADS=32
export GOMP_CPU_AFFINITY="0-47"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static


# ---- EXECUTE ----------------------------------------------------------
srun python bench_circuit.py --nqubits 30 -t quantumvolume --fused 1  --opt 99


