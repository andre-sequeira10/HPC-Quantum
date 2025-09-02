#!/bin/bash
#SBATCH --job-name=Grover
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=48                                     # 48 independent tasks instead of single task with 48 threads 
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00

#SBATCH -o grover_1node_48tasks_20_%j.out         
#SBATCH -e grover_1node_48tasks_20_%j.err

# Load environment
ml Qiskit/2.0.2-foss-2023a-opt

# Set OpenMP environment variables
#export OMP_NUM_THREADS=48
#export OMP_PLACES=cores
#export OMP_PROC_BIND=close
#export OMP_SCHEDULE=static
#export GOMP_CPU_AFFINITY="0-47"
#export OMP_DISPLAY_ENV=TRUE


# --------------------- EXECUTE -----------------------------
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_1node_48tasks_20_${SLURM_JOB_ID}.txt \
    srun python grover_qiskit_mn.py --n_qubits 20
