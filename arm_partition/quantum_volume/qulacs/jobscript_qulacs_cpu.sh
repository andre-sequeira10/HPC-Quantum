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
#SBATCH --array=30
#SBATCH -o qv_new_bench_2nodes_mem_%a_%j.out          # %a = array index (= n_qubits here)
#SBATCH -e qv_new_bench_2nodes_mem_%a_%j.err


# Load environment
ml qulacs/0.6.11-foss-2024a-mem

#export CUDA_VISIBLE_DEVICES=0
# Set OpenMP environment variables
export OMP_NUM_THREADS=32
export GOMP_CPU_AFFINITY="0-47"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static


# ---- EXECUTE ------------------------------   ----------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
srun python bench_circuit.py --nqubits ${SLURM_ARRAY_TASK_ID} -t quantumvolume --fused 1  --opt 99
srun python bench_circuit.py --nqubits ${SLURM_ARRAY_TASK_ID} -t quantumvolume --fused 1  --opt 99
srun python bench_circuit.py --nqubits ${SLURM_ARRAY_TASK_ID} -t quantumvolume --fused 1  --opt 99


