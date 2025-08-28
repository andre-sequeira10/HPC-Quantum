#!/bin/bash
#SBATCH --job-name=QGPU
#SBATCH --account=i20240010g
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --array=28,30 # six independent array tasks
#SBATCH -o qulacs_10its_%a_%j.out          # %a = array index (= n_qubits here)
#SBATCH -e qulacs_10its_%a_%j.err

# Load environment
ml qulacs/0.6.11-foss-2023a-CUDA-12.1.1

export CUDA_VISIBLE_DEVICES=0

# ---- EXECUTE ------------------------------   ----------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_qulacs_10its_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt \
    srun python grover_example.py --n_qubits ${SLURM_ARRAY_TASK_ID}
