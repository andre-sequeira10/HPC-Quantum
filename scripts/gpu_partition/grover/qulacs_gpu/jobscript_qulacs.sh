#!/bin/bash
#SBATCH --job-name=QGPU
#SBATCH --account=i20240010g
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH -o qulacs_%j.out
#SBATCH -e qulacs_%j.err

# Load environment
ml qulacs/0.6.11-foss-2023a-CUDA-12.1.1


# ---- EXECUTE ------------------------------   ----------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_qulacs_28_${SLURM_JOB_ID}.txt \
    srun python grover_example.py --n_qubits 28
