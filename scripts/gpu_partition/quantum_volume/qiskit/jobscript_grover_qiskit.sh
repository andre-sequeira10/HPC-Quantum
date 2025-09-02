#!/bin/bash
#SBATCH --job-name=GPUQV34
#SBATCH --account=i20240010g
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=2
#SBATCH --gpus=8
#SBATCH --tasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --array=34,35,36
#SBATCH -o qv_gpu_%a_%j.out          # %a -> array index 
#SBATCH -e qv_gpu_%a_%j.err

# Load environment
ml Qiskit/2.0.2-foss-2023a-CUDA-12.1.1

# ---- EXECUTE ------------------------------   ----------------------------
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_gpu_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt \
            python quantum_volume_qiskit.py --n_qubits ${SLURM_ARRAY_TASK_ID}