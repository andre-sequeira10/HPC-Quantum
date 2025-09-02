#!/bin/bash
#SBATCH --job-name=QKGPU
#SBATCH --account=i20240010g
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH -o grover_gpu_1p_%j.out  
#SBATCH -e grover_gpu_1p_%j.err

# Load environment
ml Qiskit/2.0.2-foss-2023a-CUDA-12.1.1

# ----------------------------------------------- EXECUTE ----------------------------------------------------------        
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_gpu_1p_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt \
            python /projects/I20240010/qsim/gpu_partition/grover_files/single_node/grover_qiskit_sn.py --n_qubits 30