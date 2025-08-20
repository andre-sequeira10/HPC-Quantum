#!/bin/bash
#SBATCH --job-name=QK_G
#SBATCH --account=i20240010g
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --array=30,32 # six independent array tasks
#SBATCH -o grover_%a_%j.out          # %a = array index (= n_qubits here)
#SBATCH -e grover_%a_%j.err


# Load environment
ml Qiskit/2.0.2-foss-2023a-CUDA-12.1.1
#source /projects/I20240010/qsim/gpu_partition/grover_files/single_node/qiskit_aer_gpu/bin/activate

#ml  OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0
# Set OpenMP environment variables
#export OMP_NUM_THREADS=128
#export OMP_PLACES=cores
#export OMP_PROC_BIND=close
#export OMP_SCHEDULE=static
#export GOMP_CPU_AFFINITY="0-47"
#export OMP_DISPLAY_ENV=TRUE

export CUDA_VISIBLE_DEVICES=0,1,2,3

# ---- EXECUTE ------------------------------   ----------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt \
    srun python grover_qiskit_sn.py --n_qubits ${SLURM_ARRAY_TASK_ID}
kokkos-4.6.02.tar.gz