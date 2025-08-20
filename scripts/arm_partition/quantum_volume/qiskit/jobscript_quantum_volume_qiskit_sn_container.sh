#!/bin/bash
#SBATCH --job-name=QKQVCPU
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=2
#SBATCH --ntasks=2
##--ntasks-per-node=48
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --array=33 # six independent array tasks
#SBATCH -o qv_2nodes_%a_%j.out          # %a = array index (= n_qubits here)
#SBATCH -e qv_2nodes_%a_%j.err

# Load environment
ml Qiskit

#source /projects/I20240010/qsim/gpu_partition/grover_files/single_node/qiskit_aer_gpu/bin/activate

#ml  OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0
# Set OpenMP environment variables
export OMP_NUM_THREADS=48
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static
export GOMP_CPU_AFFINITY="0-47"
export OMP_DISPLAY_ENV=TRUE

# ---- EXECUTE ------------------------------   ----------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
#singularity run --nv \
#        -B /projects:/projects \
#        /projects/I20240010/qsim/gpu_partition/grover_files/single_node/cuquantum-appliance_25.03-x86_64.sif \

srun python quantum_volume_qiskit_sn.py --n_qubits ${SLURM_ARRAY_TASK_ID}