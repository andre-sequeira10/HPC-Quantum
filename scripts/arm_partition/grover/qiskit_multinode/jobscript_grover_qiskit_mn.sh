#!/bin/bash
#SBATCH --job-name=QK_G29_48proc
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00

#SBATCH --array=20 # six independent array tasks
#SBATCH -o grover_1node_48tasks_1it_%a_%j.out          # %a = array index (= n_qubits here)
#SBATCH -e grover_1node_48tasks_1it_%a_%j.err

# Load environment
# source /projects/macc/malaca/qiskit/venv_qiskit_multinode/bin/activate
# ml foss/2024a
ml Qiskit/2.0.2-foss-2023a-opt

# Set OpenMP environment variables
#export OMP_NUM_THREADS=48
#export OMP_PLACES=cores
#export OMP_PROC_BIND=close
#export OMP_SCHEDULE=static
#export GOMP_CPU_AFFINITY="0-47"
#export OMP_DISPLAY_ENV=TRUE


# ---- EXECUTE ----------------------------------------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_1node_48tasks_1it_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt \
    srun python grover_qiskit_mn.py --n_qubits ${SLURM_ARRAY_TASK_ID}
