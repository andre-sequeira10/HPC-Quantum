#!/bin/bash
#SBATCH --job-name=PL30
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --array=30                          # array of tasks but only one task w/ 30 qubits 
#SBATCH -o grover_10its_%a_%j.out           # %a -> array index 
#SBATCH -e grover_10its_%a_%j.err


# Load environment
ml OpenMPI/5.0.3-GCC-13.3.0
ml Python/3.12.3-GCCcore-13.3.0
ml CMake/3.29.3-GCCcore-13.3.0
ml Boost/1.85.0-GCC-13.3.0
source /projects/I20240010/torchlane/torchlane_venv/bin/activate

# Set OpenMP environment variables
export OMP_NUM_THREADS=48
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static

export GOMP_CPU_AFFINITY="0-47"
export OMP_DISPLAY_ENV=TRUE


# ---- EXECUTE ----------------------------------------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_10its_flexiblas_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt \
    srun python grover_pennylane.py --n_qubits ${SLURM_ARRAY_TASK_ID}
