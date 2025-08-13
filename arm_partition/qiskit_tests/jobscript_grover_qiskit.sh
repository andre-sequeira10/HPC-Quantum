#!/bin/bash
#SBATCH --job-name=grover_AF_qiskit
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --array= # six independent array tasks
#SBATCH -o grover_test_qiskit_sv%a.out          # %a = array index (= n_qubits here)
#SBATCH -e grover_test_qiskit_sv%a.err


# Load environment
source /projects/macc/malaca/qiskit/venv_qiskit_multinode/bin/activate
ml foss/2024a

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
              -o time_test_grover_qiskit_mp${SLURM_ARRAY_TASK_ID}.txt \
    python grover_qiskit.py --n_qubits ${SLURM_ARRAY_TASK_ID}
