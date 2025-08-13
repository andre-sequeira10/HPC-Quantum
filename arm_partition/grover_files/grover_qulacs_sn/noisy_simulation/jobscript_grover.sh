#!/bin/bash
#SBATCH --job-name=grover_qulacs_scaling
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --array=2 # six independent array tasks
#SBATCH -o grover_%a_%j.out          # %a = array index (= n_qubits here)
#SBATCH -e grover_%a_%j.err


# Load environment
#source /share/env/module_select.sh
ml qulacs/0.6.11-foss-2024a-mem
#source /projects/I20240010/qulacs_python/venv/bin/activate

# Set OpenMP environment variables
export OMP_NUM_THREADS=48

#python grover_example.py --n_qubits 10
#srun --mpi=pmi2 python grover_example.py --n_qubits 10
#srun python grover_example.py --n_qubits 10
#mpirun -np 1 python grover_example.py --n_qubits 10


# ---- EXECUTE ----------------------------------------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt \
    srun python grover_example.py --n_qubits ${SLURM_ARRAY_TASK_ID}
