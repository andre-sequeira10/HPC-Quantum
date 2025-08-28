#!/bin/bash
#SBATCH --job-name=QG39
#SBATCH --account=i20240010a
#SBATCH --partition=large-arm
#SBATCH --nodes=512
#SBATCH --ntasks=512

#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --array=39 # 
#SBATCH -o grover_mn_%a.out          # %a = array index (= n_qubits here)
#SBATCH -e grover_mn_%a.err


# Load environment
source /share/env/module_select.sh
ml qulacs
#source /projects/I20240010/qulacs_python/venv/bin/activate

# Set OpenMP environment variables
export OMP_NUM_THREADS=48

# ---- EXECUTE ----------------------------------------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_mn_${SLURM_ARRAY_TASK_ID}.txt \
    srun python grover_example.py --n_qubits ${SLURM_ARRAY_TASK_ID}
