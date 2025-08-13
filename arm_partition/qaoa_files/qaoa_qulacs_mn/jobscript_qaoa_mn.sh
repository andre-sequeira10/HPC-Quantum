#!/bin/bash
#SBATCH --job-name=QAOA30
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --array=30 
#SBATCH -o qaoa_mn_%a.out          # %a = array index (= n_qubits here)
#SBATCH -e qaoa_mn_%a.err


# Load environment
source /share/env/module_select.sh
ml qulacs
module load networkx/3.1-foss-2024a
#source /projects/I20240010/qulacs_python/venv/bin/activate

# Set OpenMP environment variables
export OMP_NUM_THREADS=48

# ---- EXECUTE ----------------------------------------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_qaoa_mn${SLURM_ARRAY_TASK_ID}.txt \
    srun python qaoa_qulacs.py --n_qubits ${SLURM_ARRAY_TASK_ID}
