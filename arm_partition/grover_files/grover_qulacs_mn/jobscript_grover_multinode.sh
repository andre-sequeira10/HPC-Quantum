#!/bin/bash
#SBATCH --job-name=QG32
#SBATCH --account=i20240010a
#SBATCH --partition=normal-arm
#SBATCH --nodes=8
#SBATCH --ntasks=8

#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --array=33 # 
#SBATCH -o grover_48t_mn_%a_%j.out          # %a = array index (= n_qubits here)
#SBATCH -e grover_48t_mn_%a_%j.err


# Load environment
#source /share/env/module_select.sh
ml qulacs/0.6.11-foss-2024a-mem
#source /projects/I20240010/qulacs_python/venv/bin/activate

# Set OpenMP environment variables
export OMP_NUM_THREADS=48

# ---- EXECUTE ----------------------------------------------------------
# SLURM_ARRAY_TASK_ID takes the value 20 / 22 / … / 30 for each task
/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
              -o time_48t_mn_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.txt \
    srun python grover_example.py --n_qubits ${SLURM_ARRAY_TASK_ID}
