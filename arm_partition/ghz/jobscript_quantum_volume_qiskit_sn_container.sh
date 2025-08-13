#!/bin/bash
#SBATCH --job-name  qulacs-python
#SBATCH --account   i20240010a
#SBATCH --partition large-arm
#SBATCH --time 00:59:00

#SBATCH --nodes           4
#SBATCH --tasks-per-node  48
##SBATCH --ntasks 4
#SBATCH --cpus-per-task   1
#SBATCH --mem=0
#SBATCH --exclusive

#SBATCH -o job.out
#SBATCH -e job.err


ml Qiskit
srun python ghz.py
##32 tasks and une cpu per task