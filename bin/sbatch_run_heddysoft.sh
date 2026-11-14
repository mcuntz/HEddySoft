#!/usr/bin/env bash

# explor / curta2 / biocomp
# https://slurm.schedmd.com/sbatch.html
# Name - 8 letters and digits
#SBATCH --ignore-pbs
#SBATCH --job-name=hes12_26
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# biocomp defq gpuq priogpuq
#SBATCH --partition=gpuq
#SBATCH --nodelist=gpu02
#
# -N nodes / -n tasks (mpiexec, srun, ...)
# -c cpus_per_task (OpenMP, make -j, ...)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#
# Check memory on *nix with GNU time: \time -v ./prog
# time (day-hh:mm:ss) / memory (optional, units K,M,G,T)
#SBATCH --time=23:00:00
# #SBATCH --mem=2G
#SBATCH --mem-per-cpu=1G
#
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND,STAGE_OUT,TIME_LIMIT,ARRAY_TASKS
#SBATCH --mail-user=matthias.cuntz@inrae.fr

set -ex

env

eval "$(${HOME}/amdforge3/bin/conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate pystd

# ncpu=${SLURM_CPUS_ON_NODE:-1}
ncpu=${SLURM_JOB_CPUS_PER_NODE:-1}

if [[ -f runrun_heddysoft.py ]] ; then rm runrun_heddysoft.py ; fi

sed "s/^\([ ]*\)multiprocess.*/\1multiprocess = ${ncpu}/" run_heddysoft.py > runrun_heddysoft.py

# \time -v python runrun_heddysoft.py
python runrun_heddysoft.py

rm runrun_heddysoft.py

echo ''
echo 'Finished.'
