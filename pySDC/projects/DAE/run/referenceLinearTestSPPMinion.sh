#!/usr/bin/env bash

#SBATCH --ntasks 1 # Number of processes
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=wrcomp[2] #

#SBATCH --output slurm.%j.out
#SBATCH --error slurm.%j.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wimmer@uni-wuppertal.de

export OMPI MCA btl="tcp,self,sm"

date
mpirun /home/wimmer/Python/pySDC/bin/python3 refSolutionLinearTestSPPMinion.py
date