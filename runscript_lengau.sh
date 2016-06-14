#!/bin/bash
#########################################################################
## Name of my job
#PBS -N large_N
#PBS -l walltime=48:00:00
#########################################################################
##Export all PBS environment variables
#PBS -V
#########################################################################
##Output file. Combine stdout and stderr into one
#PBS -o /mnt/lustre/users/aroy/stdout.dat
#PBS -e /mnt/lustre/users/aroy/stderr.dat
#PBS -j oe 
#########################################################################
##Number of nodes and procs/mpiprocs per node.
#PBS -l select=4:ncpus=18:mpiprocs=9:nodetype=haswell_reg
#PBS -q normal
#PBS -P PHYS0853
#########################################################################
##Send me email when my job aborts, begins, or ends
#PBS -m ea
#PBS -M daneel@utexas.edu
#########################################################################
SCRIPT="./large_N.py"
# Make sure I'm the only one that can read my output
umask 0077
#Set BLAS threads to 1 per MPI process
export OMP_NUM_THREADS=1
# Load the module system
module load chpc/python/anaconda/2
#Locally installed PetSc compiled against Python-anaconda module above
PETSC_DIR=$HOME/.local/petsc-3.7.1
PETSC_ARCH=arch-linux2-c-opt
SLEPC_DIR=$HOME/.local/slepc-3.7.1
export PETSC_DIR
export PETSC_ARCH
export SLEPC_DIR

cd $PBS_O_WORKDIR

#########################################################################
# How many cores total do we have?
NPROCS=$(cat $PBS_NODEFILE | wc -l)
#########################################################################

#########################################################################
##Now, run the code
BEGINTIME=$(date +"%s")
mpirun -np $NPROCS -machinefile $PBS_NODEFILE  python -W ignore $SCRIPT
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))

echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
