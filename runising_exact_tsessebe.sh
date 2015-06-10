#!/bin/bash
#########################################################################
## Name of my job
#PBS -N ising_exact
#PBS -l walltime=96:00:00
#########################################################################
##Export all PBS environment variables
#PBS -V
#########################################################################
##Output file. Combine stdout and stderr into one
#PBS -o stdout.dat
#PBS -e stderr.dat
#PBS -j oe 
#########################################################################
##Number of nodes and procs per node.
##See docs at http://wiki.chpc.ac.za/howto:pbs-pro_job_submission_examples 
#U need 8 procs for the multithreaded BLAS
#PBS -l select=1:ncpus=8:mpiprocs=8

#########################################################################
##Send me email when my job aborts, begins, or ends
#PBS -m ea
#PBS -M daneel@sun.ac.za
#########################################################################

#Transverse fields
export HX=0.2
export HY=0.0
export HZ=0.0

#Hopping terms
export JX=0.0
export JY=0.0
export JZ=1.0

#Lattice size
export LATSIZE=11

export RUNPROG="./ising_exact.py"

# Make sure I'm the only one that can read my output
umask 0077
# Load the module system
source /etc/profile.d/modules.sh
#Load relevant modules. Load them with THESE TWO LINES, NOT FROM ONE LINE
module load dot intel
module load gcc/4.9.1 Anaconda/2.1.0

cd $PBS_O_WORKDIR

#########################################################################
# How many cores total do we have?
NO_OF_CORES=$(cat $PBS_NODEFILE | wc -l)
#########################################################################
#########################################################################
##Now, run the code

BEGINTIME=$(date +"%s")
python $RUNPROG -l $LATSIZE -pbc -x $HX -y $HY -z $HZ -jx $JX -y $JY -z $JZ
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))

echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
