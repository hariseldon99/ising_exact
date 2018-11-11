#!/bin/bash
#########################################################################
## Name of my job
#PBS -N ising_exact

## Name of the job queue
#PBS -q S30

## Walltime
#PBS -l walltime=00:20:00

##Number of nodes and procs per node.
#PBS -l nodes=1:ppn=1

##Send me email when my job aborts, begins, or ends
#PBS -m ea
#PBS -M daneel@utexas.edu
#########################################################################
BETA=1.0
LATSIZE=11

HX=0.0
HY=0.0
HZ=0.0

JX=0.0
JY=0.0
JZ=1.0

AMPL=1.0
FREQ=1.0

## Name of python script to be executed
SCRIPT="./curie_weiss_1d_periodic.py"
#########################################################################

##Export all PBS environment variables
#PBS -V
##Output file. Combine stdout and stderr into one
#PBS -j oe
cd $PBS_O_WORKDIR
## Number of OpenMP threads to be used by the blas library. Keep this small
export OMP_NUM_THREADS=2
##Load these modules before running
module load openblas openmpi anaconda
BEGINTIME=$(date +"%s")
python -W ignore $SCRIPT \
	-l $LATSIZE -b $BETA \
	-x $HX -y $HY -z $HZ -jx $JX -jy $JY -jz $JZ \
	-a $AMPL -w $FREQ -n
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))
echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
