#!/bin/bash
#########################################################################
## Name of my job
#PBS -N large_N
#PBS -l walltime=01:00:00
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
#PBS -l select=1:ncpus=4:mpiprocs=1:nodetype=haswell_reg
#PBS -q smp
#PBS -P PHYS0853
#########################################################################
##Send me email when my job aborts, begins, or ends
#PBS -m ea
#PBS -M daneel@utexas.edu
#########################################################################
NUM_THREADS=4
BETA=0.0
LATSIZE=7

HX=0.0
HY=0.0
HZ=0.0

JX=0.0
JY=0.0
JZ=1.0

GUD=0.0
GDU=0.0
GEL=0.0

SCRIPT="./curie_weiss_exact_lindblad.py"

#########################################################################
export OMP_NUM_THREADS=$NUM_THREADS
export OPENBLAS_NUM_THREADS=$NUM_THREADS
export MKL_NUM_THREADS=$NUM_THREADS
# Make sure I'm the only one that can read my output
umask 0077
#cd $PBS_O_WORKDIR
##Now, run the code
BEGINTIME=$(date +"%s")
python -W ignore $SCRIPT \
	-l $LATSIZE -b $BETA \
	-x $HX -y $HY -z $HZ -jx $JX -jy $JY -jz $JZ \
	-gud $GUD -gdu $GDU -gel $GEL -n
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))
echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
