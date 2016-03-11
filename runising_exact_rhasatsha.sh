#!/bin/bash
#########################################################################
## Name of my job
#PBS -N ising_exact_11
#PBS -l walltime=1:00:00
#########################################################################
##Export all PBS environment variables
#PBS -V
#########################################################################
#PBS -o stdout.txt
#PBS -e stderr.txt
#PBS -j oe

#########################################################################
##Number of nodes and procs per node.
##The ib at the end means infiniband. Use that or else MPI gets confused 
##with ethernet
#PBS -l select=1:ncpus=1

#########################################################################
##Send me email when my job aborts, begins, or ends
#PBS -m ea
#PBS -M daneel@sun.ac.za
#########################################################################

# make sure I'm the only one that can read my output
umask 0077

cd $PBS_O_WORKDIR

#########################################################################
##Make a list of allocated nodes(cores)
##Note that if multiple jobs run in same directory, use different names
##for example, add on jobid nmber.
#########################################################################
# How many cores total do we have?
NO_OF_CORES=$(cat $PBS_NODEFILE | wc -l)
#########################################################################
##Parameter ranges
export BETA=3.0
export HX=2.0
export HY=0.0
export HZ=2.0
export JX=-1.0
export JY=0.0
export JZ=0.0
export LATSIZE=11


#########################################################################
##Now run my prog
module load dot
BEGINTIME=$(date +"%s")
python ./ising_exact.py -v -n -x $HX -y $HY -z $HZ -jx $JX -jy $JY -jz $JZ -b $BETA -l $LATSIZE
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))

echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
