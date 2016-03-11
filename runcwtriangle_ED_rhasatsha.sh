#!/bin/bash
#########################################################################
## Name of my job
#PBS -N cwtriangle_ED10
#PBS -l walltime=1:00:00
#########################################################################
##Export all PBS environment variables
#PBS -V
#########################################################################
##Output file. Combine stdout and stderr into one
#PBS -j oe 
#########################################################################
##Number of nodes and procs per node.
#PBS -l nodes=1:ppn=1
#########################################################################
##Send me email when my job aborts, begins, or ends
#PBS -m ea
#PBS -M daneel@sun.ac.za
#########################################################################
# How many cores total do we have?
NO_OF_CORES=$(cat $PBS_NODEFILE | wc -l)
#########################################################################
##Parameter ranges
export BETA=3.0
export HX=0.0
export HY=0.0
export HZ=0.0
export JX=0.0
export JY=0.0
export JZ=1.0
export NROWS=3

#########################################################################
##Now run my prog
module load dot
BEGINTIME=$(date +"%s")
python ./cw_triangle_lattice.py -v -n -x $HX -y $HY -z $HZ -jx $JX -jy $JY -jz $JZ -b $BETA -l $NROWS
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))

echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
