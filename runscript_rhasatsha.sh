#!/bin/bash
#########################################################################
## Name of my job
#PBS -N ising_exact_13
#PBS -l walltime=30:00:00
#########################################################################
##Export all PBS environment variables
#PBS -V
#########################################################################
##Output file. Combine stdout and stderr into one
#PBS -j oe ${PBS_JOBNAME}.o${PBS_JOBID}
#########################################################################
##Number of nodes and procs per node.
##The ib at the end means infiniband. Use that or else MPI gets confused 
##with ethernet
#PBS -l nodes=1:ppn=1
#########################################################################
##Send me email when my job aborts, begins, or ends
#PBS -m ea
#PBS -M daneel@sun.ac.za
#########################################################################

# make sure I'm the only one that can read my output
umask 0077
TMP=/scratch2/$PBS_JOBID
mkdir -p $TMP

if [ ! -d "$TMP" ]; then
	echo "Cannot create temporary directory. Disk probably full."
	exit 1
fi

cd $PBS_O_WORKDIR

# copy the input files to $TMP
echo "Copying from ${PBS_O_WORKDIR}/ to ${TMP}/"
/usr/bin/rsync -vax "${PBS_O_WORKDIR}"/ ${TMP}/

cd $TMP

#########################################################################
##Make a list of allocated nodes(cores)
##Note that if multiple jobs run in same directory, use different names
##for example, add on jobid nmber.
#########################################################################
# How many cores total do we have?
NO_OF_CORES=$(cat $PBS_NODEFILE | wc -l)
#########################################################################
##Parameter ranges
export BETA=1.0
export HX=0.2
export HY=0.0
export HZ=0.0
export JX=0.0
export JY=0.0
export JZ=1.0
export LATSIZE=13


#########################################################################
##Now run my prog
module load python/2.7.6 dot
BEGINTIME=$(date +"%s")
python ./ising_exact.py -pbc -x $HX -y $HY -z $HZ -jx $JX -jy $JY -jz $JZ -b $BETA -l $LATSIZE
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))

# job done, copy everything back 
echo "Copying from ${TMP}/ to ${PBS_O_WORKDIR}/"
/usr/bin/rsync -vax ${TMP}/ "${PBS_O_WORKDIR}/"
# delete my temporary files
[ $? -eq 0 ] && /bin/rm -rf ${TMP}

echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
