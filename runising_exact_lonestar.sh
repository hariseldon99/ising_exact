#!/bin/bash
#$ -V	#Inherit the submission environment
#$ -cwd	# Start job in submission directory
#$ -N ising_exact	# Job Name
#$ -j y	# Combine stderr and stdout
#$ -o $JOB_NAME.o$JOB_ID	# Name of the output file (eg. myMPI.oJobID)
#$ -pe 12way 12	# Requests 12 tasks/node, 12 cores total
#$ -q normal	# Queue name normal
#$ -l h_rt=12:00:00	# Run time (hh:mm:ss) - 10.5 hours

export MKL_NUM_THREADS=12
export OMP_NUM_THREADS=12

set -x	# Echo commands, use set echo with csh

#Transverse fields
export HX=0.2
export HY=0.0
export HZ=0.0

#Hopping terms
export JX=0.0
export JY=0.0
export JZ=1.0

#Lattice size
export LATSIZE=14

export RUNPROG="./ising_exact.py"

python $RUNPROG -l $LATSIZE -pbc -x $HX -y $HY -z $HZ -jx $JX -y $JY -z $JZ

echo "End script run."
