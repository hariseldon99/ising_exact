#!/bin/bash
#$ -V	#Inherit the submission environment
#$ -cwd	# Start job in submission directory
#$ -N myMPI	# Job Name
#$ -j y	# Combine stderr and stdout
#$ -o $JOB_NAME.o$JOB_ID	# Name of the output file (eg. myMPI.oJobID)
#$ -pe 1way 12	# Requests 12 tasks/node, 12 cores total
#$ -q serial	# Queue name normal
#$ -l h_rt=10:30:00	# Run time (hh:mm:ss) - 10.5 hours

set -x	# Echo commands, use set echo with csh

#Transverse field range
export TFIELD_INIT=10.0
export TFIELD_FINAL=30.0
export TFIELD_NOS=10.0

#Lattice size
export LATSIZE=10

#Random number seed
export SEED=3

export RUNPROG="./isingrand_exact.py"

$RUNPROG -l $LATSIZE 

echo "End script run."
