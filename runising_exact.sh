#!/bin/bash

#Number of parallel threads. Set to 1 if you want serial computing
export OMP_NUM_THREADS=10

#############################################################################################################
########################################Default Sun Grid Engine options.#####################################
#############################################################################################################
#$ -S /bin/bash
#$ -V                           # Inherit the submission environment
#$ -cwd                         # Start job in  submission directory
#$ -N eigsys                    # Job Name
#$ -j y                         # Combine stderr & stdout into stdout  
#$ -o $JOB_NAME.o$JOB_ID        # Name of the output file (eg. myMPI.oJobID)
#$ -q all.q                     # Queue name
#############################################################################################################
#############################################################################################################

#Transverse field range
export TFIELD_INIT=10.0
export TFIELD_FINAL=30.0
export TFIELD_NOS=10.0

#Lattice size
export LATSIZE=10

#Random number seed
export SEED=3

export RUNPROG="./isingrand_esys.py"

float_scale=8
export TFIELD_INC=$(echo "scale=$float_scale ;($TFIELD_FINAL-$TFIELD_INIT)/$TFIELD_NOS" | bc -q 2>/dev/null)

for i in $(seq $TFIELD_NOS)
do
  #Amplitude
  export TFIELD=$(echo "scale=$float_scale ;$TFIELD_INIT + $i*$TFIELD_INC" | bc -q 2>/dev/null)
  $RUNPROG -l $LATSIZE -s $SEED -f $TFIELD
done

echo "End script run."