
#!/bin/sh
#############################################
#                                           #
# This job runs a simple OpenMP application #
#                                           #
#############################################
 
# Grid Engine options
#$ -N OpenMP-Accuracy
#$ -cwd
#$ -pe sharedmem 4
#$ -l h_rt=00:05:00
 
# Initialise the modules framework and load required modules
. /etc/profile.d/modules.sh
 module load openmpi

# set number of threads using $NSLOTS (provided by Grid Engine)
export OMP_NUM_THREADS=$NSLOTS
 
# Run the program
echo '========================================================================'
echo OMP_NUM_THREADS=$OMP_NUM_THREADS
./accuracy.cpp
echo '========================================================================'
