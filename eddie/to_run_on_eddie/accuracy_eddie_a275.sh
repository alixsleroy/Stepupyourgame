#!/bin/sh

#$ -N a275
#$ -cwd
#$ -l h_rt=00:05:00
#$ -l h_vmem=1G
#$ -M s2133976@ed.ac.uk

# Initialise the modules framework and load required modules
. /etc/profile.d/modules.sh
 module load openmpi

# set number of threads using $NSLOTS (provided by Grid Engine)
export OMP_NUM_THREADS=$NSLOTS
 
# Run the program
echo '========================================================================'
echo OMP_NUM_THREADS=$OMP_NUM_THREADS
./accuracy_eddie_a27.exe
echo '========================================================================'
