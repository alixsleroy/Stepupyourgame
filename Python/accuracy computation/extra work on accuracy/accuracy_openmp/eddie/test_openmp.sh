
#!/bin/sh

#$ -N test_openmp
#$ -cwd
#$ -l h_rt=00:02:00
#$ -l h_vmem=1G
#$ -M s2133976@ed.ac.uk

# Initialise the modules framework and load required modules
. /etc/profile.d/modules.sh

module load openmpi
module load python/3.4.3
# set number of threads using $NSLOTS (provided by Grid Engine)
export OMP_NUM_THREADS=$NSLOTS
 
# Run the program
echo '========================================================================'
echo OMP_NUM_THREADS=$OMP_NUM_THREADS
./test_openmp.py
echo '========================================================================'

