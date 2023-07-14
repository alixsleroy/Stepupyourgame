
#!/bin/sh

#$ -N This_is_Alix
#$ -cwd
#$ -l h_rt=00:02:00
#$ -l h_vmem=1G
#$ -M s2133976@ed.ac.uk

. /etc/profile.d/modules.sh

module load openmpi
export OMP_NUM_THREADS=2
./hello_alix.exe

