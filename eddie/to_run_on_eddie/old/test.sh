
#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N This_is_Alix
#$ -cwd
#$ -l h_rt=00:02:00
#$ -l h_vmem=1G
#$ -M s2133976@ed.ac.uk

. /etc/profile.d/modules.sh

# Load Python
module load intel

# Run the program
./test.exe
