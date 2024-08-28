#!/bin/bash
#SBATCH -p all
#SBATCH -N 20
#SBATCH --ntasks-per-node 1
#SBATCH -J dihe-feat
#SBATCH -o logs/dihe-feat-%J.log

echo "executing task "
srun jug execute dihedral-featurize.py dihedral-cluster.json
echo 'done with task '
