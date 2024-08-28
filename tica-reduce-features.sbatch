#!/bin/bash
#SBATCH -p all
#SBATCH -n 128
#SBATCH -N 1
#SBATCH --mem 0
#SBATCH -J p1-reduce
#SBATCH -o logs/dihe-reduce-%J.log

echo "executing task "
python tica-reduce-features.py dihedral-cluster-postfeat.json
echo 'done with task '
