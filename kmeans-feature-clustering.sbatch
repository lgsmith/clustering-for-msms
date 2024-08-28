#!/bin/bash
#SBATCH -p all
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --array=[0-9]
#SBATCH --mem 250GB
#SBATCH -J p1-clus
#SBATCH -o logs/cluster-%J.log

echo 'start task '
srun jug execute kmeans_feature_clustering.py dihedral-clustering-posttica.json
echo 'done with task '
