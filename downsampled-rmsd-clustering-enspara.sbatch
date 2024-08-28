#!/bin/bash
#SBATCH -N 6
#SBATCH --ntasks-per-node=8
#SBATCH -c 2
#SBATCH -p all
#SBATCH --mem-per-cpu=8GB
#SBATCH -J enspara-rmsd-clustering
#SBATCH -o enspara-rmsd-clustering.log

# CHANGE THIS TO MATCH YOUR INSTALL PATH!
enspara_install_dir=~/software
# stride rate for downsampling.
ds=100
# number of clusters to do.
k=200
# list of trajectories in a newline delimited text file
traj_list=traj-list.txt
# the topology that matches te trajectory files
top=mytopology.pdb

# number of medoid sweeps to do; note that each medoid pass 
# will take approx as long as the original kcenters.
iters=5  
# file containing mdtraj selection string for atoms to compute pairwise RMSD for
selstr_file=my_selection.txt
# note that you could replace the 'atoms' flag line below with a straight-up string;
# this would make sense if you were going for something simple, like 'name CA'. So like
# '--atoms "name CA"'
distance=rmsd
outdir=rmsd-clustering/$k
if [[ ! -d outdir ]]; then
    mkdir -p $outdir
fi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS
echo OMP_NUM_THREADS $OMP_NUM_THREADS
# your cluster might have a different system for getting its MPI env set up. 
# Do whatever dance you need to to make it work here.
module load mpi/openmpi-x86_64
# Run the clustering app with mpiexec
mpiexec -np $SLURM_NTASKS python $enspara_install_dir/enspara/enspara/apps/cluster.py \
  --trajectories $(tr '\n' ' ' < $traj_list) \
  --topology $top \
  --cluster-distance $distance\
  --algorithm khybrid \
  --cluster-iterations $iters \
  --subsample $ds \
  --cluster-number $k \
  --distances $outdir/distances.h5 \
  --center-features $outdir/centers.pickle \
  --atoms "$(cat $selstr_file)" \
  --assignments $outdir/assigns.h5
