#!/bin/bash
#SBATCH --mem=0
#SBATCH -N 1
#SBATCH -c 128
#SBATCH -p all
#SBATCH -J reassign
#SBATCH -o reassign-rmsd-clustering-enspara.out


# CHANGE THIS TO MATCH YOUR INSTALL PATH!
enspara_install_dir=~/software
# list of trajectories in a newline delimited text file
traj_list=traj-list.txt
# file containing mdtraj selection string for atoms to compute pairwise RMSD for
selstr_file=my_selection.txt
# the topology that matches te trajectory files
top=mytopology.pdb
distance=rmsd
# although reassign infers the number of clusters from the centers location,
# we put the chosen 'k' in the file paths, which is why we define it here.
k=200
# prevent loading any more than half the available ram at any one time.
# if getting out of memory errors, reduce this value.
mem_frac=0.5
outdir=rmsd-clustering/$k
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS
python $enspara_install_dir/enspara/enspara/apps/reassign.py \
 --trajectories $(tr '\n' ' ' < $traj_list) \
 --topology $top \
 --centers $outdir/centers.pickle \
 --cluster-distance $distance \
 -m $mem_frac \
 --distances $outdir/reassign-dists.h5 \
 --atoms "$(cat $selstr_file)" \
 --assignments $outdir/reassigns.h5

