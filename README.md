# Clustering for MSM construction
## What is here
In this repository I have included some scripts for clustering (and also featurization) that I and my labmates in the Bowman group have passed around. These nearly always get modified in the process of building production quality MSMs, so do not be afraid to color outside of the lines with these; they're just a place to start. I've tried to go through and anonymize the system specific variable names that creep their way into such analysis scripts. Sorry if I missed a spot, and feel free to let me know about it.

This directory contains scripts to do the following things:
 - Cluster by RMSD with kcenters, then use kmedoids updates to refine center positions.
 - Cluster a collection of features saved to `.npy` files using kmeans (from `deeptime`).
 - Featurize trajectories using dihedrals computed with `PyEmma`, a particular package for doing this kind of analysis.
 - Reduce the dimensionality of a collection of features using tICA.

There are other ways to do all of these things, but sometimes having a way to do them that is compliant with a commodity cluster and works well enough on larger datasets is much of the battle.

## Installation requirements

To run these scripts you'll need a number of python packages, nearly all of which are available from `conda-forge`. Notably, `enspara` is not, but the nicest way to install it is to generate a conda env with its dependencies, then use pip to install it into that environment. More details on all of this follow.

### Getting an env with everything you need

Use the following command to get every dependency you need in one fell swoop:
```
mamba create -n msms -c conda-forge mdtraj scipy jug pyemma deeptime mpi4py cython matplotlib jupyter-notebook
```
- `jug`: a package that allows task based concurrent workload management within python. Sometimes surprising, but often quite handy for a cluster reading files from a shared file-system.
- `mdtraj`: probably a dependency of at least pyemma, but it is needed for the enspara install procedure described below.
- `scipy`: obviously useful, needed for enspara, not sure whether it's a dependency of some of the other libraries listed here.
- `pyemma`: the old 'Noe and Co.' msm construction library. Not always as great as deeptime anymore, and poorly maintained, but dihedral-based featurization still works and that's what we need from it.
- `deeptime`: The new 'Noe and Co.' msm construction library. Situates itself as being your one-stop-shop for kinetic modeling \& ML, but doesn't have many biomolecule specific things like pyemma's featurization facilities.
- `mpi4py`: a way for python programs to use an MPI for parallelism. Enspara uses this to achieve multinodal clustering (which is helpful because it allows you to use more IO bandwidth from multiple nodes to read your dataset).
- `cython`: enspara has compiled components that use `cython` as the compiler.
- `matplotlib`: Nice to have around.
- `jupyter-notebook`: also nice to have around.

You could also consider adding `scikit-learn` or other modules with other clustering codes; trying other clustering methods is a good idea, but isn't needed to get the scripts in this repository working.
  
With the above environment **active**, install [enspara](https://github.com/bowman-lab/enspara) with pip (execute the following in the directory you'd like enspara's source to be located in):
```
git clone https://github.com/bowman-lab/enspara
cd enspara
pip install -e .
```

## Driving the scripts
### feature based clustering and reduction
Because there are a lot of variables that need to be handed between the various scripts, I've organized the inputs to them as .json configuration files that the scripts read, and also write to. The idea is that one script creates outputs another script needs, and those paths go into the config so it's pretty easy to keep the process rolling once featurization works. Obviously this is fragile and there are a lot of ways it could break, so watch your step.[^1]

Each of the feature based python scripts should have an sbatch script that shows an example of how I'd ask for resources for that script with a large trajectory dataset. The pragmas in these should be adjusted to suit your computing environment, but they hopefully will give an idea for how you can use the resources you have to get the work done.

The RMSD clustering is a lot simpler; because it's purely geometric, and because distances are computed on the fly, it's just an sbatch that drives a tool packaged with enspara. The reassign tool is similarly simple.

## Received wisdom
 - Use a reasonably multidimensional set of features--a collection of atom positions, or more than 10 geometric features--to make biomacromolecule MSMs.  
 - These then get compared, potentially with feature scaling, using some kind of norm (often L2), which is the metric for clustering. 
 - 'CV's that are more human interpretable are often not great choices for features because they'll have too much degeneracy, and because they may be subtly difficult to scale relative to one another.
 - The locations of clusters in phase space isn't as sensitive to the density of your samples as MSM construction may be, so if you have a huge dataset don't be afraid to cluster on a downsampled subset, then reassign frames to that subset. 
 - Because reassignment is embarassingly parallel, it is often much easier to 'fit' onto a computing resource.
 - There are lots of ways clustering can go wrong, so manually inspecting your results in whatever ways make sense for your system (including visualizing cluster centers) is a very good idea.

[^1]: I think I could have done better by using a yaml, since json doesn't permit comments. That'd be a nice upgrade to these scripts, if ever there was time to do it.