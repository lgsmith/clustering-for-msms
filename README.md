# Clustering for MSM construction
## Received wisdom
It is probably best to have some reasonably multidimensional set of features---a collection of atom positions, or more than 10 geometric features---to make biomacromolecule MSMs. These then get compared (potentially with feature scaling, if that's needed) using some kind of L2 norm. Often 'CV's that are more human interpretable in general are not great choices for features because they'll have two much degeneracy.
