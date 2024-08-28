import os
from pathlib import Path
import pickle
import jug
import numpy as np
from sys import argv
import json
import tables
from itertools import islice


# Straight from https://docs.python.org/3/library/itertools.html#itertools.batched
def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

# helper to iterload_ra. Takes a batch of keys, a tables handle, stride, and dtype
# returns an ndarray that is concatenated along the first dimension of the array from handle
def _build_concat(keys, shapes, lengths, tables_handle, stride, dtype):
    # shapes = np.array([tables_handle.get_node(where='/', name=k).shape 
                    #    for k in keys], dtype=int)
    # from enspara.ra load source-code
    # lengths = [(shape[0] + stride - 1) // stride for shape in shapes]
    # splat in case there are more dimensions, but none of this code will work if they're ragged
    shapes = np.array(shapes)
    concat_shape = (sum(lengths), *shapes[0,1:])
    # allocate empty array to put stuff in, below
    concat = np.zeros(concat_shape, dtype=dtype)
    # put the stuff in the array, one key at a time, to avoid ram-use spike.
    start = 0
    for k in keys:
        node = tables_handle.get_node(where='/', name=k)[::stride]
        end = start + len(node)
        concat[start:end] = node
        start = end 
    return concat


# read an RA in interator form. Returns a tuple, the first element of whic is a generator
# that will create concatenations of ragged arrays along the first (trajectory) dimension,
# in batches of size batchsize. The second element is the file handle into the PyTable 
# (the ragged array). 
# NOTE: handle must be MANUALLY CLOSED after the generator has been used.
def iterload_ra(fn, dtype=np.float32, key_inds=None, keys=..., stride=1, batchsize=1, handle=None):
    if not handle:
        handle = tables.open_file(fn)
    if key_inds:
        node_list = handle.list_nodes('/')
        keys = [node_list[i].name for i in key_inds]
    elif keys is Ellipsis:
        keys = [k.name for k in handle.list_nodes('/')]
    shapes = np.array([handle.get_node(where='/', name=k).shape 
                    for k in keys], dtype=int)
    lengths = [(shape[0] + stride - 1) // stride for shape in shapes]
    if len(keys) == 1:
        print(f"Found only one key ('{keys[0]}') returning that as numpy array")
        return np.array(handle.get_node('/' + keys[0])[::stride])
    if batchsize > 1:
        param_pack = zip(
            batched(keys, batchsize), 
            batched(shapes, batchsize), 
            batched(lengths, batchsize)
        )
        return (_build_concat(bkeys, bshapes, blengths, handle, stride, dtype) 
                for bkeys, bshapes, blengths in param_pack), lengths, handle
    else: 
        return (np.array(handle.get_node(where='/', name=k)[::stride], 
                         dtype=dtype, copy=False) 
                for k in keys), lengths, handle


@jug.TaskGenerator
def minikm_deeptime_cluster(tica_filename, fit_stride, assign_stride=1, batchsize=10, n_clusters=50, overwrite=False):
    from enspara import ra
    from deeptime.clustering import MiniBatchKMeans

    if assign_stride != fit_stride:
        print('WARN: assign_stride of ', assign_stride, ', but fit_stride of ', fit_stride, flush=True)
    tica_fp = Path(tica_filename)
    tica_stem = tica_fp.stem
    atag = tica_stem.replace('-tica-reduced', '')
    clustag = f'{atag}-k-{n_clusters}'
    clusterer_fp = tica_fp.parent / f'{clustag}-mini-cluster-object.pkl'
    # check for overwrite before anything hard.
    if clusterer_fp.is_file() and not overwrite:
        print('cluster file already exists; aborting.')
        return clusterer_fp

    clusterer = MiniBatchKMeans(n_clusters)  # for future ref, additional params go here.
    trj_generator, lengths, handle = iterload_ra(tica_filename, 
                                                 batchsize=batchsize, 
                                                 stride=fit_stride)
    for traj_slab in  trj_generator:
        clusterer.partial_fit(traj_slab)
    # need to reset the generator to transform. 
    trj_generator, lengths, handle = iterload_ra(tica_filename, 
                                                 handle=handle, 
                                                 batchsize=batchsize, 
                                                 stride=assign_stride)
    cm = clusterer.fetch_model()
    # note this will correspond to the batch-flattened tica data, and will therefore need 
    # to be reshaped by strided lengths.
    dtrajs_flat = np.concatenate([cm.transform(traj_slab) for traj_slab in trj_generator])
    # need to explicitly close handle when done accessing the feature trajs.
    handle.close()
    r = ra.RaggedArray(dtrajs_flat, lengths=lengths, copy=False)
    dtraj_fp = tica_fp.parent / f'{clustag}-dtrajs.h5'
    # since overwrite is broken, delete file and create new
    if dtraj_fp.is_file() and not overwrite:
        os.remove(dtraj_fp)
    ra.save(str(dtraj_fp), r)
    # save centers; NOTE these are the actual coordinates of the centers, 
    # not data points closest to the center of each cluster.
    centers = cm.cluster_centers
    centers_p = tica_fp.parent / f'{clustag}-centers.npy'
    np.save(centers_p, centers)
    with clusterer_fp.open('wb') as f:
        pickle.dump(cm, f)

    return dtraj_fp


@jug.TaskGenerator
def save_filenames_to_specs(specs, filename, key):
    specs[key].append(jug.bvalue(filename))

new_config_tag = 'postclus'
config_p = Path(argv[1])
print('looking for config file', config_p, flush=True)
with config_p.open('r') as f:
    config = json.load(f)
print('read config file', config_p, flush=True)

# selected params
# chosen_tica_lag = 50
# chosen_cluster_counts = [22, 25, 27, 30, 32, 35]
cf_key = 'cluster_filenames'
for protein, specs in config.items():
    # injects the contents of 'specs' into local namespace!
    locals().update(specs)
    features_list = Path(features_filename).read_text().split()

    outstem_p = Path(out_stem)
    tica_dir = Path(tica_dir)
    # Here we are saving this piece of metadata about the job to the config
    if not tica_dir.is_dir():
        tica_dir.mkdir(parents=True)

    if 'which_chis' in specs.keys():
        tica_p = tica_dir / f"{description}-{which_chis}-{chosen_tica_lag}-tica-reduced.h5"
    else:
        tica_p = tica_dir / f"{description}-bb-{chosen_tica_lag}-tica-reduced.h5"
    for chosen_clusters in chosen_cluster_counts:
        cluster_filename = minikm_deeptime_cluster(tica_p, stride, n_clusters=chosen_clusters, overwrite=overwrite)
        if not cf_key in specs.keys():
            specs[cf_key] = []
        save_filenames_to_specs(specs, cluster_filename, cf_key)

# save the config file to include the new paths we've created.
new_config_p = config_p.parent / f'{config_p.stem}-{new_config_tag}.json'
print('saving updated config with data for next steps to:', new_config_p)
new_config_p.write_text(json.dumps(config, indent=4))
