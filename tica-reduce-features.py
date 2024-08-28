import os
from pathlib import Path
import pickle
import numpy as np
from sys import argv
import json
import tables


# A repoff of enspara.ra.save that allows us to use a generator instead of an array.
def save_generator_ra(filename, array_gen, outmost_len, tag='arr', compression_level=1):
    n_zeros = len(str(outmost_len)) + 1
    compression = tables.Filters(
        complevel=compression_level,
        complib='zlib',
        shuffle=True
    )
    with tables.open_file(filename, 'w') as handle:
        for i, subarr in enumerate(array_gen):
            zerostr = str(i).zfill(n_zeros)
            atom = tables.Atom.from_dtype(subarr.dtype)
            t = f'{tag}_{zerostr}'
            node = handle.create_carray(
                where='/', name=t, atom=atom,
                shape=subarr.shape, filters=compression
            )
            node[:] = subarr
    return filename


def tica_reduce(feature_filenames, lag_time, tica_path, feature_trajs=None,
                var_cutoff=0.9, use_koopman_reweighting=True, save_tica_obj=False,
                overwrite=False, chunksize=100000, resect_to_frac=None):
    from deeptime.decomposition import TICA
    from deeptime.util.data import timeshifted_split
    if use_koopman_reweighting:
        from deeptime.covariance import KoopmanWeightingEstimator
    # if output and we are not overriding, return filename
    if os.path.exists(tica_path) and not overwrite:
        print('tica file already exists')
        return None
    tica_fn = str(tica_path)
    print('writing to', tica_fn, flush=True)
    # if data was passed in as an argument, don't read it from disk
    if feature_trajs:
        pass
    elif resect_to_frac:
        feature_trajs = []
        for i in feature_filenames:
            feat_traj = np.load(i)
            length = feat_traj.shape[0]
            resect_length = int(length * resect_to_frac)
            feature_trajs.append(feat_traj[:resect_length, :])
    else:
        print('Loading feature trajs for a partial fit.')
    print('set up data load', flush=True)
    # I assume lag is in frames
    tica_fitter = TICA(lagtime=lag_time, var_cutoff=var_cutoff,
                       scaling='kinetic_map')
    lengths = []
    if use_koopman_reweighting:
        """if we are providing many short trajectories we should use a Koopman estimator to reweight the input data. See https://deeptime-ml.github.io/latest/notebooks/tica.html#Koopman-reweighting"""
        koopman_estimator = KoopmanWeightingEstimator(lagtime=lag_time)
        for tn in feature_filenames:
            feature_traj = np.load(tn)
            for lagged_data_tuple in timeshifted_split(feature_traj, lagtime=lag_time, chunksize=chunksize):
                koopman_estimator.partial_fit(lagged_data_tuple)
        reweighting_model = koopman_estimator.fetch_model()
        for tn in feature_filenames:
            feature_traj = np.load(tn, dtype=np.float32)
            lengths.append(len(feature_traj))
            for lagged_data_tuple in timeshifted_split(feature_traj, lagtime=lag_time, chunksize=chunksize):
                tica_fitter.partial_fit(feature_traj, weights=reweighting_model)
    else:
        for tn in feature_filenames:
            feature_traj = np.load(tn, dtype=np.float32)
            lengths.append(len(feature_traj))
            for lagged_data_tuple in timeshifted_split(feature_traj, lagtime=lag_time, chunksize=chunksize):
                tica_fitter.fit(lagged_data_tuple)
    
    tica_model = tica_fitter.fetch_model()
    save_generator_ra(tica_fn, (tica_model.transform(traj).astype(np.float32) 
                      for traj in map(np.load, feature_filenames)), len(lengths))

    np.save(tica_fn.replace('tica-reduced.h5', 'tica-cumvar.npy'),
            tica_model.cumulative_kinetic_variance)

    # save out eigenvectors to get a sense of which features are being selected
    np.save(tica_fn.replace('tica-reduced.h5', 'tica-lsvs.npy'),
            tica_model.singular_vectors_left)
    np.save(tica_fn.replace('tica-reduced.h5', 'tica-rsvs.npy'),
            tica_model.singular_vectors_right)
    np.save(tica_fn.replace('tica-reduced.h5', 'tica-feat-corr.npy'),
            tica_model.feature_component_correlation)

    print('Number of dimensions saved is: ',
          tica_model.output_dimension, 'out of', 
          feature_traj.shape[1], flush=True)

    if save_tica_obj:
        with open(tica_fn.replace('tica-reduced.h5', 'tica-object.pkl'), 'wb') as f:
            pickle.dump(tica_model, f)
    return tica_fn


def save_tica_filenames_to_specs(specs, tica_fn, key='tica_filenames'):
    specs[key].append(tica_fn)

new_config_tag = 'posttica'
config_p = Path(argv[1])
with config_p.open('r') as f:
    config = json.load(f)
print('read config file', config_p, flush=True)
for protein, specs in config.items():
    # this is tricky, but it will turn the contents of specs
    # into local variables with key being variable name.
    locals().update(specs)
    features_list = Path(features_filename).read_text().split()
    # feature_trajs = [np.load(i) for i in features_list]
    outstem_p = Path(out_stem)
    outdir = outstem_p/'tica'
    # Here we are saving this piece of metadata about the job to the config
    specs['tica_dir'] = str(outdir)
    if not outdir.is_dir():
        outdir.mkdir(parents=True)
    specs['tica_filenames'] = []
    for lag_time in tica_lags:
        if 'which_chis' in specs.keys():
            tica_filename = outdir / \
                f"{description}-{which_chis}-{lag_time}-tica-reduced.h5"
        else:
            tica_filename = outdir / \
                f"{description}-bb-{lag_time}-tica-reduced.h5"
        print('About to reduce', tica_filename, flush=True)
        tica_filename = tica_reduce(features_list, lag_time, tica_filename,
                                    var_cutoff=var_cutoff, overwrite=overwrite,
                                    use_koopman_reweighting=False, save_tica_obj=True)

        save_tica_filenames_to_specs(specs, tica_filename)
# save the config file to include the new paths we've created.
new_config_p = config_p.parent / f'{config_p.stem}-{new_config_tag}.json'
print('saving updated config with data for next steps to:', new_config_p)
new_config_p.write_text(json.dumps(config, indent=4))
