import os
from glob import glob

import jug
import mdtraj as md
import numpy as np
import pyemma
from pathlib import Path
import json
from sys import argv


@jug.TaskGenerator
def featurize_write(featurizer, traj_name, stride, traj_feature_suffix='.h5', parent=None, 
                    rename_depth=-4, chunksize=2000):
    features = pyemma.coordinates.load(traj_name, features=featurizer, stride=stride, chunksize=chunksize)
    if parent:
        tp = Path(traj_name).with_suffix(traj_feature_suffix)
        traj_specific_feature_p = (parent / Path(*tp.parts[rename_depth:])).resolve()
        traj_specific_feature_fn = str(traj_specific_feature_p)
        traj_specific_feature_p.parent.mkdir(exist_ok=True, parents=True)
    else:
        traj_p = Path(traj_name)
        traj_specific_feature_fn = str(traj_p.with_suffix(traj_feature_suffix))
    print(traj_specific_feature_fn)
    np.save(traj_specific_feature_fn, features.astype(np.float32))
    return traj_specific_feature_fn


def read_traj_lists(traj_paths):
    traj_list = []
    for traj_list_path in traj_paths:
        traj_list += Path(traj_list_path).read_text().strip().split()
    return traj_list


@jug.TaskGenerator
def write_feature_traj_fns(feature_traj_list_name, features_list):
    with open(feature_traj_list_name, 'w') as f:
        f.write('\n'.join(features_list))

new_config_tag = 'postfeat'
config_p = Path(argv[1])
with config_p.open('r') as f:
    config = json.load(f)

for protein, specs in config.items():
    # unpack the specs dictionary into python variables
    traj_paths = specs['traj_paths']
    top_path = specs['top_path']
    sel_resids = np.loadtxt(specs['sel_resid_path'], dtype=int)
    selstr = ' or '.join(f'residue {r}' for r in sel_resids)
    # write the translation of this selstr to the specs file.
    specs['selstr'] = selstr
    stride = specs['stride']
    description = specs['description']
    out_stem = Path(specs['out_stem'])
    overwrite = specs['overwrite']
    feat_traj_suff = specs['feat_traj_suff']

    # now start prepping output file names
    if not out_stem.is_dir():
        out_stem.mkdir(parents=True, exist_ok=True)

    # try to figure out which sidechains to save out.
    try:
        which_chis = specs['which_chis']
        include_sidechains = True
        try:
            chi_selstr = specs['chi_selstr']
        except KeyError:
            chi_selstr = selstr
        output_filename = f"{out_stem}/{which_chis}-{description}-feature-fns.txt"
    except KeyError:
        output_filename = f'{out_stem}/bb-{description}-feature-fns.txt'
        include_sidechains = False
    specs['features_filename'] = output_filename
    descriptions_filename = output_filename.replace('feature-fns.txt', 'feature-descriptions.npy')
    specs['descriptions_filename'] = descriptions_filename
    # If feature filename already exists, continue to next component of pipeline.
    if os.path.exists(output_filename):
        if overwrite:
            print(f'Features filename list found, but overwrite set to {overwrite}, so overwriting.')
        else:
            print(f'Features filename list found; but overwrite set to {overwrite}, so refusing to proceed.')
            exit(1) 
    if Path(top_path).suffix == '.prmtop':
        top = md.load_prmtop(top_path)
    else:
        top = md.load(top_path)
    feat = pyemma.coordinates.featurizer(top)
    feat.add_backbone_torsions(selstr=selstr, cossin=True, periodic=False)
    if include_sidechains:
        feat.add_sidechain_torsions(selstr=chi_selstr, cossin=True, periodic=False, which=which_chis)

    # save out description of features
    out_stem.mkdir(exist_ok=True)
    np.save(descriptions_filename, feat.describe())
    if Path(traj_paths[0]).suffix == '.txt':
        traj_list = read_traj_lists(traj_paths)
    else:  # assume these are trajectory filenames
        print("Warning, using python's sorted builtin to put the list into a consistent order.", flush=True)
        traj_list = sorted(list(np.concatenate([glob(traj_path) for traj_path in traj_paths])))
    
    specs['traj_list'] = traj_list

    print('Beginning to generate tasks for reading, featurizing, and writing per trajectory.')
    # spawn a bunch of jug tasks to read and featurize individual trajectories, saving them to independent files
    # NOTE: if parent is defined, will write to new directory structure rename_depth below top levels of traj files
    # within parent. If not, it will write a 'feature.h5' in the same path as the parent of the trajectory file.
    features_list = [featurize_write(feat, traj_name, stride, traj_feature_suffix=feat_traj_suff, parent=out_stem)
                     for traj_name in traj_list]
    # write the file names for each feature traj to a list that can be read by subsequent (clustering) scripts.
    write_feature_traj_fns(output_filename, features_list) 
# save the updated config to a new file.
new_config_p = config_p.parent / f'{config_p.stem}-{new_config_tag}.json'
print('saving updated config with data for next steps to:', new_config_p)
new_config_p.write_text(json.dumps(config, indent=4))
