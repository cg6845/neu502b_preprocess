"""
Run the new GLMsingle on LB-LLM data.
"""

# Adapted from Greta's script in May 2024.

# Last edited: 4/28/2025 by biancas



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import os
from os.path import join, exists, split
import sys
import time
import copy
import warnings
from tqdm import tqdm
from pprint import pprint
import sys
import datetime
import getpass
import argparse


def str2none(v):
    """If string is 'None', return None. Else, return the string"""
    if v is None:
        print(f'Already None: {v}')
        return v
    if v.lower() in ('none'):
        print(f'String arg - None: {v}')
        return None
    else:
        return v


## SESSIONS FOR EACH PARTICIPANT
# remember to update for each participant
# NOTE: this is not currently being used in the script. keeping it because it's nice to have this written down, but it's not crucially necessary
# this information is already conveyed in the specified nii file path in the stimset files

d_UID_to_session_list = {
                         'UID': ['SESSIONID', 'SESSIONID'],

                         }

d_UID_to_session_list_ss_aud = {
                                1111: ['SESSIONID'],
                                }

def main(raw_args=None):
    parser = argparse.ArgumentParser(description='Run GLMsingle (version: 1.2, fetched on 2024 June 4)')
    parser.add_argument('--UID', default='864', type=str, help='UID str')
    parser.add_argument('--FL', default='gs', type=str, help='FL')
    parser.add_argument('--stimdur', default=4, type=int, help='Stimulus duration in seconds')
    parser.add_argument('--tr', default=2, type=int, help='TR sampling rate')
    parser.add_argument('--preproc', default='swr', type=str,
                        help='Which preprocessing pipeline to use. Default is swr.')
    parser.add_argument('--pcstop', default=-5, type=int,
                        help='How many PCs to remove. Default is 5.')
    parser.add_argument('--fracs', default=0.05, type=float,
                        help='Fraction of ridge regularization to use. Default is 0.05.')
    parser.add_argument('--want_library', default=1, type=int,
                        help='Whether we want to do HRF library estimation. Set to 1 for True, 0 for False')
    parser.add_argument('--test', default=False, type=bool,
                        help='Whether to run test mode and only use one run for testing')
    parser.add_argument('--verbose', default=False, type=bool,
                        help='Whether to print output and not create a log file')
    parser.add_argument('--stimset_fname', default='lbllm', type=str,
                        help='Stimset filename to use')
    parser.add_argument('--STIMSET_DIR', default='PATH',
                        type=str, help='Directory to fetch stimsets from (with run and dicom path info)')
    parser.add_argument('--DESIGN_MATRIX_DIR', default='PATH',
                        type=str, help='Directory to fetch design matrices from')
    parser.add_argument('--overwrite', default=False, type=bool,
                        help='Whether to overwrite results in case outputdir already exists')
    parser.add_argument('--external_output_root', default='PATH', type=str2none,
                        help='If not None, supply a path to a directory to save outputs to')
    args = parser.parse_args(raw_args)

    import glmsingle
    from glmsingle.glmsingle import GLM_single
    plot = False

    ### Set paths ###
    user = getpass.getuser()
    print(f'Running as user {user}')
    if user != 'XXXX':
        root = 'PATH'
    else:
        root = 'PATH'
    os.chdir(join(root))

    # Create directory for saving data
    if args.external_output_root is None:
        output_root = join(root, 'output_glmsingle')
    else:
        output_root = join(args.external_output_root, 'output_glmsingle')

    ### Arguments for GLMsingle ###
    pcstop = -args.pcstop
    if pcstop == 0:
        pcstop = '-0'  # make sure the string names are correct!
    fracs = args.fracs

    preproc = args.preproc

    ### Set output and log directories ###
    UID_str = f'{args.UID}'
    identifier = f'preproc-{preproc}_pcstop{pcstop}_fracs-{fracs}_UID-{UID_str}'
    if args.want_library == 0:
        identifier += '_noHRF' # Run without HRF library

    OUTPUTDIR = join(output_root,
                     f'output_glmsingle_{identifier}')
    LOGDIR = join(root, 'logs')

    if not args.verbose:
        date = datetime.datetime.now().strftime("%Y%m%d-%T")
        sys.stdout = open(
            join(LOGDIR, f'eval_{args.stimset_fname}_{identifier}_{date}.log'), 'a+')

    print('*' * 40)
    print(vars(args))
    print('*' * 40)

    print(f'Preprocessing pipeline: {preproc} with {pcstop} PCs and {fracs} fracridge')
    print(f'\nSave output dir: {OUTPUTDIR}')
    print(f'\nStimset dir: {args.STIMSET_DIR}')
    print(f'\nDesign matrices dir: {args.DESIGN_MATRIX_DIR}')
    print(f'\nLog dir: {LOGDIR}\n')

    if pcstop == '-0':
        pcstop = -0  # revert back


    ### Organize BOLD data, design matrices, metadata ###


    ## Design matrix
    design_fname = f'design_matrices_{args.stimset_fname}_{UID_str}_all_sessions.pkl'
    design = pd.read_pickle(join(args.DESIGN_MATRIX_DIR, args.stimset_fname, design_fname))

    ## Associated stimset
    stimset_fname = f'stimset_{args.stimset_fname}_{UID_str}_all.csv'
    stimset = pd.read_csv(join(args.STIMSET_DIR, args.stimset_fname, stimset_fname))

    ## Assertions
    # Assert that the design matrix has the correct number of runs
    assert len(design) == stimset.run_id.nunique(), "design matrix does not have the same number of runs as the stimset"
    # Assert that all design matrices have the same shape
    assert len(np.unique([x.shape[0] for x in design])) == 1, "design matrices for each run do not have the same shape"
    assert len(np.unique([x.shape[1] for x in design])) == 1, "design matrices for each run do not have the same shape"
        # if UID_str != '1074': # was collected with IPS = 171 -- LEAVING FOR PARTIAL RUNS LATER
    # Assert that the design matrix has expected width (based on IPS)
    assert design[0].shape[0] == stimset.expected_IPS.values[0], "design matrix does not have expected width (IPS)"
    # Assert that the first design matrix col item is the same as the first stimset item
    assert design[0][5, :].argmax() == stimset.iloc[0, :].item_id - 1, "first chronological item in design matrix does not match first stimset item"


    ### Load data ###
    data = []
    session_num_for_sessionindicator = []

    # Iterate through the runs
    _, idx = np.unique(stimset.run_idx.values, return_index=True)
    run_ids = stimset.run_idx.values[np.sort(idx)]
    assert len(run_ids) == len(design), "number of run ids in stimset does not match number of runs in design matrix"

    print_cols = ['run_id', 'run_idx', 'session_id', 'dicomnumber', 'dicomid', 'dicom_path', f'nii_{args.preproc}_path',
                  'expected_IPS']

    # For each run, load the nii file
    for i, run_id in enumerate(run_ids):
        if args.test:
            if i == 1:
                break

        stimset_run = stimset[stimset.run_idx == run_id]

        ## Find the unique session number for the run and add it to the session indicator
        unique_session_num = np.unique(stimset_run['session_id'].values)
        # Check that there is only one session number per run
        assert len(unique_session_num) == 1, f"there is more than one unique session number for run_id {run_id}"
        session_num = int(unique_session_num[0])
        session_num_for_sessionindicator.append(session_num)

        ## Find the unique nii paths for the run
        unique_nii_paths = np.unique(stimset_run[f'nii_{args.preproc}_path'].values)
        # Check that there is only one unique nii path per run
        assert len(unique_nii_paths) == 1, f"there is more than one unique nii path for run_id {run_id}"
        nii_path = unique_nii_paths[0]

        ## Print out the columns for the stimset for checking/print purposes
        df_print = stimset_run[print_cols]
        # Check that there is only one unique value for the run in each of the print_cols
        assert len(df_print.drop_duplicates()) == 1, f"there is more than one unique value in stimset columns that should have one value per run for run_id {run_id}"
        vals_print = df_print.values[0]

        print()
        print(dict(zip(print_cols, vals_print)))

        ## Load the nii data
        file = np.array(nib.load(nii_path).dataobj)
        file_orig = copy.deepcopy(file)

        ## Assert that the length of the nii file matches the design matrix width
        assert (file.shape[3] == design[i].shape[0]), "length of nii file does not match design matrix width"
        data.append(file)


    # get shape of data volume (XYZ) for convenience
    xyz = data[0].shape[:3]
    xyzt = data[0].shape

    # Print relevant information about data and design matrix shapes
    assert all([x.shape == xyzt for x in data]), "shape of nii data file is not the same across runs"
    print(f'Number of runs in data: {len(data)}.\nShape of Images (brain XYZ and TR): {data[0].shape}')
    if args.test:
        design = design[:1]

    print(
        f'Number of runs in design matrix: {len(design)}, with unique number of TRs across runs: {np.unique([x.shape[0] for x in design])}\n'
        f'and unique number of conditions: {np.unique([x.shape[1] for x in design])}\n'
        f'TR: {args.tr} and stimulus duration (in seconds): {args.stimdur}')

    assert (len(data) == len(design)), "number of runs is not the same for nii data and design matrix"
    assert (xyzt[-1] == design[0].shape[0]), "width of design matrices does not match time dimension of nii files (IPS)"
    sys.stdout.flush()

    ### Visualize sample data and design matrix

    if plot:
        # plot example slice from run 1
        plt.figure(figsize=(20, 6))
        plt.subplot(121)
        plt.imshow(data[0][:, :, 50, 0])
        plt.title('example slice from run 1', fontsize=16)
        plt.subplot(122)
        plt.imshow(data[1][:, :, 50, 0])
        plt.title('example slice from run 2', fontsize=16)
        plt.show()


    # print some relevant metadata
    print(f'There are {len(data)} runs in total\n')
    print(f'N = {data[0].shape[3]} TRs per run\n')
    print(f'The dimensions of the data for each run are: {data[0].shape}\n')
    print(f'The stimulus duration is {args.stimdur} seconds (TR={args.tr})\n')
    print(f'XYZ dimensionality is: {data[0].shape[:3]}\n')
    print(f'Numeric precision of data is: {type(data[0][0, 0, 0, 0])}\n')



    ### Run GLMsingle to estimate single-trial betas ###

    # create a directory for saving GLMsingle outputs
    opt = dict()

    # set important fields for completeness (but these would be enabled by default)
    opt['wantlibrary'] = args.want_library
    opt['wantglmdenoise'] = 1
    opt['wantfracridge'] = 1

    # for the purpose of this example we will keep the relevant outputs in memory
    # and also save them to the disk
    opt['wantfileoutputs'] = [1, 1, 1, 1]
    opt['wantmemoryoutputs'] = [1, 1, 1, 1]
    opt['sessionindicator'] = session_num_for_sessionindicator

    # add changing parameters
    opt['pcstop'] = pcstop
    opt['fracs'] = fracs

    # add wanthdf5 flag
    opt['wanthdf5'] = 1

    # running python GLMsingle involves creating a GLM_single object
    # and then running the procedure using the .fit() routine
    glmsingle_obj = GLM_single(opt)

    # visualize all the hyperparameters
    pprint(glmsingle_obj.params)

    sys.stdout.flush()

    start_time = time.time()
    if args.overwrite or not exists(OUTPUTDIR):
        print(f'running GLMsingle... OUTPUTDIR exists: {exists(OUTPUTDIR)} and is being overwritten: {args.overwrite}')

        # run GLMsingle
        results_glmsingle = glmsingle_obj.fit(
            design,
            data,
            args.stimdur,
            args.tr,
            outputdir=OUTPUTDIR,
            figuredir=join(OUTPUTDIR, 'figures'),)

    else:
        print(f'GLMsingle outputs already exists in directory:\n\t{OUTPUTDIR}')

    sys.stdout.flush()
    elapsed_time = time.time() - start_time

    print(
        '\telapsed time: ',
        f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
    )

    print('finished running GLMsingle')


if __name__ == '__main__':
    main()
