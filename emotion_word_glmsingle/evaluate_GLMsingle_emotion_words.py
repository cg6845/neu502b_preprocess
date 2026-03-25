"""
Run the GLMsingle on emotion words data.
"""

# Adapted from script for running GLMsingle with LB-LLM data.


## TODO:
# Figure out package stuff on the cluster - theoretically done 
# Clone GLMsingle repo on the cluster - done 
# Change paths and filenames throughout - theoretically done (need to double check)
# Adapt to actually use cross-validation for GLMsingle - theoretically done (need to double check)
# Change any other parameters as needed - theoretically done (need to double check)


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




def main(raw_args=None):
    parser = argparse.ArgumentParser(description='Run GLMsingle (version: XX')
    parser.add_argument('--UID', default='sub-001', type=str, help='UID str')
    parser.add_argument('--stimdur', default=2, type=int, help='Stimulus duration in seconds')
    parser.add_argument('--tr', default=2, type=int, help='TR sampling rate')
    parser.add_argument('--n_runs', default=1, type=int, help='Number of runs.')
    parser.add_argument('--pcstop', default = 5, 
                        help='How many PCs to remove if not performing cross-validation. If None, uses standad GLMsingle parameters to perform cross-validation')                      
    parser.add_argument('--fracs', default = 0.05, 
                        help='Fraction of ridge regularization to use if not performing cross-validation. If None, uses standad GLMsingle parameters to perform cross-validation')      
    parser.add_argument('--want_library', default=1, type=int,
                        help='Whether we want to do HRF library estimation. Set to 1 for True, 0 for False')
    parser.add_argument('--test', default=False, type=bool,
                        help='Whether to run test mode and only use one run for testing')
    parser.add_argument('--verbose', default=False, type=bool,
                        help='Whether to print output and not create a log file')
    parser.add_argument('--stimset_fname', default='emotion_word', type=str,
                        help='Stimset filename to use')                     
    parser.add_argument('--STIMSET_DIR', default='/usr/people/bs1799/neu502b/neu502b_fmri/emotion_word_glmsingle/emotion_word_stim',
                        type=str, help='Directory to fetch stimsets from with run and dicom path into')         
    parser.add_argument('--DESIGN_MATRIX_DIR', default='/usr/people/bs1799/neu502b/neu502b_fmri/emotion_word_glmsingle/design_matrices',
                        type=str, help='Directory to fetch design matrices from')                               
    parser.add_argument('--overwrite', default=True, type=bool,
                        help='Whether to overwrite results in case outputdir already exists')       
    parser.add_argument('--external_output_root', default='/usr/people/bs1799/neu502b/neu502b_fmri/emotion_word_glmsingle/output_glmsingle', type=str2none,
                        help='If not None, supply a path to a directory to save outputs to')                    
    parser.add_argument('--FMRI_DATA_DIR', default = '/usr/people/cg6845/neu502b/preprocess/502b_language/pygers_workshop/sample_study/data/bids/derivatives/fmriprep',
                        help='Directory where to fetch fMRI data from')
    parser.add_argument('--brain_R2', default = None, 
                        help = "Threshold R^2 value for determining whether or not voxels are in the noise pool. If None, uses standard GLMsingle parameters to determine best value.")
    args = parser.parse_args(raw_args)

    import glmsingle
    from glmsingle.glmsingle import GLM_single
    plot = False
    
     

    ### Set paths ###
    user = getpass.getuser()
    print(f'Running as user {user}')
    root = '/usr/people/bs1799/neu502b/neu502b_fmri/emotion_word_glmsingle/'
    # if user != 'holson':
    #     root = '/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/glmsingle/'
    # else:
    #     root = '/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/glmsingle/'
    os.chdir(join(root))

    # Create directory for saving data
    if args.external_output_root is None:
        output_root = join(root, 'output_glmsingle')
    else:
        output_root = args.external_output_root    

    ### Arguments for GLMsingle ###


    if args.pcstop is not None:
        pcstop = -args.pcstop
    else:
        pcstop = args.pcstop

    if pcstop == 0:
        pcstop = '-0'  # make sure the string names are correct!
    fracs = args.fracs
    brain_R2 = args.brain_R2

    ### Set output, log, and MRI data directories ###

    UID = args.UID
    identifier = f'UID-{UID}'
    if pcstop is not None:
        identifier += f"_pcstop{pcstop}"
    if fracs is not None:
        identifier += f"_fracs-{fracs}"
    if brain_R2 is not None:
        identifier += f"_brainR2-{brain_R2}"
    if args.want_library == 0:
        identifier += '_noHRF' # Run without HRF library

    OUTPUTDIR = join(output_root,
                     f'output_glmsingle_{identifier}')
    LOGDIR = join(root, 'logs')


    if not args.verbose:
        date = datetime.datetime.now().strftime("%Y%m%d-%T")
        sys.stdout = open(
            join(LOGDIR, f'eval_{args.stimset_fname}_{identifier}_{date}.log'), 'a+')


    # Path for accessing fMRI data
    FMRI_DATA_DIR = join(args.FMRI_DATA_DIR, UID, 'ses-01/func')
    fmri_data_fname = f"{UID}_ses-01_task-langXtask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    fmri_data_path = join(FMRI_DATA_DIR, fmri_data_fname)
    

    print('*' * 40)
    print(vars(args))
    print('*' * 40)

    print(f'\nSave output dir: {OUTPUTDIR}')
    print(f'\nStimset dir: {args.STIMSET_DIR}')
    print(f'\nDesign matrices dir: {args.DESIGN_MATRIX_DIR}')
    print(f'\nLog dir: {LOGDIR}\n')

    if pcstop == '-0':
        pcstop = -0  # revert back

    ### Organize BOLD data, design matrices, metadata ###


    # Design matrix
    design_fname = f'design_matrices_{args.stimset_fname}_{UID}.pkl'
    design = pd.read_pickle(join(args.DESIGN_MATRIX_DIR, design_fname))

    # Associated stimset
    stimset_fname = f'stimset_{args.stimset_fname}_{UID}.csv'
    stimset = pd.read_csv(join(args.STIMSET_DIR, stimset_fname))

    ## Assertions
    # Assert that all design matrices have the same shape
    assert len(np.unique([x.shape[0] for x in design])) == 1, "design matrices for each run do not have the same shape"
    assert len(np.unique([x.shape[1] for x in design])) == 1, "design matrices for each run do not have the same shape"
    # # Assert that the design matrix has expected width (based on IPS)
    # assert design[0].shape[0] == stimset.expected_IPS.values[0], "design matrix does not have expected width (IPS)"
    # # Assert that the first design matrix col item is the same as the first stimset item
    # assert design[0][5, :].argmax() == stimset.iloc[0, :].item_id - 1, "first chronological item in design matrix does not match first stimset item"


    ### Load data ###
    data = []

    ## NOT NEEDED

    # # Iterate through the runs
    # _, idx = np.unique(stimset.run_idx.values, return_index=True)
    # run_ids = stimset.run_idx.values[np.sort(idx)]
    # assert len(run_ids) == len(design), "number of run ids in stimset does not match number of runs in design matrix"

    # print_cols = ['run_id', 'run_idx', 'session_id', 'dicomnumber', 'dicomid', 'dicom_path', f'nii_{args.preproc}_path',
    #               'expected_IPS']

    n_runs = args.n_runs

    # For each run, load the nii file 
    for i in range(n_runs):
        if args.test:
            if i == 1:
                break
        
        # stimset_run = stimset[stimset.run_idx == run_id]

        # ## Find the unique session number for the run and add it to the session indicator
        # unique_session_num = np.unique(stimset_run['session_id'].values)
        # # Check that there is only one session number per run
        # assert len(unique_session_num) == 1, f"there is more than one unique session number for run_id {run_id}"
        # session_num = int(unique_session_num[0])

        # ## Find the unique nii paths for the run
        # unique_nii_paths = np.unique(stimset_run[f'nii_{args.preproc}_path'].values)
        # # Check that there is only one unique nii path per run
        # assert len(unique_nii_paths) == 1, f"there is more than one unique nii path for run_id {run_id}"
        # nii_path = unique_nii_paths[0]

        # ## Print out the columns for the stimset for checking/print purposes
        # df_print = stimset_run[print_cols]
        # # Check that there is only one unique value for the run in each of the print_cols
        # assert len(df_print.drop_duplicates()) == 1, f"there is more than one unique value in stimset columns that should have one value per run for run_id {run_id}"
        # vals_print = df_print.values[0]

        # print()
        # print(dict(zip(print_cols, vals_print)))

        ## Load the nii data
        file = np.array(nib.load(fmri_data_path).dataobj)
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

    # add changing parameters
    if pcstop is not None:
        opt['pcstop'] = pcstop
    if fracs is not None:
        opt['fracs'] = fracs
    if brain_R2 is not None:
        opt['brainR2'] = brain_R2

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
