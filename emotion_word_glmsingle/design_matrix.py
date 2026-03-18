import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from os.path import join
import pickle
import re




def fmri_output_to_design(subject_id: str,              # keep
                      FMRI_OUTPUT_DIR: str,             # keep 
                      STIMSET_DIR: str,                 # no need
                      OUTPUT_DESIGN_MATRIX_DIR: str,    # keep
                      OUTPUT_STIMSET_DIR: str,          # keep
                      stimset_name: str,                # no need
                      n_runs: int = 20,                 # keep? but always 1 
                      n_unique_stim: int = 880,         # keep 
                      n_stim_per_run: int = 44,         # keep 
                      n_rep: int = 1,                   # keep 
                      timing_tolerance: float = 0.20,   # keep, should be lower 
                      expected_duration: float = 392,   # keep, should be adjusted to account for different lengths for each participant  
                      stim_dur: float = 4,              # keep 
                      isi_dur: float = 4,               # keep
                      break_dur: float = 10,            # keep 
                      tr: float = 2,                    # keep 
                      save: bool = True,
                      overwrite: bool = False,) -> None:
    """
    Check whether the output files from fMRI are precise and match the stimsets created for that subject.
    Assert a bunch of stimulus and timing info.
    Create design matrices for GLMsingle.
    Generate new combined stimset and timekeys files in chronological order.
    :param subject_id: Subject ID
    :param FMRI_OUTPUT_DIR: Path to the output files from fMRI
    :param STIMSET_DIR: Path to the stimsets created for that subject
    :param OUTPUT_DESIGN_MATRIX_DIR: Path to save the design matrices
    :param OUTPUT_STIMSET_DIR: Path to save the stimsets
    :param stimset_name: Name of the stimset/experiment
    :param n_runs: Number of runs (expected)
    :param n_unique_stim: Number of unique stimuli (expected)
    :param n_stim_per_run: Number of stimuli per run (expected)
    :param n_rep: Number of times each stimulus is repeated (expected)
    :param timing_tolerance: Tolerance for difference between expected and empirical timing (in seconds)
    :param expected_duration: Expected duration of each run (in seconds)
    :param stim_dur: Stimulus duration (sentence) (in seconds)
    :param isi_dur: Inter-stimulus interval duration (in seconds)
    :param break_dur: Break duration (in seconds)
    :param tr: Repetition time (in seconds)
    :param save: Whether to save timing table, stimset, and design matrices files
    :param overwrite: Whether to overwrite existing files when saving output files
    :param verbose: Whether to print extra information
    """
    
    # Print all args
    print(f'Running fmri_output_to_design with the following args: {locals()}\n')

    # POTENTIALLY TO CHANGE/DELETE
    timing_table_across_runs = [] # For storing the timing info across runs (all trials, also breaks/ISIs)
    timing_table_stimset_across_runs = [] # For storing the timing info across runs (only trials) with stimset info appended

    # Create n_runs lists of matrices size (trs_in_run, n_unique_stim)
    design_matrices = []
    for run in range(n_runs):
        design_matrices.append(np.zeros((expected_duration // tr, n_unique_stim)))
    print(f'Initialized {n_runs} design matrices of size {expected_duration} seconds by {n_unique_stim} unique stimuli')

    # TO CHANGE                             
    # Generate a simple savestr:
    savestr = f'{stimset_name}_{subject_id}_all_sessions'
    
    # TO CHANGE - JUST READ IN OUTPUT FILE FOR THE SUBJECT 
    # Load participant's stimfile (across all runs)
    # IMPORTANT -- stimset file should have a run_idx column so the runs can be sorted in chronological order
    stimset_file = f"stimset_{stimset_name}_{subject_id}_all.csv" # Load the version with all the stimuli
    
    # stimset_all = pd.read_csv(join(STIMSET_DIR, stimset_file))
    stimset_path = join(STIMSET_DIR, stimset_name)
    stimset_all = pd.read_csv(join(stimset_path, stimset_file))


    # TO CHANGE -- don't need to loop across runs                                    
    """
    Loop across runs in the chronological order they were shown in.
    In this loop, run_idx refers to the chronological index of the run, as opposed to run_id, which identifies which run it is.
    """
    for run_idx in np.arange(1, n_runs+1):
        
        # TO CHANGE -- NOTHING BELOW IS NEEDED 

        # Filter stimset for appropriate run, based on run_idx
        stimset = stimset_all[stimset_all.run_idx == run_idx]
        
        # Make sure we only have the one run included and create a string for run_id (which may differ from run_idx)
        match_run_id = np.unique(stimset['run_id'].values)
        assert (len(match_run_id) == 1), f"Too many run IDs included"
        run_id_str = str(match_run_id[0]).zfill(2)

        # Load the output file for the run (csv file)                                      
        output_file_name = f'{stimset_name}_{subject_id}_session-*_run-{run_id_str}_data.csv'
        output_file_name_search = join(FMRI_OUTPUT_DIR, output_file_name)
        output_file = glob.glob(output_file_name_search)
        
        # Make sure only one output file exists for the run 
        assert len(output_file) == 1, f"More than one output files found for run {run_id_str}"
        
        # Create dataframe for the output file  
        timing_table = pd.read_csv(output_file[0])
      
    
        """
        Check that the output table matches expected parameters and stimset, and combine it with the stimset information.
        """
        # TO CHANGE -- NOT NEEDED

        # Add the session ID from stimset table to the output table (not the original file)
        # Check that there's only one session ID for the run 
        match_session_id = np.unique(stimset['session_id'].values)
        assert (len(match_session_id) == 1), f"Too many session IDs included"
        # Add session ID to the empty column in timing table
        timing_table.session_id = match_session_id[0]    


        # TO CHANGE -- KEEP THIS BUT ADJUST -- just for emotion_word_task

        # Create a copy of the timing table with just the stimuli and calculate the onset time lag (expected - recorded)
        timing_table_sent = timing_table.copy(deep=True)[timing_table.copy(deep=True)['cond_expt'] == 'sentence']
        timing_table_sent['onset_diff_recorded_time_onset'] = abs(timing_table_sent['onset'] - timing_table_sent['recorded_time_onset'])

        # TO CHANGE -- KEEP THIS BUT ADJUST 

        # Perform assertions against the expected params and stimset
        assert len(timing_table_sent) == n_stim_per_run, f"Number of stimuli in {output_file} is not {n_stim_per_run}"
        assert (timing_table_sent['sent_index'].values == stimset['trial_within_run'].values).all(), f"Trial IDs and sent_index in {output_file} do not match the stimset"
        assert (np.diff(timing_table_sent['trial_within_run'].values) == 1).all(), f"Trial IDs in {output_file} are not ascending and incrementing by 1"
        
        # TO CHANGE -- DON'T NEED THIS

        # For overlapping columns between timing_table and stimset, assert that they are the same
        # TODO: fix this so it doesn't break with deviations from planned order
        overlapping_cols = np.intersect1d(timing_table_sent.columns, stimset.columns)
#         for col in overlapping_cols:
#             # print(col)
#             # if the column is not one with problematic values (NaN)
#             if col not in ['babyLM_100M', 'bucket', 'diff_baby_full', 'fully_trained', 'sentence_id']:
#                 assert (timing_table_sent[col].values == stimset[col].values).all(), f"Column {col} in {output_file} does not match the stimset"

        # Get cols in stimset but not in timing_table
        cols_in_stimset_not_in_timing_table = list(np.setdiff1d(stimset.columns, timing_table_sent.columns))
        # Add these cols to timing_table_sent such that we obtain one "unified" timing_table with stimset info too
        # (we've just asserted that the critical cols, e.g., sentence and item_id are identical) <- NOT DONE YET
        timing_table_sent = pd.merge(timing_table_sent, stimset[['item_id'] + cols_in_stimset_not_in_timing_table], on='item_id')
        
        # Add the unified timing table for the run to the overall one
        timing_table_stimset_across_runs.append(timing_table_sent)
        
        print(f' == {output_file} passed assertions against the stimset == ')
        
        
        """
        Check the timing in the timing table
        """

        # TO CHANGE -- KEEP THIS BUT ADJUST

        # Assert that onset and recorded_time_onset are no more than timing tolerance apart
        if not (timing_table_sent['onset_diff_recorded_time_onset'] <= timing_tolerance).all():
            print(f'WARNING: Timing in {output_file} is not precise enough: the max difference between onset and recorded_time_onset is {timing_table_sent["onset_diff_recorded_time_onset"].max()} seconds')
        # Assert a larger latency just to make sure we catch any issues
        assert ((timing_table_sent['onset_diff_recorded_time_onset'].values) <= timing_tolerance + 0.70).all(), f"Timing in {output_file} is not precise enough: the max difference between onset and recorded_time_onset is {timing_table_sent['onset_diff_recorded_time_onset'].max()} seconds"
        assert ((timing_table.recorded_time_offset - timing_table.run_start_time.unique()[0]) == timing_table.offset).all(), f"Recorded time offset does not match intended offset"
        # Sum up the duration in timing table
        total_duration = timing_table['cond_duration'].sum()
        # Assert that it is close to the code_end_time_relative_to_run_start
        empirical_duration = timing_table['code_end_time_relative_to_run_start'].unique()
        assert len(empirical_duration) == 1, f"More than one empirical duration in {output_file}"
        empirical_duration = empirical_duration[0]
        assert abs(total_duration - empirical_duration) <= timing_tolerance, f"Total duration in {output_file} is not precise enough"
        # Assert the timing table duration matches the expected duration
        assert total_duration == expected_duration, f"Total duration in {output_file} is not {expected_duration} seconds"
        print(f'Total duration in {output_file} is {total_duration} seconds, which is close to the empirical duration {empirical_duration:.3f} seconds')

        print(f' == {output_file} passed timing check == ')
        timing_table_across_runs.append(timing_table)

        
        """
        Make design matrix for GLMsingle:
        Get them in GLMsingle design matrix format, where each row is TR=2. Mark a 1 on stimulus onset and next TR (since our stimuli are 4s, 2 TR).
        Let’s use the item_ids I generated for this experiment (col = item_id) which go from 1 to 880. Those are the columns of the design matrix. (which, of course, we need to subtract by 1, because the cols in Python run from 0-399 and not 1-400).

        So for each run, the design matrix will be [total_duration; n_unique_stim] [196; 880]
        """

        # TO CHANGE -- DON'T NEED SEPARATE RUNS 
        # Fill in the design matrices
        run_id_python = run_idx - 1  # python indexed (i.e. we want to fill in the first one at index 0 and not 1). i.e. which design matrix to fill in


        # TO CHANGE -- KEEP BUT ADJUST

        # Iterate through timing_table_sent and for each value in the onset, add a 1 to the design matrix as col = item_id - 1
        for stim in timing_table_sent.itertuples():
            
            # TO CHANGE -- BASICALLY CHANGE COLUMN NAMES ETC. BUT OTHERWISE KEEP EVERYTHING THE SAME 

            onset = stim.onset # We already asserted that onset and recorded_time_onset are no more than timing tolerance apart, so we can just use onset
            
            item_id = stim.item_id 
            if stimset_name == "ss_aud":
                item_id = int(item_id.strip('ss').lstrip('0'))
            
            # Assertions to check the stimulus 
            assert stim.cond_duration == stim_dur, f"Stimulus duration does not match specified stim_dur"
            assert stim.cond_expt == 'sentence', f"Stimulus is not a sentence"
            assert item_id <= n_unique_stim, f"Item ID is greater than the number of unique stimuli"
            
            
            # Assert that onset and item_id are integers
            assert onset - int(onset) == 0, f"Onset is not an integer"
            item_id = int(item_id)

            # Convert the onset from seconds to TRs
            onset_tr = int(onset // tr)

            # Fill in a 1 at the onset time
            design_matrices[run_id_python][onset_tr, item_id - 1] = 1
            


        # Design matrix assertions
        # Check the sum of the design matrices. We expect the sum to be n_stim_per_run
        assert (np.sum(design_matrices[run_id_python]) == n_stim_per_run), f"Sum of design matrix is not {n_stim_per_run}"

        # TO CHANGE -- ADJUST THIS FOR COLUMN NAMES ETC. 

        # Using the timing_table with breaks, find the cond_expt == 'break' and those onsets
        timing_table_break = timing_table[timing_table['cond_expt'] == 'break']
        # Check that the design matrix has NO 1s at those onsets and cond_duration ahead
        for stim in timing_table_break.itertuples():
            onset = stim.onset
            onset_tr = int(onset // tr)
            cond_duration = stim.cond_duration
            cond_duration_tr = int(cond_duration // tr)
            assert cond_duration == break_dur, f"Break duration is not {break_dur} seconds"
            assert (design_matrices[run_id_python][onset_tr:onset_tr+cond_duration_tr, :] == 0).all(), f"Design matrix has 1s at break onsets"            
            
        # Using the timing_table with breaks, find the cond_expt == 'isi' and those onsets
        timing_table_isi = timing_table[timing_table['cond_expt'] == 'isi']
        # Check that the design matrix has NO 1s at those onsets and cond_duration ahead
        for stim in timing_table_isi.itertuples():
            onset = stim.onset
            onset_tr = int(onset // tr)
            cond_duration = stim.cond_duration
            cond_duration_tr = int(cond_duration // tr)
            assert cond_duration == isi_dur, f"ISI duration is not {isi_dur} seconds"
            assert (design_matrices[run_id_python][onset_tr:onset_tr+cond_duration_tr, :] == 0).all(), f"Design matrix has 1s at ISI onsets"


        # Flatten into time axis (sum on dim 1)
        time_axis = np.sum(design_matrices[run_id_python], axis=1)
        assert(np.sum(time_axis) == n_stim_per_run), f"Sum of design matrix time axis is not {n_stim_per_run}"

        print(f' == {output_file} passed design matrix check == \n')

        # Plot example design matrix
        plt.figure(figsize=(20, 20))
        plt.imshow(design_matrices[run_id_python], interpolation='none')
        plt.title(f'example design matrix from run index {run_idx}', fontsize=18)
        plt.xlabel(f'conditions (unique sentences: {n_unique_stim})', fontsize=18)
        plt.ylabel('time (assuming TR = 2s)', fontsize=18)
        plt.tight_layout()
        plt.show()


        sys.stdout.flush()


    # TIMING TABLE: Concatenate across runs
    timing_table_all_runs = pd.concat(timing_table_across_runs)

    # Assertions for timing_table_all_runs
    assert len(timing_table_all_runs[timing_table_all_runs['cond_expt'] == 'sentence']) == n_stim_per_run * n_runs, f"Total number of stimuli in all runs is not {n_stim_per_run * n_runs}"
    if stimset_name != 'vvsa':
        assert timing_table_all_runs.sentence.nunique() == n_unique_stim, f"Total number of unique stimuli in all runs is not {n_unique_stim}"
    assert len(timing_table_all_runs[timing_table_all_runs['cond_expt'] == 'sentence'].run_id.unique()) == n_runs, f"Total number of runs is not {n_runs}"
    # checks that run_start_time is ascending (to ensure that runs were in actually started in consecutive order)
    assert (np.sort(timing_table_all_runs.run_start_time.values) == timing_table_all_runs.run_start_time.values).all(), f"Run start times in timing_table_all_runs are not ascending"

    # TIMING TABLE STIMSET: Concatenate across runs
    timing_table_stimset_all_runs = pd.concat(timing_table_stimset_across_runs)
    timing_table_stimset_all_runs['UID'] = timing_table_stimset_all_runs['UID'].astype(str)
    timing_table_stimset_all_runs['UID_session'] = timing_table_stimset_all_runs['UID'].astype(str) + '_' + timing_table_stimset_all_runs['session'].astype(str)
    first_cols = ['item_id', 'sentence', 'condition', 'UID', 'run_id']
    timing_table_stimset_all_runs = timing_table_stimset_all_runs[first_cols + [col for col in timing_table_stimset_all_runs.columns if col not in first_cols]]

    # Assertions for timing_table_stimset_all_runs
    assert len(timing_table_stimset_all_runs) == n_stim_per_run * n_runs, f"Total number of stimuli in all runs is not {n_stim_per_run * n_runs}"
    if stimset_name != 'vvsa':
        assert timing_table_stimset_all_runs.sentence.nunique() == n_unique_stim, f"Total number of unique stimuli in all runs is not {n_unique_stim}"
    
    stimset_full = pd.read_csv(join(STIMSET_DIR, stimset_name, f"stimset_{stimset_name}_{subject_id}_all.csv")) # Load coherent, full stimset for assertion
    
    # NOTE: the following assertions are problematic if the order of runs deviated from the planned order
    # TODO: figure out how to make these assertions not dependent on the order 
    # assert (stimset_full.item_id.values == timing_table_stimset_all_runs.item_id.values).all(), f"Item IDs in timing_table_stimset_all_runs do not match the item IDs in stimset"
    # assert (stimset_full.sentence.values == timing_table_stimset_all_runs.sentence.values).all(), f"Sentences in timing_table_stimset_all_runs do not match the sentences in stimset"   
    # assert (stimset_full.run_id.values == timing_table_stimset_all_runs.run_id.values).all(), f"Run IDs in timing_table_stimset_all_runs do not match the run IDs in stimset"
    # assert (np.sort(timing_table_stimset_all_runs.run_id.values) == timing_table_stimset_all_runs.run_id.values).all(), f"Run IDs in timing_table_stimset_all_runs do not match the run IDs in timing_table_stimset_across_runs"

    if save:
        # Save timing table with all trials
        if not os.path.exists(join(OUTPUT_STIMSET_DIR, stimset_name, f"timing_table_{savestr}.csv")) or overwrite:
            timing_table_all_runs.to_csv(join(OUTPUT_STIMSET_DIR, stimset_name, f"timing_table_{savestr}.csv"), index=False)
            print(f'Saved timing table to {OUTPUT_STIMSET_DIR}/{stimset_name}/timing_table_{savestr}.csv')
        elif os.path.exists(join(OUTPUT_STIMSET_DIR, stimset_name, f"timing_table_{savestr}.csv")) and not overwrite:
            print(f"File {OUTPUT_STIMSET_DIR}/{stimset_name}/timing_table_{savestr}.csv already exists. Set overwrite=True to overwrite")
        elif os.path.exists(join(OUTPUT_STIMSET_DIR, stimset_name, f"timing_table_{savestr}.csv")) and overwrite:
            timing_table_all_runs.to_csv(join(OUTPUT_STIMSET_DIR, stimset_name, f"timing_table_{savestr}.csv"), index=False)
            print(f'Overwrote timing table to {OUTPUT_STIMSET_DIR}/{stimset_name}/timing_table_{savestr}.csv')

        # Save stimset with all trials
        if not os.path.exists(join(OUTPUT_STIMSET_DIR, stimset_name, f"stimset_{savestr}.csv")) or overwrite:
            timing_table_stimset_all_runs.to_csv(join(OUTPUT_STIMSET_DIR, stimset_name, f"stimset_{savestr}.csv"), index=False)
            print(f'Saved stimset to {OUTPUT_STIMSET_DIR}/{stimset_name}/stimset_{savestr}.csv')
        elif os.path.exists(join(OUTPUT_STIMSET_DIR, stimset_name, f"stimset_{savestr}.csv")) and not overwrite:
            print(f"File {OUTPUT_STIMSET_DIR}/{stimset_name}/stimset_{savestr}.csv already exists. Set overwrite=True to overwrite")
        elif os.path.exists(join(OUTPUT_STIMSET_DIR, stimset_name, f"stimset_{savestr}.csv")) and overwrite:
            timing_table_stimset_all_runs.to_csv(join(OUTPUT_STIMSET_DIR, stimset_name, f"stimset_{savestr}.csv"), index=False)
            print(f'Overwrote stimset to {OUTPUT_STIMSET_DIR}/{stimset_name}/stimset_{savestr}.csv')


    # DESIGN MATRICES: Check all design matrices
    # Check the sum of the design matrices. We expect the sum to be n_unique_stim
    assert (np.sum(design_matrices) == n_unique_stim), f"Sum of design matrices is not 2*{n_unique_stim * n_rep}"

    # Stack them such that we have n_runs * total_duration by n_unique_stim matrix
    design_matrices_stacked = np.vstack(design_matrices)
    # Assert that each stimulus occurred n_rep times (i.e. col sum is n_rep)
    assert (np.sum(design_matrices_stacked, axis=0) == n_rep).all(), f"Each stimulus did not occur {n_rep} times"

    if save:
        # Save the design_matrices list as pickle
        if not os.path.exists(join(OUTPUT_DESIGN_MATRIX_DIR, stimset_name, f"design_matrices_{savestr}.pkl")) or overwrite:
            with open(join(OUTPUT_DESIGN_MATRIX_DIR, stimset_name, f"design_matrices_{savestr}.pkl"), 'wb') as f:
                pickle.dump(design_matrices, f)
            print(f'Saved design matrices to {OUTPUT_DESIGN_MATRIX_DIR}/{stimset_name}/design_matrices_{savestr}.pkl')
        elif os.path.exists(join(OUTPUT_DESIGN_MATRIX_DIR, stimset_name, f"design_matrices_{savestr}.pkl")) and not overwrite:
            print(f"File {OUTPUT_DESIGN_MATRIX_DIR}/{stimset_name}/design_matrices_{savestr}.pkl already exists. Set overwrite=True to overwrite")
        elif os.path.exists(join(OUTPUT_DESIGN_MATRIX_DIR, stimset_name, f"design_matrices_{savestr}.pkl")) and overwrite:
            with open(join(OUTPUT_DESIGN_MATRIX_DIR, stimset_name, f"design_matrices_{savestr}.pkl"), 'wb') as f:
                pickle.dump(design_matrices, f)
            print(f'Overwrote design matrices to {OUTPUT_DESIGN_MATRIX_DIR}/{stimset_name}/design_matrices_{savestr}.pkl')

            
            
### RUN THE FUNCTION 
# uncomment appropriate function call based on experiment/task  

## lbllm adult/child  
# fmri_output_to_design('PED003','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/task_output','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/stimuli','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/glmsingle/design_matrices','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/glmsingle/stimset_outputs','lbllm')


## lbllm repeat session
# fmri_output_to_design('1197b','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/task_output','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/stimuli','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/glmsingle/design_matrices','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/glmsingle/stimset_outputs','lbllm', n_runs = 10, n_unique_stim = 440)


## ss_aud 
# fmri_output_to_design('1198','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/task_output','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/stimuli','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/glmsingle/design_matrices','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/glmsingle/stimset_outputs','ss_aud', n_runs = 2, n_unique_stim = 80, n_stim_per_run = 40, expected_duration = 360)


# vvsa - TO ADD
# fmri_output_to_design('1232','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/task_output','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/stimuli','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/glmsingle/design_matrices','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/glmsingle/stimset_outputs','vvsa', n_runs = 4, n_unique_stim = 120, n_stim_per_run = 30, expected_duration = 220, stim_dur = 2)


## TEST
# fmri_output_to_design('1194','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/task_output','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/stimuli','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/glmsingle/design_matrices/TEST','/nese/mit/group/evlab/u/holson/EXPT_LBLLM/analyses/glmsingle/stimset_outputs/TEST','lbllm')# 