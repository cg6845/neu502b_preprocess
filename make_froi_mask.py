# Define and make fROI masks, outputs both 1) who brain mask and 2) masks by frois 
# To run, change parameters in the DEFINE PARAMETERS
# Update LOGS
# 2024-06-07: Created by Selena She <jshe@mit.edu>

import numpy as np
import nibabel as nib
from os.path import join, exists, split
import os

###### DEFINE PARAMETERS ######

# defining global vars
SUBJECTS = '/mindhive/evlab/u/Shared/SUBJECTS'
LANG_PARCEL = '/mindhive/evlab/u/Shared/ROIS_Nov2020/Func_Lang_LHRH_SN220/allParcels_language.nii'
LANG_FIRST_LEVEL = 'DefaultMNI_PlusStructural/results/firstlevel/langlocSN'
LANG_SPMT = 'spmT_0003.nii'
EXPORT_PATH = '/nese/mit/group/evlab/u/holson/LBLLM_analysis/output_localizers/langloc/top10percent'
ROI_MAPS = {1:'LH_IFGorb', 2:'LH_IFG', 3:'LH_MFG', 4:'LH_AntTemp', 5:'LH_PostTemp', 6:'LH_AnG', 7:'RH_IFGorb', 8:'RH_IFG', 9:'RH_MFG', 10:'RH_AntTemp', 11:'RH_PostTemp', 12:'RH_AnG'}

# defining task specific vars
sub = '1197_FED_20240919a_3T1_PL2017'
first_level_dir = join(SUBJECTS, sub, LANG_FIRST_LEVEL)
task_name = LANG_FIRST_LEVEL.split('/')[-1]
print('fetching data from first level directory: ', first_level_dir)

###### DEFINE PARAMETERS ######


def check_unique(matrix):
    '''
    making sure masks are binary
    '''
    # Flatten the matrix to 1D array
    flattened_matrix = matrix.flatten()

    # Get unique values and their counts
    unique_values, counts = np.unique(flattened_matrix, return_counts=True)

    # Print the unique values and their counts
    for value, count in zip(unique_values, counts):
        print(f"Value: {value}, Count: {count}")


def get_top_10_thresholded_t_maps(lang_t_map, lang_parcel):
    '''
    function to threhold the t maps.

    for each parcel in the lang parcel:
        only keep the top 10% numerical values, set all else to 0

    returns the thresholded array
    '''
    # Counts of unique values in lang_parcel_img
    unique_values, counts = np.unique(lang_parcel, return_counts=True)
    print(f"Unique values in lang_parcel: {unique_values}")
    print(f"Counts of unique values: {counts}")

    # Initialize the output array for final t maps
    output_array_all = np.zeros_like(lang_t_map)
    
    # Process each unique value in lang parcel
    for value in unique_values:
        if value == 0.0:  # Skip the zero value
            continue
        
        output_array_a = np.zeros_like(lang_t_map)

        # Get the indices of the current value in lang parcel
        indices = np.where(lang_parcel == value)
        
        # Get the corresponding values in t maps
        corresponding_values = lang_t_map[indices]
        
        # Find the threshold for the top 10% values
        threshold = np.percentile(corresponding_values, 90)
        
        # Get the mask for the top 10% values
        top_10_percent_mask = corresponding_values >= threshold

        # Set the top 10% values in the output array
        output_array_a[indices] = np.where(top_10_percent_mask, corresponding_values, 0)
        output_array_all[indices] = np.where(top_10_percent_mask, corresponding_values, 0)

        # saving individual froi maps
        save_maps_as_npy_and_nii(output_array_a, ROI_MAPS[value])
    
        # Sanity checks
        num_top_10_percent = np.sum(top_10_percent_mask)
        num_set_to_zero = len(corresponding_values) - num_top_10_percent
        
        print(f"Parcel number: {value}")
        print(f"Number of top 10% values kept: {num_top_10_percent}")
        print(f"Number of values set to 0: {num_set_to_zero}")

    return output_array_all


def save_maps_as_npy_and_nii(thresholded_t_map, froi_region):
    '''
    save maps as both nii and npy files
    '''

    # binarize thresholded t maps
    thresholded_t_map[thresholded_t_map!=0] = 1
    
    export_path = f'{EXPORT_PATH}/{sub}'
    export_filename = f'{export_path}/{task_name}_{froi_region}_fROI_mask_binary_top10percent'
    
    # if no dir make dir
    if not exists(export_path):
        os.makedirs(export_path)

    # saving np array to .npy file
    np.save(f'{export_filename}.npy', thresholded_t_map)

    # converting to nfti object and exporting as .nii
    AFFINE = np.array([
        [-2.,0.,0.,90.],
        [0.,2.,0.,-126.],
        [0.,0.,2.,-72.],
        [0.,0.,0.,1.],
        ])

    thresholded_t_maps_img = nib.Nifti1Image(thresholded_t_map.astype(np.int16), affine=AFFINE)
    nib.save(thresholded_t_maps_img, f'{export_filename}.nii')


def main():

    # load lang t maps and parcel
    lang_t_map_img = np.array(nib.load(f'{first_level_dir}/{LANG_SPMT}').dataobj)
    lang_parcel_img = np.array(nib.load(LANG_PARCEL).dataobj)
    
    # this steps iteratively exports all individual frois
    thresholded_t_maps = get_top_10_thresholded_t_maps(lang_t_map_img, lang_parcel_img)

    # now save the map with all frois
    save_maps_as_npy_and_nii(thresholded_t_maps, 'ALL')
    
main()