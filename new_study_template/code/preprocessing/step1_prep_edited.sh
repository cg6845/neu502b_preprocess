#!/bin/bash

source globals.sh

subj=$1
session=$2
subj_dir=$3

echo "Converting subject $subj session $session"

# Full path to DICOM folder
dicom_path=$scanner_dir/$subj_dir

echo "DICOM path: $dicom_path"

# Run heudiconv
singularity exec --cleanenv \
	    --bind $scanner_dir:/data \
	    --bind $project_dir:$project_dir\
    /jukebox/hasson/singularity/heudiconv/heudiconv-v0.8.0.simg \
    heudiconv -f reproin \
    --subject $subj \
    --ses $session \
    --bids \
    --outdir $bids_dir \
    --files /data/$subj_dir

echo "Conversion complete."
