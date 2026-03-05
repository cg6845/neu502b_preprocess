#!/bin/bash
#SBATCH --job-name=sub002_prep
#SBATCH --output=logs/sub002_%j.out
#SBATCH --error=logs/sub002_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

cd /usr/people/cg6845/neu502b/preprocess/new_study_template/code/preprocessing

bash step1_prep_edited.sh 002 01 Subj2_502b-1550-020326
