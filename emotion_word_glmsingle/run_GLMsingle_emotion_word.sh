#!/bin/bash
#SBATCH --job-name=eval_glm_emotion_words-%j
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --output="out/eval_glm_control_beta-%j.out"
#SBATCH --error="err/eval_glm_control_beta-%j.err"
#SBATCH --mem=120G
source /usr/people/bs1799/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /usr/people/bs1799/neu502b/neu502b_fmri/emotion_word_glmsingle/
python evaluate_GLMsingle_emotion_words.py --UID "${1}" --stimdur "2"
