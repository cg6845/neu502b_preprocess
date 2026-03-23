#!/bin/bash
#SBATCH --job-name=eval_glm_lbllm-%j
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --output="out/eval_glm_control_beta-%j.out"
#SBATCH --error="err/eval_glm_control_beta-%j.err"
#SBATCH --mem=120G
#SBATCH -p evlab
source PATH/anaconda/etc/profile.d/conda.sh
conda activate base
cd PATH/EXPT_LBLLM/analyses/glmsingle/GLMsingle/
python evaluate_GLMsingle_lbllm.py --UID "${1}" --stimdur "4"
