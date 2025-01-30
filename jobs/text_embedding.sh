#!/bin/bash
#SBATCH --job-name=text_embedding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=GPU_investor
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/Peraza_BrainDecoder/brain-decoder/jobs/log/%x_%j.out
#SBATCH --error=/home/data/nbc/misc-projects/Peraza_BrainDecoder/brain-decoder/jobs/log/%x_%j.err
# ------------------------------------------

# IB_44C_512G, IB_40C_512G, IB_16C_96G, for running workflow
# investor, for testing
pwd; hostname; date

#==============Shell script==============#
# Load evironment
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
source activate /home/data/nbc/misc-projects/Peraza_BrainDecoder/env/conda_env
set -e

PROJECT_DIR="/home/data/nbc/misc-projects/Peraza_BrainDecoder/brain-decoder"
MODEL_ID="BrainGPT/BrainGPT-7B-v0.2"
SECTION="abstract"

# Setup done, run the command
cmd="python ${PROJECT_DIR}/jobs/text_embedding.py \
    --project_dir ${PROJECT_DIR} \
    --section ${SECTION} \
    --model_id ${MODEL_ID} \
    --device cuda"
echo Commandline: $cmd
eval $cmd

# Output results to a table
echo Finished tasks with exit code $exitcode
exit $exitcode

date

python text_embedding.py --project_dir /Users/julioaperaza/Documents/GitHub/brain-decoder --section abstract --model_id BrainGPT/BrainGPT-7B-v0.2 --device cpu