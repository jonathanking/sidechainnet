#!/bin/bash
#SBATCH --job-name=scnmin
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --time=2-00:00:00
#SBATCH --partition=any_gpu
#SBATCH --exclude=g122
#SBATCH --output="/net/pulsar/home/koes/jok120/repos/sidechainnet/sidechainnet/research/cluster/220427/out4/%A_%6a.out"

############################
##       Description      ##
############################
# This slurm script will help minimize a SidechainNet dataset.
# This file is meant to be run in 3 steps.
#   1. Run sbatch --export=ALL,JOB_TYPE=setup,CASP_V=7,CASP_T=30 THISFILE.
#           This splits up the desired sidechainnet file into individual protein files.
#   2. Run sbatch --export=ALL,JOB_TYPE=process_index,CASP_V=7,CASP_T=30 --array 0-N_PROTEINS-1 THISFILE
#           This minimizes each protein file independently.
#   3. Run sbatch --export=ALL,JOB_TYPE=cleanup,CASP_V=7,CASP_T=30 THISFILE.
#           This collapses the minimized structures into a single minimized dataset.
echo "Running job ${SLURM_JOB_ID} on node ${SLURMD_NODENAME}."

############################
##       Script Vars      ##
############################
CONDA_ENV="sidechainnetv2"
GIT_BRANCH="research_openmm"

############################
##       Environment      ##
############################
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

############################
##  Move to Scratch Drive ##
############################
mkdir -p /scr/jok120
cd /scr/jok120
if ! [[ -d sidechainnet ]]
then
    echo "Cloning repo."
    git clone git@github.com:jonathanking/sidechainnet.git
    cd sidechainnet
else
    cd sidechainnet
    git pull
fi
git checkout $GIT_BRANCH
echo "Updated repo."

# Move relevant data to local dir
# mkdir -p sidechainnet_data
# rsync -ruv /net/pulsar/home/koes/jok120/repos/sidechainnet/sidechainnet_data sidechainnet_data

############################
##     Array Job Exec.    ##
############################
SCRIPT_PATH="/net/pulsar/home/koes/jok120/repos/sidechainnet/sidechainnet/utils/minimize_scn.py"
DATAPATH="/net/pulsar/home/koes/jok120/scn220502/sidechainnet_casp${CASP_V}_${CASP_T}.pkl"
UNMIN_PATH="/net/pulsar/home/koes/jok120/scnmin220511/unmin"
MIN_PATH="/net/pulsar/home/koes/jok120/scnmin220512/min"

if [[ $JOB_TYPE == setup ]]
then
    echo "Setting up files (splitting SidechainNet into separate files for minimization)."
    python $SCRIPT_PATH setup $DATAPATH $UNMIN_PATH $MIN_PATH 0
    echo "Done."
    echo $(ls -1q $UNMIN_PATH/ | wc -l) files processed.
elif [[ $JOB_TYPE == process_index ]]
then
    python $SCRIPT_PATH process_index $DATAPATH $UNMIN_PATH $MIN_PATH $SLURM_ARRAY_TASK_ID
elif [[ $JOB_TYPE == cleanup ]]
then
    echo "Cleaning up files (merging minimized SidechainNet files into a single dict)."
    python $SCRIPT_PATH cleanup $DATAPATH $UNMIN_PATH $MIN_PATH 0
else
    echo "$JOB_TYPE is not a valid command."
fi

############################
##     Copy Files Back    ##
############################
# rsync -avzh --exclude "*.gltf" --exclude "*.png" wandb /net/pulsar/home/koes/jok120/protein-transformer/protein_transformer/wandb
# echo "Completed rsync."

exit 0
