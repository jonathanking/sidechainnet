#!/bin/bash
#SBATCH --job-name=scnmin
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --time=2-00:00:00
#SBATCH --partition=any_gpu
#SBATCH --output="sidechainnet/research/cluster/220427/out/slurm-%A_%a.out"


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
SCRIPT_PATH="/net/pulsar/home/koes/jok120/repos/sidechainnet/utils/minimize_scn.py"
DATAPATH="/net/pulsar/home/koes/jok120/scn220502/sidechainnet_casp7_30.pkl"
PICKLEOUTPATH="/net/pulsar/home/koes/jok120/scnmin220502/"

python $SCRIPT_PATH process_index $DATAPATH $PICKLEOUTPATH $SLURM_ARRAY_TASK_ID


############################
##     Copy Files Back    ##
############################
# rsync -avzh --exclude "*.gltf" --exclude "*.png" wandb /net/pulsar/home/koes/jok120/protein-transformer/protein_transformer/wandb
# echo "Completed rsync."

exit 0
