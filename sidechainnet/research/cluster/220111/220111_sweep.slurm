#!/bin/bash
#SBATCH --job-name=sctransformer
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --time=14-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="sidechainnet/research/cluster/220111/%A_%a.out"
#SBATCH --mem=32GB


############################
##       Script Vars      ##
############################
CONDA_ENV="sidechainnetv2"
GIT_BRANCH="research_openmm"
SWEEP_NAME="jonathanking/sidechainnet-sidechainnet_research_cluster_211102/pxdrbwr4"

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
mkdir -p sidechainnet_data
rsync -ru /net/pulsar/home/koes/jok120/repos/sidechainnet/sidechainnet_data sidechainnet_data
echo "Updated repo."

############################
##     Array Job Exec.    ##
############################
export WANDB_DIR="/scr/jok120/wandb"
mkdir -p /scr/jok120/wandb
WANDB_DIR="/scr/jok120/wandb" wandb agent --count 1 $SWEEP_NAME
echo "Completed wandb agent."


############################
##     Copy Files Back    ##
############################
# rsync -avzh --exclude "*.gltf" --exclude "*.png" wandb /net/pulsar/home/koes/jok120/protein-transformer/protein_transformer/wandb
# echo "Completed rsync."

exit 0
