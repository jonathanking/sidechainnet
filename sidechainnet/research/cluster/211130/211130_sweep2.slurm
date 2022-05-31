#!/bin/bash
#SBATCH --job-name=sctransformer-sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --time=14-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="sidechainnet/research/cluster/211102/slurm-%A_%a.out"


############################
##       Environment      ##
############################
eval "$(conda shell.bash hook)"
conda activate sidechainnet


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
    git checkout research_openmm
else
    cd sidechainnet
    git pull
    git checkout research_openmm
fi
mkdir -p sidechainnet_data
rsync -ru /net/pulsar/home/koes/jok120/repos/sidechainnet/sidechainnet_data sidechainnet_data
echo "Updated repo."

############################
##     Array Job Exec.    ##
############################
export WANDB_DIR="/scr/jok120/wandb"
mkdir -p /scr/jok120/wandb
WANDB_DIR="/scr/jok120/wandb" wandb agent --count 1 koes-group/sidechain-transformer/y8yezn3l
echo "Completed wandb agent."


############################
##     Copy Files Back    ##
############################
# rsync -avzh --exclude "*.gltf" --exclude "*.png" wandb /net/pulsar/home/koes/jok120/protein-transformer/protein_transformer/wandb
# echo "Completed rsync."

exit 0
