#!/bin/bash
#SBATCH --job-name=goodjob
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=7-00:00:00
#SBATCH --partition=any_gpu
#SBATCH --exclude=g122
#SBATCH --output="/net/pulsar/home/koes/jok120/repos/sidechainnet/sidechainnet/research/cluster/221004/%A_%6a.out"
#SBATCH -c 4

############################
##       Description      ##
############################
# This slurm script simply accesses python/wandb/ML training commands from a file and runs
# them. They must include references to the Python/script variables listed below.
echo "Running job ${SLURM_JOB_ID} on node ${SLURMD_NODENAME}."

############################
##       Script Vars      ##
############################
CONDA_ENV="sidechainnetv2"
GIT_BRANCH="dkoes-research_openmm"
SCN_PYTHON="/net/pulsar/home/koes/jok120/anaconda3/envs/sidechainnetv2/bin/python"
TRAIN_PL="/net/pulsar/home/koes/jok120/repos/sidechainnet/sidechainnet/research/cluster/train_pl.py"
SCN_DATA="/net/pulsar/home/koes/jok120/scnmin220915/scn_minimized.pkl"
CMD_FILE="/net/pulsar/home/koes/jok120/repos/sidechainnet/sidechainnet/research/cluster/221004/221004cmds.txt"

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

############################
##     Array Job Exec.    ##
############################
cmd=`sed "${SLURM_ARRAY_TASK_ID}q;d" "${CMD_FILE}"`

if [ "$cmd" != "" ]; then 
    # Because our commands will have variables (like SCN_PYTHON, TRAIN_PL), we substitute them
    cmd=$(eval echo "$cmd")
    echo "Running ${cmd}"
    eval $cmd;
else
    echo "Line $SLURM_ARRAY_TASK_ID is empty in ${CMD_FILE}."
fi 

exit 0
