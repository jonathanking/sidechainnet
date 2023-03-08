#!/bin/bash
#SBATCH --job-name=remin
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --partition=any_gpu
#SBATCH --exclude=g122,g001
#SBATCH --output="/net/pulsar/home/koes/jok120/repos/sidechainnet/sidechainnet/research/cluster/230307/%A_%6a.out"

############################
##       Description      ##
############################
# This slurm script will help minimize a SidechainNet dataset.
# This file is meant to be run in 3 steps.
#   1. Run sbatch --export=ALL,JOB_TYPE=setup,CASP_V=12,CASP_T=100 THISFILE.
#           This splits up the desired sidechainnet file into individual protein files.
#   2. Run sbatch --export=ALL,JOB_TYPE=process_index,CASP_V=12,CASP_T=100 --array 0-N_PROTEINS-1 THISFILE
#           This minimizes each protein file independently.
#   3. Run sbatch --export=ALL,JOB_TYPE=cleanup,CASP_V=12,CASP_T=100 THISFILE.
#           This collapses the minimized structures into a single minimized dataset.

############################
##       Script Vars      ##
############################
CONDA_ENV="sidechainnetv2"
GIT_BRANCH="dkoes-research_openmm"
SCRIPT_PATH="/net/pulsar/home/koes/jok120/repos/sidechainnet/sidechainnet/utils/minimize_scn.py"
DATAPATH="/net/pulsar/home/koes/jok120/scn221001/sidechainnet_casp${CASP_V}_${CASP_T}.pkl"
UNMIN_PATH="/net/pulsar/home/koes/jok120/scnmin221001/unmin"
MIN_PATH="/net/pulsar/home/koes/jok120/scnmin230306/min"

############################
##       Environment      ##
############################
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV
echo "Running job ${SLURM_JOB_ID} on node ${SLURMD_NODENAME}."

# Move relevant data to local dir
# mkdir -p sidechainnet_data
# rsync -ruv /net/pulsar/home/koes/jok120/repos/sidechainnet/sidechainnet_data sidechainnet_data

############################
##     Array Job Exec.    ##
############################
if [[ $JOB_TYPE == setup ]]
then
    echo "Setting up files (splitting SidechainNet into separate files for minimization)."
    python $SCRIPT_PATH --step=setup --datapath=$DATAPATH --unmin_path=$UNMIN_PATH --min_path=$MIN_PATH --index=0
    echo "Done."
    echo $(ls -1q $UNMIN_PATH/ | wc -l) files processed.
elif [[ $JOB_TYPE == process_index ]]
then
    echo "python $SCRIPT_PATH --step=process_index --datapath=$DATAPATH --unmin_path=$UNMIN_PATH --min_path=$MIN_PATH --index=$SLURM_ARRAY_TASK_ID"
    python $SCRIPT_PATH --step=process_index --datapath=$DATAPATH --unmin_path=$UNMIN_PATH --min_path=$MIN_PATH --index=$SLURM_ARRAY_TASK_ID
elif [[ $JOB_TYPE == cleanup ]]
then
    echo "Cleaning up files (merging minimized SidechainNet files into a single dict). See $MIN_PATH for output."
    python $SCRIPT_PATH --step=cleanup --datapath=$DATAPATH --unmin_path=$UNMIN_PATH --min_path=$MIN_PATH --index=0
else
    echo "$JOB_TYPE is not a valid command."
fi

exit 0
