#! /bin/bash
#
# Jonathan King, Jan 22, 2021.
#
# This scrip is an example of how to generate all SidechainNet datasets for all
# available CASPS. It assumes you have cloned both ProteinNet and SidechainNet repos,
# have downloaded the raw, text data linked in ProteinNet, and stored the data in the
# ProteinNet repo as follows:
#       - proteinnet/
#           - data/
#               - caspX/
#                   - caspX/
#                       - targets/ 
#                           - T089.pdb
#                           _ ... .pdb
#                       - testing
#                       - validation
#                       - training_30
#                           ...
#                       - training_100
#
# Temporarily parsed ProteinNet files will live in sidechainnet/data/proteinnet.
# Generated SidechainNet files will live in sidechainnet/data/sidechainnet.
#
# To run, replace the first two variables with the paths to your respective repos.


PATH_TO_PROTEINNET_REPO=/home/jok120/proteinnet
PATH_TO_SIDECHAINNET_REPO=/home/jok120/sidechainnet

cd ${PATH_TO_SIDECHAINNET_REPO}/sidechainnet

echo "Processing CASP7..."
PN_PATH=${PATH_TO_PROTEINNET_REPO}/data/casp7/casp7
python create.py --training_set all $PN_PATH | tee casp7.log
rm ../data/proteinnet/*

echo "Processing CASP8..."
PN_PATH=${PATH_TO_PROTEINNET_REPO}/data/casp8/casp8
python create.py --training_set all $PN_PATH | tee casp8.log
rm ../data/proteinnet/*

echo "Processing CASP9..."
PN_PATH=${PATH_TO_PROTEINNET_REPO}/data/casp9/casp9
python create.py --training_set all $PN_PATH | tee casp9.log
rm ../data/proteinnet/*

echo "Processing CASP10..."
PN_PATH=${PATH_TO_PROTEINNET_REPO}/data/casp10/casp10
python create.py --training_set all $PN_PATH | tee casp10.log
rm ../data/proteinnet/*

echo "Processing CASP11..."
PN_PATH=${PATH_TO_PROTEINNET_REPO}/data/casp11/casp11
python create.py --training_set all $PN_PATH | tee casp11.log
rm ../data/proteinnet/*

echo "Processing CASP12..."
PN_PATH=${PATH_TO_PROTEINNET_REPO}/data/casp12/casp12
python create.py --training_set all | tee casp12.log
rm ../data/proteinnet/*






