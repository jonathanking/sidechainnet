 # Reproducing SidechainNet
 For steps 1 and 2, pay careful attention to the subdirectory structure indicated by `cd` and `mkdir` commands when downloading ProteinNet data. After ProteinNet data has been downloaded, you may clone the SidechainNet repository anywhere you wish.

### 1. Download raw ProteinNet data using links from [proteinnet/README.md](https://github.com/aqlaboratory/proteinnet/blob/master/README.md)
```shell script
mkdir -p proteinnet/casp12/targets
cd proteinnet

# Ensure you are downloading the correct CASP version here
wget https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp12.tar.gz casp12/
cd casp12
tar -xvf casp12.tar.gz

# Save the path to this directory for generating SidechainNet
PN_PATH=$(pwd)
```
### 2. Download raw CASP target data into `targets` subdirectory
We must also download the target structure files used in the CASP competitions. For each compeition, you can vist the corresponding target data webpage (replace `CASP12` with the competition of interest). 
```shell script
https://predictioncenter.org/download_area/CASP12/targets/
```
On this webpage, we can identify a compressed file to download (the largest and most recent file, assumedly) that contains all of the relevant target files. Then, download and unarchive the corresponding file. SidechainNet assumes that that there will be a subdirectory title `targets` within the CASP directory you downloaded from ProteinNet previously.
```shell script
wget https://predictioncenter.org/download_area/CASP12/targets/casp12.domains_T0.releaseDec022016.tgz targets/
tar -xvf targets/targets.gz
```

### 3. Generate SidechainNet
```shell script
git clone https://github.com/jonathanking/sidechainnet.git
cd sidechainnet/sidechainnet
python create.py $PN_PATH
```
SidechainNet files are now created in `../data/sidechainnet/casp12_100.pkl`