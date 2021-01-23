# Reproducing SidechainNet

*A boilerplate script for generation of all datasets (very time-consuming) can be found at [sidechainnet/generate_all_sidechainnets.sh](sidechainnet/generate_all_sidechainnets.sh). For a more detailed/piece-wise generation of SidechainNet, see below.*
 
 For steps 1 and 2, pay careful attention to the subdirectory structure indicated by `cd` and `mkdir` commands when downloading the raw ProteinNet data. **You will not need to download the ProteinNet repository, only the data linked to by ProteinNet's README.** 

After downloading the ProteinNet data, you may clone the SidechainNet repository anywhere you wish. 
 
 The entire procedure to generate SidechainNet takes 2 hrs and 45 minutes on a workstation with 16 cores and 64 GB RAM.


### 1. Download raw ProteinNet data using links from [proteinnet/README.md](https://github.com/aqlaboratory/proteinnet/blob/master/README.md)
```shell script
mkdir -p proteinnet/casp12/targets
cd proteinnet

# Ensure you are downloading the correct CASP version here
wget https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp12.tar.gz -P casp12/
cd casp12
tar -xvf casp12.tar.gz

# Save the path to this directory for generating SidechainNet
PN_PATH=$(pwd)
```

After downloading the data, you should have a directory structure that looks like this:

 ```
proteinnet/
├── casp12/
│  ├── testing
│  ├── training_30
│  ├── training_50
│  ├── training_70
│  ├── training_90
│  ├── training_95
│  ├── training_100
│  ├── validation
│  └──targets/
│    ├── T0283.pbd
│    ├── T0284.pbd
│    ├── ...
│    └── T0386.pbd
└── caspX/
   ├── testing
   ├── training_30
   ├── ...
   └──targets/
     └── ...
 ```

### 2. Download raw CASP target data into `targets` subdirectory
We must also download the target structure files used in the CASP competitions. For each compeition, you can vist the corresponding target data webpage (replace `CASP12` with the competition of interest). 


[https://predictioncenter.org/download_area/CASP12/targets/](https://predictioncenter.org/download_area/CASP12/targets/)



On this webpage, we can identify** a compressed file to download that contains all of the relevant target files. Then, download and unarchive the corresponding file. SidechainNet assumes that that there will be a subdirectory title `targets` within the CASP directory you downloaded from ProteinNet previously. I have selected an appropriate file for the CASP12 targets below.

_**Unfortunately, there doesn't seem to be a consistent naming convention across CASP target download directories. `R` usually stands for refinement, and `0` sometimes refers to protein structure prediction (the task we're interested in). Also, files annotated with `D` or `domain` contain the separate domains for each target, something we don't want. We are only interested in the files that contain the entire target proteins with names like `T0950.pdb` instead of `T0950-D1.pdb`._

```shell script
wget https://predictioncenter.org/download_area/CASP12/targets/casp12.targets_T0.releaseDec022016.tgz -P targets/
tar -xvf targets/*.gz
```

### 3. Generate SidechainNet (in a dierctory of your choosing)
```shell script
git clone https://github.com/jonathanking/sidechainnet.git
cd sidechainnet/sidechainnet
python create.py $PN_PATH
```
SidechainNet files are now created in `sidechainnet/data/sidechainnet/casp12_100.pkl`
