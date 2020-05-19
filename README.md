SidechainNet
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/jonathanking/sidechainnet.svg?branch=master)](https://travis-ci.com/jonathanking/sidechainnet)


A protein structure prediction data set that includes sidechain information. A direct extension of ProteinNet by Mohammed AlQuraishi.

| ProteinNet | SidechainNet | Entry | Dimensionality | Label in SidechainNet data |
| --- | --- | --- | --- |  --- |
| X | X | Primary sequence | *L x 1* | `seq` |
| X | X | Secondary Structure<sup>⸸</sup> | *L x 8* |  `sec` |
| X | X | [PSSM](https://en.wikipedia.org/wiki/Position_weight_matrix) + Information content | *L x 21* |  `evo` |
| X | X | Missing residue mask | *L x 1* |  `msk` |
| X | X | Backbone coordinates | *L x 4\* x 3* |  `crd`, subset `[0:4]` |
|  | X | Backbone torsion angles | *L x 3* |  `ang`, subset `[0:3]` |
|  | X | Backbone bond angles | *L x 3* |  `ang`, subset `[3:6]` |
|  | X | Sidechain torsion angles | *L x 6* |   `ang`, subset `[6:12]` |
|  | X | Sidechain coordinates | *L x 10 x 3* |  `crd`, subset `[4:14]` |

⸸Currently unsupported in ProteinNet raw files and, therefore, unsupported in SidechainNet.
*SidechainNet explicitly includes Oxygen atoms as part of the backbone coordinate data.

## Downloading SidechainNet

For every existing ProteinNet dataset, a corresponding SidechainNet dataset has been created and can be downloaded on Box [here](https://www.youtube.com/watch?v=dQw4w9WgXcQ). 

There are separate datasets for each available CASP competition (CASP 7-12) as well as each "thinning" of the data as described by ProteinNet (`30, 50, 70, 90, 95, 100`) which all contain the same data, but at varying levels of clustered-downsampling.

## Usage Examples

### Loading SidechainNet as a Python dictionary

```python
import pickle
with open("casp12_100.pt", "rb") as f:
    data = pickle.load(f)
```
Here, SidechainNet is stored as a simple Python dictionary organized by training, validation, and test sets. Within each training set is another dictionary mapping data entry types (`seq`, `ang`, etc.) to a list containing this data type for every protein. In the example below, `seq1`, `ang1`, ... all refer to the same protein.
```python
data = {"train": {"seq": [seq1, seq2, ...],
                  "ang": [ang1, ang2, ...],
                  "crd": [crd1, crd2, ...],
                  "evo": [evo1, evo2, ...],
                  "sec": [sec1, sec2, ...],
                  "ids": [id1, id2, ...],               # Corresponding ProteinNet IDs
                  },
        "valid-30": {...},
            ...
        "valid-90": {...},
        "test": {...},
        "settings": {...}
        }
```

### Using SidechainNet to train an all-atom protein structure prediction model 

```python
import sidechainnet
from sidechainnet.examples.models import RGN
from sidechainnet.utils.structure import angles_to_coords
from sidechainnet.utils.losses import drmsd


def compute_loss(model, seq, tgt_coords):
    """Computes model loss when predicting protein coordinates from amino acid sequence."""
    pred_angles = model(seq)
    pred_coords = angles_to_coords(pred_angles)
    loss = drmsd(pred_coords, tgt_coords)
    return loss

train, train_eval, validations, test = sidechainnet.get_dataloaders("../data/sidechainnet/casp12_100.pt")

model = RGN()

for epoch in range(10):
    # Training epoch
    for seq, tgt_angles, tgt_coords in train:
        loss = compute_loss(model, seq, tgt_coords)
        loss.backwards()
        ...
    
    # Evaluate performance on downsampled training set
    for seq, tgt_angles, tgt_coords in train_eval:
        train_loss = compute_loss(model, seq, tgt_coords)
        ...

    # Evaluate performance on each of the 7 validation sets
    for validation_set in validations:
        for seq, tgt_angles, tgt_coords in validations:
            val_loss = compute_loss(model, seq, tgt_coords)
            ...

# Evaluate performance on test set
for seq, tgt_angles, tgt_coords in test:
    test_loss = compute_loss(model, seq, tgt_coords)
    ...
```


## Directions to Reproduce SidechainNet

If you are just interested in using and interacting with SidechainNet data, please see the above examples. If you would like to reproduce our work and generate SidechainNet, or if you would like to make modifications to the dataset, please follow the directions below to generate SidechainNet from scratch. 

### 1. Download raw ProteinNet data 
```shell script
mkdir -p proteinnet/casp12/targets
cd proteinnet

# Ensure you are downloading the correct CASP version here
wget https://alquiraishi.github.com/proteinnet/data/casp12 casp12/
cd casp12

# Save the path to this directory for generating SidechainNet
PN_PATH=$(pwd)
```
### 2. Download raw CASP target data into `targets` subdirectory
```shell script
wget https://casp.targets/12 targets/
tar -xvf targets/targets.gz
```

### 3. Generate SidechainNet
```shell script
git clone https://github.com/jonathanking/sidechainnet.git
cd sidechainnet/sidechainnet
python create.py $PN_PATH
```
SidechainNet files are now created in `../data/sidechainnet/casp12_100.pt`

## Todo

1. Upload pre-processed files for downloading:
    - PyTorch versions of ProteinNet
    - Raw sidechain data
    - SidechainNet (ProteinNet + sidechain data) as a Python dictionary

### Copyright

Copyright (c) 2020, Jonathan King


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
