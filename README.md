SidechainNet
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/jonathanking/sidechainnet.svg?branch=master)](https://travis-ci.com/jonathanking/sidechainnet)


A protein structure prediction data set that includes sidechain information. A direct extension of ProteinNet by Mohammed AlQuraishi.

| ProteinNet | Sidechainnet | Entry | Dimensionality |
| --- | --- | --- | --- |
| X | X | Primary sequence | *L x 1* |
| X | X | Secondary Structure | *L x 8* |


## Examples

### Loading SidechainNet as a Python dictionary

```python
import pickle
with open("../data/sidechainnet/casp12_100.pt", "rb") as f:
    data = pickle.load(f)
```
Here, SidechainNet is stored in `data` as a simple Python dictionary organized by training, validation, and test sets.
```python
data = {"train": {"seq": [seq1, seq2, ...],
                  "ang": [ang1, ang2, ...],
                  "crd": [crd1, crd2, ...],
                  "ids": [id1, id2, ...],
                  "evo": [evo1, evo2, ...]
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
from sidechainnet.examples.models import RGN as model
from sidechainnet.utils.structure import angles_to_coords
from sidechainnet.utils.losses import drmsd

train, train_eval, validation, test = sidechainnet.get_dataloaders("../data/sidechainnet/casp12_100.pt")

for epoch in range(10):
    for seq, tgt_angles, tgt_coords in train:
        pred_angles = model(seq)
        pred_coords = angles_to_coords(pred_angles)
        loss = drmsd(pred_coords, tgt_coords)
        loss.backwards()
        ...
```


## Directions to Reproduce SidechainNet
### Prerequisites
```shell script
# 1. Download raw ProteinNet data 
mkdir -p proteinnet/casp12/targets
cd proteinnet
wget https://alquiraishi.github.com/proteinnet/data/casp12 casp12/
cd casp12
PN_PATH=$(pwd)

# 2. Download raw CASP target data
wget https://casp.targets/12 targets/
tar -xvf targets/targets.gz
```

### Generate SidechainNet
```shell script
git clone https://github.com/jonathanking/sidechainnet.git
cd sidechainnet/sidechainnet
python create.py $PN_PATH
# SidechainNet files are now created in ../data/sidechainnet/sidechainnet_casp12_100.pt
```

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
