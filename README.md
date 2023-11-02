SidechainNet
==============================


>**NEW: SidechainNet v1.0 introduces new features (e.g., OpenMM interaction, hydrogen support) and several breaking changes.** Please see our [release notes](version1_notes.md) for more information. An updated walkthrough is available below.

**[Colab Walkthrough v1.0](https://colab.research.google.com/drive/1-d2p_1G0o4W1DlegyUA6D7ErLUNEsakX?usp=sharing), [Paper](https://doi.org/10.1002/prot.26169)**

SidechainNet is a protein structure prediction dataset that directly extends [ProteinNet](https://github.com/aqlaboratory/proteinnet)<sup>2</sup> by Mohammed AlQuraishi.

Specifically, SidechainNet adds measurements for protein angles and coordinates that describe the complete, all-atom protein structure (backbone *and* sidechain, excluding hydrogens) instead of the protein [backbone](https://foldit.fandom.com/wiki/Protein_backbone) alone.

**This repository provides the following:**
1. SidechainNet datasets stored as pickled Python dictionaries.
2. Methods for loading and batching SidechainNet data efficiently in PyTorch. 
3. Methods for generating protein structure visualizations (`.pdb`, [`3Dmol`](http://3dmol.csb.pitt.edu), `.gltf`) from model predictions.
4. Methods for augmenting SidechainNet to include new proteins and specify dataset organization.
5. Interfacing with OpenMM to compute protein energy and atomic forces.

## Installation
```
conda install -c conda-forge openmm
pip install sidechainnet
```

 
## Summary of SidechainNet data

These attributes can be accessed via `SCNProtein` or `ProteinBatch` objects.
 
| Entry | Dimensionality* | Attribute name | ProteinNet | SidechainNet | 
| :---: | :---: |  :---: | :---: | :---: | 
| Primary sequence<sup>§</sup> | *L* | `seq` | X | X | 
| [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/DSSP_2.html) Secondary structure<sup>\*\*,§</sup> | *L* | `secondary_structure` | X | X |
| [PSSM](https://en.wikipedia.org/wiki/Position_weight_matrix) + Information content | *L x 21* |  `evolutionary` | X | X | 
| Missing residue mask<sup>§</sup> | *L* |  `mask` | X | X | 
| Backbone coordinates | *L x 4<sup>\*\*\*</sup> x 3* |  `coords`, subset `[0:5]` | X | X | 
| Backbone torsion angles | *L x 3* |  `angles`, subset `[0:3]` |  | X | 
| Backbone bond angles | *L x 3* |  `angles`, subset `[3:6]` |  | X | 
| Sidechain torsion angles | *L x 6* |   `angles`, subset `[6:12]` |  | X | 
| Sidechain coordinates | *L x 10 x 3* |  `coords`, subset `[5:15]` |  | X |
||||
| Experimental resolution | *1* |  `resolution` |  | X |
| Unmodified (non-standardized) sequence | *L* |  `unmodified_seq` |  | X |


---

**L* reperesents the length of any given protein in the dataset.

<sup>*\*</sup>Secondary structure is acquired from ProteinNet for training sets only. Blank characters are added by SidechainNet to refer to unmatched amino acids after alignment.(Added January 2021)

<sup>**\*</sup>SidechainNet explicitly includes oxygen atoms as part of the backbone coordinate data in contrast to ProteinNet, which only includes the primary `N, C_alpha, C` atoms.

<sup>§</sup>Stored as string values in the underlying SidechainNet data dictionary.


## Quick-start Examples

> For hands on learning to use SidechainNet, see our walkthrough above. Here are the basics:

### Loading SidechainNet as a SCNDataset


The easiest way to interact with SidechainNet is most likely by using the `SCNDataset` and
`SCNProtein` objects. 

```python
>>> data = scn.load("debug", scn_dataset=True)
>>> data
SCNDataset(n=461)
>>> data["1HD1_1_A"]
SCNProtein(1HD1_1_A, len=75, missing=0, split='train')
>>> data[0]
SCNProtein(2CMY_d2cmyb1, len=23, missing=2, split='train')
```

Available features:
* `SCNDataset` is iterable,
* proteins (`SCNProtein`s) can selected from the dataset by name or index,
* proteins can be visualized with `.to_3Dmol()` and writable to PDBs with `.to_pdb()`. 
* non-terminal hydrogens can be added with `SCNProtein.add_hydrogens()`,

Additionally, all of the attributes below are available directly from the `SCNProtein` object:
* `coords, angles, seq, unmodified_seq, mask, evolutionary, secondary_structure, resolution, is_modified, id, split`



### Loading SidechainNet with PyTorch DataLoaders
The `load` function can also be used to load SidechainNet data as a dictionary of `torch.utils.data.DataLoader` objects. PyTorch `DataLoaders` make it simple to iterate over dataset items for training machine learning models. This method is recommended for using SidechainNet data with PyTorch.

Iterating over our Dataloaders in this manner yields `ProteinBatch` objects that facilitate 
training and data collation/padding.


```python
>>> dataloaders = scn.load(casp_version=12, casp_thinning=30, with_pytorch="dataloaders")
>>> dataloaders.keys()
['train', 'train_eval', 'valid-10', ..., 'valid-90', 'test']
>>> for protein_batch in dataloaders['train']:
....    pred_protein = model(protein_batch.seqs)
....    loss = compute_loss(protein_batch.angles, protein_batch.coords,  # True values
....                        pred_protein.angles, pred_protein.coords)    # Predicted values
....    ...

```


### Quickly build all-atom protein coordinates from torsional angles
An important component of this work is the inclusion of both angular and 3D coordinate representations of each protein. Researchers who develop methods that rely on angular representations may be interested in converting this information into 3D coordinates. For this reason, SidechainNet provides a method to convert the angles it provides into Cartesian coordinates.


```python
>>> d = scn.load(casp_version=12, casp_thinning=30)
>>> p = d[100]
>>> p.fastbuild(inplace=True)
```

### Add hydrogens to structures
Gradient-friendly methods for adding hydrogens to protein structures.
```python
>>> p.add_hydrogens()
>>> # OR
>>> p.fastbuild(add_hydrogens=True, inplace=True)  # builds atoms (including Hs) from angles
```

### Visualization
SidechainNet also makes it easy to visualize both existing and predicted all-atom protein structures. These visualizations are available as `PDB` files, `py3Dmol.view` objects, and Graphics Library Transmission Format (`gLTF`) files.


![StructureBuilder.to_3Dmol() example](./docs/_static/structure_example.png)

```python
>>> sb.to_pdb("example.pdb")
>>> sb.to_gltf("example.gltf")
```

### Compute protein energy (if all-atom coordinates are available)
```python
>>> p.get_energy()
>>> # OR
>>> p.fastbuild(add_hydrogens=True, inplace=True)  # builds atoms (including Hs) from angles 
Quantity(value=-3830.165405454086, unit=kilojoule/mole)
```



## Reproducing or Extending SidechainNet

If you would like to reproduce our work or make modifications/additions to the dataset, please see 
the example we provide in our walkthrough (above). In simple terms, you will need to call `scn.create`
with the desired CASP/ProteinNet information or provide a list of ProteinNet-formatted IDs to
 `scn.create_custom`. Please note that since some data is acquired from ProteinNet directly (e.g., Position Specific Scoring Matrices), protein entries will exclude this data if it was not previously available in ProteinNet.

 ```python
 # Reproduce SidechainNet
scn.create(casp_version=12, training_set=30)

# Create a custom version of SidechainNet
 custom_ids = scn.get_proteinnet_ids(casp_version=12, split="train", thinning=30)
 # Include a protein released in April 2021 (not included in SidechainNet)
 custom_ids += ['7C3K_1_A'] 
scn.create_custom(pnids=custom_ids,
                   output_filename="custom.pkl",
                   short_description="Custom SidechainNet.")
 ```


## Package Requirements

- Python 3
- ProDy (`pip install ProDy`)
    - Biopython
    - numpy
    - scipy
- OpenMM v8 or greater (`conda install -c conda-forge openmm`)
- [PyTorch](https://pytorch.org/get-started/locally/)
- tqdm
- py3Dmol (`pip install py3Dmol`)
- pymol (optional, for `gltf` support, [repo](https://pymolwiki.org/index.php/Linux_Install), [linux install](https://pymolwiki.org/index.php/Linux_Install) requires `libxml2`)
- pytorch_lightning (optional, for model training examples)


## Acknowledgements

Thanks to Mohammed AlQuraishi for building ProteinNet, upon which our work is built. Thanks, also, to [Jeppe Hallgren](https://github.com/JeppeHallgren) for his development of a ProteinNet text record [parser](https://github.com/biolib/openprotein/blob/master/preprocessing.py), which I have used in part here.

 This work is supported by R01GM108340 from the National Institute of General Medical Sciences, is supported in part by the University of Pittsburgh Center for Research Computing through the resources provided, and by NIH T32 training grant T32 EB009403 as part of the HHMI-NIBIB Interfaces Initiative.

Project structure (continuous integration, docs, testing) based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.

## References
1. [SidechainNet: An All-Atom Protein Structure Dataset for Machine Learning](https://doi.org/10.1002/prot.26169). J.E. King, D. Koes. Proteins: Structure, Function, and Bioinformatics (Vol. 89, Issue 11, pp. 1489–1496). Wiley. (2021).
1. [ProteinNet: a standardized data set for machine learning of protein structure.](https://doi.org/10.1186/s12859-019-2932-0). M. AlQuraishi. BMC Bioinformatics 20, 311 (2019).
2. [3dmol.js: molecular visualization with WebGL.](https://doi.org/10.1093/bioinformatics/btu829) N. Rego and D. Koes. Bioinformatics, 31(8):1322–1324, (2014).
 

## Other Resources
Computational Biology Skills Seminar, U.C. Berkeley, May 13, 2021

* [Slides](https://docs.google.com/presentation/d/1yEWBIKjjJ-N1lC7Krw40VlLxg94-cGCrS1R30Pgogq4/edit?usp=sharing)
* [Notebook](https://colab.research.google.com/drive/1J5pUnPuANM6cPXaR2eVNLI6c5wfZgr3X#scrollTo=4tBXWlrt-IWD)
* [Recording](https://youtu.be/1gZAYO7hl80)

## Citation
Please cite our paper(s) if you find SidechainNet useful in your work.
```bibtex
@article{https://doi.org/10.1002/prot.26169,
	author = {King, Jonathan Edward and Koes, David Ryan},
	title = {SidechainNet: An all-atom protein structure dataset for machine learning},
	journal = {Proteins: Structure, Function, and Bioinformatics},
	volume = {89},
	number = {11},
	pages = {1489-1496},
	keywords = {dataset, deep learning, machine learning, protein structure, proteins, software},
	doi = {https://doi.org/10.1002/prot.26169},
	url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.26169},
	eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/prot.26169},
	abstract = {Abstract Despite recent advancements in deep learning methods for protein structure prediction and representation, little focus has been directed at the simultaneous inclusion and prediction of protein backbone and sidechain structure information. We present SidechainNet, a new dataset that directly extends the ProteinNet dataset. SidechainNet includes angle and atomic coordinate information capable of describing all heavy atoms of each protein structure and can be extended by users to include new protein structures as they are released. In this article, we provide background information on the availability of protein structure data and the significance of ProteinNet. Thereafter, we argue for the potentially beneficial inclusion of sidechain information through SidechainNet, describe the process by which we organize SidechainNet, and provide a software package (https://github.com/jonathanking/sidechainnet) for data manipulation and training with machine learning models.},
	year = {2021}
}
```
```bibtex
@article {King2023.10.03.560775,
	author = {Jonathan Edward King and David Ryan Koes},
	title = {Interpreting Molecular Dynamics Forces as Deep Learning Gradients Improves Quality Of Predicted Protein Structures},
	elocation-id = {2023.10.03.560775},
	year = {2023},
	doi = {10.1101/2023.10.03.560775},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/10/05/2023.10.03.560775},
	eprint = {https://www.biorxiv.org/content/early/2023/10/05/2023.10.03.560775.full.pdf},
	journal = {bioRxiv}
}

```


## Copyright

Copyright (c) 2023, Jonathan King
