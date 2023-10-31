# Version 1.0 Update Notes
SidechainNet v1.0 introduces new features (e.g., OpenMM interaction, hydrogen support) and several breaking changes. 

This release coincides with the publication of our recent [research paper](https://doi.org/10.1101/2023.10.03.560775) where
we utilized molecular dynamics forcefields as gradients for training OpenFold (AlphaFold2). This work was developed
as part of Jonathan King's PhD thesis work. We are taking this opportunity to make public the
useful changes to SidechainNet that have been developed as part of our research. Jonathan has graduated from the Koes Lab
and will no longer be actively developing SidechainNet. However, we are interested in its continued development and
welcome any interested parties to contact us at dkoes@pitt.edu, or by opening an issue here.

## New Data
* 'scnmin' and 'scnunmin', minimized and unminimized data subsets from our [paper](https://doi.org/10.1101/2023.10.03.560775)
    * taken from the SidechainNet Casp12/100% thinning dataset and energetically minimized with OpenMM
    * ~30k protein structures
    * includes new validation and testing sets from CAMEO
    * access via `scn.load(casp_version=12, casp_thinning='scnmin')` or `scn.load(casp_version=12, casp_thinning='scnunmin')`
    * please see [paper](https://doi.org/10.1101/2023.10.03.560775) for more thorough descriptions
    

## New Features

### SCNProtein class
- The default data representation for convenience is now a SCNProtein object. SCNDatasets are comprised of SCNProteins. An SCNProtein has many convenience functions and attributes. It also allows for the measurement of energy, or various simulation/minimization experiments.
  - scn.load_pdb: create a SCNProtein from a pdb file (does not support gaps)
  - can be pickled (SCNProtein.pickle())
  - visually compare two structures for the same protein with SCNProtein.to_3Dmol(other_protein=another_scnprotein)

### SCNDataset class improvements
SCNDatasets make it easy to interface with data while also being a torch Dataset.
- Index by SidechainNet ID or index: `scn_dataset['1A2K_1_A']` or `scn_dataset[0]`
- can be made from list of SCNProteins (SCNDataset.from_scnproteins())
- can be filtered via SCNDataset.filter()
- can be pickled (SCNDataset.pickle()), or converted to FASTA/FASTAs

### ProteinBatch
When iterating over a SCNDataset for pytorch, we now yield iterable ProteinBatch objects instead of tuples
  - access ProteinBatch class properties to get a batched tensor of that data
  - e.g., pb = ProteinBatch([scnprotein1, scnprotein2]); pb.seqs; pb.masks;
  - available attributes
    - angles
    - seqs (aka seqs_onehot)
    - seqs_int
    - secondary
    - masks
    - evolutionary (PSSM)
    - coords
    - is_modified
    - ids
    - resolutions
  - other helper fns
    - cuda()
    - cpu()
    - torch()
    - copy()
    - len()

### Hydrogen Support
We now support protein structures with hydrogen atoms. See `SCNProtein` for details on adding/removing hydrogen atoms or 
computing energy. 
* We now have ATOM_MAP_H and ATOM_MAP_HEAVY for atom mappings with hydrogens and without, respectively
* Building from angles to coordinates with Hydrogens utilizes a residue-level paralell algorithm 

## Breaking Changes
- Protein coordinate representation now has tensors with 3 dimensions (length x num_coords_per_res x 3) instead of 2 ((length * num_coords_per_res) x 3)
* scn.load(...thinning=...) is now casp_thinning
* scn.load returns SCNDataset object by default instead of a Python dictionary representation
- default pad character is NaN, not 0
- SCNProtein.build_coords_from_angles is no longer supported, use SCNProtein.fastbuild to build coordinates from angles instead.
- NUM_COORDS_PER_RES is now 15, not 14. This is primarily to support terminal residue atoms (OXT, H2, and H3)


## To Dos
- [] remove research code from gitignore
- [] discuss batched structure builder
- [] either regenerate data or update code to support both new and old formats of scndata
- [] what was original iteration format, tuples?




## Other Notes


### New feature accessibility
- changed global pad char loc
- allow specification of num cores for scn.create functions
- new fn "scn.create.generate_all_from_proteinnet" to generate all scn data from proteinnet files for curation and upload

    
SCNDatasets
- can be made from list of SCNProteins (SCNDataset.from_scnproteins())
- can filter a SCNDataset via SCNDataset.filter()
- pickle entire dataset, SCNDataset.pickle(), or convert to FASTA/FASTAs

- improved SimilarLengthBatchSampler, collate.py,
- added an alphabet protein example (contains all amino acids) in scn.examples, get_alphabet_protein()
- examples.lightning.AnlgePredictionHelper class, which takes a ProteinBatch as well as true and expected angle tensors and allows the user to evaluate a model trained to predict protein torsional angles
- LitSCNDataModule, a pytorch lightning datamodule for loading and batching sidechainnet data
- LitSidechainTransformer - a pytorch lightning module for training a transformer to predict protein torsional angles
- LoggingHelper - a class for logging training progress, removing ugly code from training script
- MyPLCallbacks - custom pytorch lightning callbacks, includes structure visualization with wandb library
- a plethora of new structure metrics in examples/losses.py (lddt, gdt, etc.)
- NoamOpt implementation in examples/optim.py
- Sidechain only torsional predictive models (no backbone) in examples/sidechain_only_models.py
- ATOM_MAP_14 replaced with ATOM_MAP_HEAVY


HydrogenBuilder
- Uses vector operations to add hydrogens to heavy atom SCNProtein


Research Code
- sidechainnet/research/build_parameter_optim/optimize_build_params.py
  - code to optimize build parameters for sidechainnet. build parameters
  are the hard-coded values used to model proteins (bond lengths etc.). This code allows us to minimized these hard coded values with respect to a certain force field.

