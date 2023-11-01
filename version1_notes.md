# Version 1.0 Update Notes
SidechainNet v1.0 introduces new features (e.g., OpenMM interaction, hydrogen support) and several breaking changes. 

This release coincides with the publication of our recent [research paper](https://doi.org/10.1101/2023.10.03.560775) where
we utilized molecular dynamics forcefields as gradients for training OpenFold (AlphaFold2). For more information on
using the methods from that paper, see its [repository](https://github.com/jonathanking/openfold). 

SidechainNet and the OpenMM-Loss method were developed as part of Jonathan King's PhD thesis work. We are taking 
this opportunity to make accessible the useful changes to SidechainNet that have been developed as part of our research. Jonathan has graduated from the Koes Lab and will no longer be actively developing SidechainNet. However, we are interested in its continued development and welcome any interested parties to contact us at dkoes@pitt.edu, or by opening an issue here.

 

## New Features
*Summary: Proteins are now represented as `SCNProtein`s and can be collated/organized into `SCNDataset`s or `ProteinBatch`es.*

### SCNDataset class
SCNDatasets are now the default return type when using `scn.load`. SCNDatasets make it easy to interface with data and are a subclass of torch Dataset. SCNDatasets contain SCNProteins.
- index by SidechainNet ID or numerical index: `scn_dataset['1A2K_1_A']` or `scn_dataset[0]`
- iterate over proteins in a dataset: `for protein in scn_dataset: ...`
- can be created from a list of SCNProteins (`SCNDataset.from_scnproteins()`)
- can be filtered via `SCNDataset.filter()`
- can be pickled (`SCNDataset.pickle()`), or converted to FASTA/FASTAs


### SCNProtein class
The default data representation for a single protein is now a SCNProtein object. A SCNProtein has many convenience functions and attributes. It also allows for the measurement of energy, or various simulation/minimization experiments. Some new features include:
- `scn.load_pdb()`: create a SCNProtein from a pdb file (*does not support gaps*)
- can be pickled (`SCNProtein.pickle()`)
- visually compare two structures for the same protein with `SCNProtein.to_3Dmol(other_protein=another_scnprotein)`
- add hydrogens with `SCNProtein.add_hydrogens()` or quickly convert angles to coordinates with `SCNProtein.fastbuild()`


### ProteinBatch
When iterating over a SCNDataset for pytorch, we now yield iterable ProteinBatch objects instead of named tuples.
ProteinBatch objects are a convenient way to access data for training, and interface with SCNProteins under the hood. 
ProteinBatch objects have class properties that you can access to get a batched tensor of that data.
  - available class properties
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
  - other helper fns to change or inspect underlying data format
    - cuda()
    - cpu()
    - torch()
    - copy()
    - len()

### Hydrogen Support
We now support protein structures with hydrogen atoms. See SCNProtein for details on adding/removing hydrogen atoms or 
computing energy. 
* Building from angles to coordinates with hydrogens via `SCNProtein.fastbuild()` utilizes a residue-level paralell algorithm 

## Breaking Changes
- **New atom mapping definitions**
  - Protein coordinate representation now has tensors with 3 dimensions (`length x num_coords_per_res x 3`) instead of 2 ((`length * num_coords_per_res) x 3`)
  - `NUM_COORDS_PER_RES` is now 15, not 14. This is primarily to support terminal residue atoms (OXT, H2, and H3)
  - `ATOM_MAP_14` replaced with `ATOM_MAP_HEAVY` and its counterpart containing hydrogens, `ATOM_MAP_H`
* `scn.load(...thinning=...)` is now `casp_thinning`
* `scn.load` returns SCNDataset object by default instead of a Python dictionary representation
- default pad character is NaN, not 0
- `SCNProtein.build_coords_from_angles` is no longer supported, use `SCNProtein.fastbuild` to build coordinates from angles instead.
- `scn.load(dynamic_batching)` defaults to False.
- As mentioned above, when iterating over dataloaders, we now yield `ProteinBatch` objects instead of tuples
  - `batch.int_seqs` -> `batch.seqs_int`

## New Data
* 'scnmin' and 'scnunmin', minimized and unminimized data subsets from our [paper](https://doi.org/10.1101/2023.10.03.560775)
    * taken from the SidechainNet Casp12/100% thinning dataset and energetically minimized with OpenMM
    * ~30k protein structures without gaps
    * includes new validation and testing sets from CAMEO
    * access via `scn.load(casp_version=12, casp_thinning='scnmin')` or `scn.load(casp_version=12, casp_thinning='scnunmin')`
    * note: to facilitate training and minimization, proteins with more than 750 residues have been 
    trimmed to a maximum 750 residues, and missing residues at the termini have been trimmed
    * please see [paper](https://doi.org/10.1101/2023.10.03.560775) for more thorough descriptions


## Other new (research oriented) features 
- added an alphabet protein example (contains all amino acids) in `scn.examples.get_alphabet_protein()`
- allow specification of number of cores for scn.create functions to create SidechainNet data
- improved SimilarLengthBatchSampler, collate.py,
- `examples.lightning.AnlgePredictionHelper` class, which takes a ProteinBatch as well as true and expected angle tensors and allows the user to evaluate a model trained to predict protein torsional angles
- LitSCNDataModule, a pytorch lightning datamodule for loading and batching sidechainnet data
- LitSidechainTransformer - a pytorch lightning module for training a transformer to predict protein torsional angles
- LoggingHelper - a class for logging training progress, removing ugly code from training script
- MyPLCallbacks - custom pytorch lightning callbacks, includes structure visualization with wandb library
- a plethora of new structure metrics in examples/losses.py (lddt, gdt, etc.)
- NoamOpt implementation in examples/optim.py
- Sidechain only torsional predictive models (no backbone) in `examples/sidechain_only_models.py`
- `sidechainnet/research/build_parameter_optim/optimize_build_params.py`
  - code to optimize build parameters for sidechainnet. build parameters
  are the hard-coded values used to model proteins (bond lengths etc.). This code allows us to minimize these hard coded values with respect to a certain force field.
- HydrogenBuilder
  - Uses vector operations to add hydrogens to heavy atom SCNProtein
- Code for minimizing SCNDatasets and SCNProteins with OpenMM in `research/minimize*.py`

## Things we're aware of but haven't implemented
- Providing access to later CASP competitions (current implementation is reliant on ProteinNet which does not support past CASP12)
- Energetically minimizing proteins containing gaps
- Parsing PDB files with gaps into SCNProtein objects

