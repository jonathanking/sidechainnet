# Version 1.0 Update Notes
SidechainNet v1.0 introduces new features and several breaking changes. 

This release coincides with the publication of our recent [research paper](https://doi.org/10.1101/2023.10.03.560775) where
we utilized molecular dynamics forcefields as gradients for training OpenFold (AlphaFold2). This work was developed
as part of Jonathan King's PhD thesis work. We are taking this opportunity to make public the
useful changes to SidechainNet that have been developed as part of our research. Jonathan has graduated from the Koes Lab
and will no longer be actively developing SidechainNet. However, we are interested in its continued development and
welcome any interested parties to contact us at dkoes@pitt.edu, or by opening an issue here.

## New Data
* 'scnmin' and 'scnunmin', minimized and unminimized subnsets from our [paper](https://doi.org/10.1101/2023.10.03.560775)
    * taken from the SidechainNet Casp12/100% thinning dataset and energetically minimized with OpenMM
    * ~30k protein structures
    * includes new validation and testing sets from CAMEO
    * access via `scn.load(casp_version=12, casp_thinning='scnmin')` or `scn.load(casp_version=12, casp_thinning='scnunmin')`
    * please see [paper](https://doi.org/10.1101/2023.10.03.560775) for more thorough descriptions
    

## New Features

### Hydrogen Support
* We now support protein structures with hydrogen atoms. See `SCNProtein` for details on adding/removing hydrogen atoms or 
computing energy. 

## Breaking Changes

### Internal data representation via `SCNProtein``

### scn.load (loading data)
* 





