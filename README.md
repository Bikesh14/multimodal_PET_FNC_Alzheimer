# Analysis and Classification of Alzheimer’s Disease via Multimodal Fusion of PET and fMRI Data 

This repository contains code and data related to the analysis and classification of Alzheimer's disease (AD) using multimodal fusion of amyloid-PET and functional network connectivity (FNC) data via independent component analysis (ICA).

## Repository Structure

- **dataset/**  
  Contains ICA-derived feature matrices (component loadings) used as input for classification models.  

- **Independent_Components/**  
  Includes independent components (ICs) obtained from ICA of multimodal data. Each set of ICs has been remapped to their respective spatial domains (FNC and PET).

- **MATLAB codes (data_preparation, processing and analysis)/**  
  Contains all MATLAB scripts used for preprocessing, matching datasets, performing ICA, and feature extraction from raw PET and fMRI data.

- **data_analysis/**  
  Includes statistical analyses and results.

- **feature_analysis_and_classification.ipynb**  
  Jupyter notebook for model training, evaluation, and visualization using the extracted features.

## Acknowledgement
Data used in this study were obtained from the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)** database.
 