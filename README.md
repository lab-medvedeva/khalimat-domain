# SCAMmer v.1: a baseline classification model to predict SCAMs (Small Colloidally-Aggregating molecules)

Structure filtering is an essential step in SAR modelling. I do not have Chemaxon Structure Checker license, therefore I used ChemSAR (http://chemsar.scbdd.com/tools/mol_validate/).

ChemSAR input file should be given as SDF, so I converted SMILES to SDF using [PandasTools](http://rdkit.org/docs/source/rdkit.Chem.PandasTools.html) from RDKit, see [SMILES_to_SDF.py](SCAMmer/SMILES_to_SDF.py).

[filter_ChemSAR_val_res.py](SCAMmer/filter_ChemSAR_val_res.py) filters dataset [aggregator_hts.csv](Data/aggregator_hts.csv) using [ChemSAR](http://chemsar.scbdd.com/tools/mol_validate/) validation results [val_table.csv](Data/val_table.csv). Example of command to run script:

```$ python filter_ChemSAR_val_res.py -a /home/khali/Desktop/Reker_Lab/SCAMs/Data -b val_table.csv -c aggregator_hts.csv```


[SCAMmer.ipynb](SCAMmer/SCAMmer.ipynb): a notebook with baseline model training


## Requirements
To check a list of basic dependencies please see [requirements.txt](SCAMmer/requirements.txt)

## Implemented Model
I trained [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)

## Imbalanced classification
To deal with the issue of imbalanced dataset I used [ADASYN](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.ADASYN.html) approach



# Research Summary

**hypothesis** *Training data sampling can significantly improve the performance of SCAM classification models*

**measure of success** We define a data sampling strategy to improve the performance if a machine learning model that includes the sampling strategy has a significantly better performance compared to the same machine learning model but without the sampling strategy. To evaluate such pairs of models, we will
- keep all other model parameters and pre-processing steps consistent (e.g. dataset, train-test split, parameter optimization strategy, used descriptors, feature selection strategy) 
- performance metric: ROC AUC, MCC, F1
- test set: 60-30% training-test split of original data, while ensuring consistent imbalance in test set (stratified) and using scaffold-based group assignment.
- we will repeat training-test split 10-times and use bonferroni corrected t-test p values to ensure differences are significant.

To ensure that our hypothesis is generalizable and not limited to a single use case, we will explore these different scenarios
- sampling strategies
  - ADASYN
  - SMOTE
  - CondensedNearestNeighbor
  - ActiveLearning
- dataset
   - small Shoichet dataset from Excel sheet
   - larger Shoichet dataset Excel + large set of positive data from AggAdvisor
   - dataset from Tropshka SCAMDetective based PubChem
     - bLactamase https://pubs-acs-org.proxy.lib.duke.edu/doi/suppl/10.1021/acs.jcim.0c00415/suppl_file/ci0c00415_si_002.zip
     - Cruzain https://pubs-acs-org.proxy.lib.duke.edu/doi/suppl/10.1021/acs.jcim.0c00415/suppl_file/ci0c00415_si_003.zip
- descriptor
  - ECFP (Morgan)
  - RDKit Fingerprint
  - MACCs
- feature processing
  - none
  - feature scaling
- parameter optimization
  - none
  - optuna
- models
  - XGBClassifier
  - RandomForestClassifier
  - LGBMClassifier
  - CatBoostClassifier
  - GaussianNB
  - SVC
