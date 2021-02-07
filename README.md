# [Some name] a class to compare performance of AL and non-AL models

Example of command to run script:

```$ python main.py -p '/home/khali/Desktop/SCAMmer/' -f 'SCAMS_filtered.csv'```

I used [t-test for means of two independent samples](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind_from_stats.html), since our AL and non-AL models are trained on different data 
(AL training set is a subset of non-AL training set). 

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
   - dataset from Tropsha SCAMDetective based PubChem
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

# Results
Results could be found [here](Description/Update.pdf)


# Folder names

| Model name        | Sampling           | Dataset  | Split |
| ------------- |:-------------:| -----:|-----:|
| RF (RandomForestClassifier)      | N (No sampling) | SF (SCAMS_filtered.csv) | TTS (train_test_split) | 
| LGBM (LGBMClassifier)     | SMOTE    |   SP1 (SCAMS_balanced_with_positive.csv) | B (split_with_butina) |
| XGBC (XGBClassifier) | ADASYN  |    SP2 (SCAMS_added_positives_653_1043.csv) | SS (split_with_scaffold_splitter) |
| ETC (ExtraTreesClassifier) | CondensedNearestNeighbour (CNN)     |    __ | __ |
| __ | InstanceHardnessThreshold (IHT)    |    __ | __ |



For example, LGBM_N_SF_SS stands for run with LGBMClassifier with no sampling on SCAMS_filtered.csv and scaffold_splitter
