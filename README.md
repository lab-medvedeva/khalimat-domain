# [Some name] a class to compare performance of AL and non-AL models

Example of command to run script:

```$ python main.py -p '/home/khali/Desktop/SCAMmer/' -f 'SCAMS_filtered.csv'```

I used [t-test for means of two independent samples](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind_from_stats.html), since our AL and non-AL models are trained on different data 
(AL training set is a subset of non-AL training set). I can not use [5x2cv paired t test](http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/), since it requires the
 same training sets for models:
```
t, p = paired_ttest_5x2cv(estimator1=clf1,
                          estimator2=clf2,
                          X=X, y=y,
                          random_seed=1)
``` 



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

# Update
I calculated t-stats. They are presented below

| Metrics        | p_adj           | significant  |
| ------------- |:-------------:| -----:|
| AUC_LB      | 4.04 | False |
| AUC     | 4.72     |   False |
| AUC_UB | 4.47     |    False |
| Accuracy | 0.15     |    False |
| F1 | 0.00045     |    True |
| MCC | 0.00066     |    True |



