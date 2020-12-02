from __future__ import print_function
import pandas as pd
import numpy as np
import scipy.stats as stats

from sklearn.utils import resample

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, RDKFingerprint
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from modAL.uncertainty import uncertainty_sampling
from imblearn.over_sampling import ADASYN, SMOTE  # upsampling
from imblearn.under_sampling import CondensedNearestNeighbour


DESCRIPTORS = {'MorganFingerprint': AllChem.GetMorganFingerprintAsBitVect,
               'RDKFingerprint': RDKFingerprint,
               'MACCSkeys': MACCSkeys}

MODELS = {'GradientBoosting': GradientBoostingClassifier(),
          'CatBoostClassifier': CatBoostClassifier(silent=True),
          'RandomForestClassifier': RandomForestClassifier(),
          'SVC': SVC(probability=True), 'GaussianNB': GaussianNB(),
          'ExtraTreesClassifier': ExtraTreesClassifier(),
          'LGBM': LGBMClassifier()}

SAMPLING = {'SMOTE': SMOTE(),
            'ADASYN': ADASYN(),
            'CondensedNearestNeighbour': CondensedNearestNeighbour(random_state=0)}

#
# AL_STRATEGY = {'GradientBoosting': uncertainty_sampling,
#                      'CatBoostClassifier': uncertainty_sampling,
#                      'RandomForestClassifier': uncertainty_sampling,
#                      'SVC': SVC(probability=True), 'GaussianNB': GaussianNB(),
#                      'ExtraTreesClassifier': ExtraTreesClassifier(),
#                      'LGBM': uncertainty_sampling}
#
METRICS = ['AUC_LB', 'AUC', 'AUC_UB', 'Accuracy', 'F1', 'MCC']


# from https://github.com/yandexdataschool/roc_comparison/blob/master/compare_auc_delong_xu.py

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight=None):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    sample_weight = None
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    sample_weight = None
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


def calc_auc_ci(y_true, y_pred, alpha=0.95):
    auc, auc_cov = delong_roc_variance(y_true, y_pred)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1
    return auc, ci


def bootstrap_error_estimate(pred, truth, method, method_name="", alpha=0.95, sample_frac=0.5, iterations=1000):
    """
    Generate a bootstrapped estimate of confidence intervals
    :param pred: list of predicted values
    :param truth: list of experimental values
    :param method: method to evaluate performance, e.g. matthews_corrcoef
    :param method_name: name of the method for the progress bar
    :param alpha: confidence limit (e.g. 0.95 for 95% confidence interval)
    :param sample_frac: fraction to resample for bootstrap confidence interval
    :param iterations: number of iterations for resampling
    :return: lower and upper bounds for confidence intervals
    """
    index_list = range(0, len(pred))
    num_samples = int(len(index_list) * sample_frac)
    stats = []
    for _ in range(0, iterations):
        sample_idx = resample(index_list, n_samples=num_samples)
        pred_sample = [pred[x] for x in sample_idx]
        truth_sample = [truth[x] for x in sample_idx]
        stats.append(method(pred_sample, truth_sample))
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    return lower, upper


def butina_cluster(mol_list, cutoff=0.35):
    """
    Reference
    __________
    @article{butina1999unsupervised,
      title={Unsupervised data base clustering based on daylight's fingerprint and Tanimoto similarity:
      A fast and automated way to cluster small and large data sets},
      author={Butina, Darko},
      journal={Journal of Chemical Information and Computer Sciences},
      volume={39},
      number={4},
      pages={747--750},
      year={1999},
      publisher={ACS Publications}
    }
    Make a clusters based Butina clustering algorithm using MorganFingerprints similarity

    Function adapted from:
    https://github.com/PatWalters/Learning_Cheminformatics/blob/master/clustering.ipynb

    Parameters
    __________
    :param mol_list: list of rdkit.Chem.rdchem.Mol object to cluster
    :param cutoff: Tanimoto similarity cutoff

    Return
    _______
    :return: list with cluster id for every molecule from mol_list

    """
    fp_list = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in mol_list]
    dists = []
    nfps = len(fp_list)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1 - x for x in sims])
    mol_clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    cluster_id_list = [0] * nfps
    for idx, cluster in enumerate(mol_clusters, 1):
        for member in cluster:
            cluster_id_list[member] = idx
    return cluster_id_list
