from pathlib import Path
import argparse
import math

import pandas as pd
import numpy as np

#  Importing packages for visualization
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from IPython import display

#  Importing modules from sklearn
import sklearn
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, \
                             RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, \
                            f1_score, fbeta_score
from scipy.stats import ttest_ind_from_stats



#  Importing module for imbalanced learning
from imblearn.over_sampling import ADASYN, SMOTE  # upsampling

from imblearn.under_sampling import CondensedNearestNeighbour  # downsampling


#  Importing module for active learning
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from modAL.multilabel import SVM_binary_minimum, avg_score

# Importing modules to calculate confidence intervals and descriptors
from utilities import calc_auc_ci, bootstrap_error_estimate

from utilities import DESCRIPTORS, MODELS, METRICS

# Importing lightgbm to train classifier
from lightgbm import LGBMClassifier
import optuna.integration.lightgbm as lgb
from xgboost import XGBClassifier

from catboost import CatBoostClassifier
import xgboost as xgb


#  Importing packages to enable processing of chemical structures
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, RDKFingerprint

#  Importing RDLogger to filter out rdkit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

#  Importing package to filter out warnings
import warnings
warnings.filterwarnings("ignore")


from collections import Counter
from itertools import combinations
from mlxtend.evaluate import paired_ttest_5x2cv

#  Importing modules and packages for model tunning
import optuna
from hyperopt import tpe
from scipy import stats


class TrainModel():
    """
    A class to build a binary classification model
    for cheminformatics

    Attributes
    __________
    dataset: pd.DataFrame
             training dataset with SMILES and activity class (0 or 1)
             column 'Smiles String' contains SMILES

    activity_column_name: str,
             name of column with activity label in dataset

    descriptor: str,
             descriptor used for instance
    models: dictionary,
             key - str, name of model
             value - Classifier object

    test_split_r: float,
             test to split ratio

    feature_selection: class or function
             select most informative features

    scaler: class or function
             transform features

    sampling: class
             over/under sample data to handle imbalanced learning issues

    n_features: int,
             number of most informative features to select

    non_AL_stats: pd.DataFrame,
             performance statistics for model implemented without uncertainty sampling

    AL_stats: pd.DataFrame,
             performance statistics for model implemented with uncertainty sampling (AL)
    cv_n: int,
             number of cross-validation splits
    t_test: pd.DataFrame,
             t-test (independent) statistics for AL and non-AL approaches


    Methods
    -------
    calculate_descriptors()
             calculate descriptor from SMILES str

    transform_X()
             transform pandas to numpy

    split_train_val()
             split dataset to test and train using

    feature_scaling()
             scale features

    loc_inf()
             calculate feature importance using feature_selection
             and loc top n_features of most informative
    sampl()
             up or down-sample training set using attribute sampling

    auc_for_modAL()
             calculate ROC-AUC values for AL-model


    fit_model_CV()
             fit model and cross validate

    AL_strategy()
             select most informative samples and train a model

    f_one_mcc_score()
             calculate f1-score and the Matthews correlation coefficient

    calculate_t_test()
             calculate t-test statistics to determine statistical significance
             of observed differences between AL and non-AL models



    """

    def __init__(self, dataset, activity_colunm_name,
                 descriptor, models, test_split_r,
                 feature_selection, scaler=StandardScaler(),
                 sampling=ADASYN(), n_features=300,
                 initial=10):

        self.dataset = dataset.copy()
        self.activity_column_name = activity_colunm_name
        self.descriptor = {descriptor: DESCRIPTORS[descriptor]}
        self.models = models
        self.test_split_r = test_split_r
        self.feature_selection = feature_selection
        self.scaler = scaler
        self.sampling = sampling
        self.n_features = n_features
        self.initial = initial
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.SCAMmer = None
        self.test_predicted = None
        self.non_AL_stats = None
        self.AL_stats = None
        self.cv_n = 10
        self.t_test = None

    def calculate_descriptors(self):
        """
        Converts SMILES to mol_obj and calculate descriptors
        """
        for d_nm, d_fn in self.descriptor.items():
            self.dataset['mol_obj'] = self.dataset['Smiles String'].apply(lambda x: Chem.MolFromSmiles(x))  # Make mol_obj from SMILES
            if d_nm == 'MorganFingerprint':
                self.dataset['{}'.format(d_nm)] = self.dataset['mol_obj'].apply(lambda x: d_fn(x, 3, nBits=2048))  # Calculate Morgan descriptors
            else:
                self.dataset['{}'.format(d_nm)] = self.dataset['mol_obj'].apply(lambda x: d_fn(x))  # Calculate other descriptors

    def transform_X(self, X):
        """
        Convert pandas series to numpy array

        Parameters
        __________
        :param X: pd.Series to convert

        Return
        ______
        numpy array from X
        """
        return np.array(X.tolist())


    def split_train_val(self, dataset, test_split_r):
        """
        Split data into train and test
        """
        X_train, X_test, Y_train, Y_test = train_test_split(
            dataset[[d for d in self.descriptor]],
            dataset[self.activity_column_name],
            test_size=test_split_r)  # Split data into test and train

        print('{} samples in X_train'.format(X_train.shape[0]))  # Print train sample size
        print('{} samples in X_test'.format(X_test.shape[0]))  # Print test sample size

        X_train = self.transform_X(X_train.iloc[:, 0])  # Select 1-st column with calculated descriptors and transform to numpy array
        X_test = self.transform_X(X_test.iloc[:, 0])
        return X_train, X_test, Y_train, Y_test

    def feature_scaling(self):
        """
        Scale features
        """
        sc = self.scaler  # Initialize feature scaler
        self.X_train = sc.fit_transform(self.X_train)  # Scale train set
        self.X_test = sc.fit_transform(self.X_test)  # Scale test set

    def loc_inf(self):
        """
        Select self.n_features most informative features
        """
        info_features = self.feature_selection(self.X_train, self.Y_train)  # Initialize feature selection
        sorted_ind_features = np.argsort(info_features)[:: -1]  # Features indices sorted by informativeness
        indecies_im_f = sorted_ind_features[0: self.n_features]  # Select self.n_features most informative
        self.X_train = pd.DataFrame(self.X_train)
        self.X_train = self.X_train[indecies_im_f].to_numpy()  # Loc most informative features from X_train
        self.X_test = pd.DataFrame(self.X_test)
        self.X_test = self.X_test[indecies_im_f].to_numpy()  # Loc most informative features from X_test

    def sampl(self):
        """
        Up- or down-sample training dataset
        """
        sl = self.sampling
        self.X_train, self.Y_train = sl.fit_resample(np.array(self.X_train), self.Y_train.tolist())  # Up- or down-sample training set
        self.Y_train = pd.Series(self.Y_train)

    @staticmethod
    def auc_for_modAL(model, X_test, Y_test):
        """
        Calculate ROC-AUC lower bound, ROC-AUC, ROC-AUC upper bound for AL-model

        :param model: trained model
        :param X_test: test samples to predict labels
        :param Y_test: true labels
        :return: ROC-AUC lower bound, ROC-AUC, ROC-AUC upper bound
        """
        y_pred = model.predict_proba(X_test)
        return calc_auc_ci(Y_test, y_pred[:, 1])

    @staticmethod
    def f_one_mcc_score(model, X_test, Y_test):
        """
        Calculate F1-score and Matthews correlation coefficient (MCC)

        :param model:trained model
        :param X_test: test samples to predict labels
        :param Y_test: true labels
        :return: F1-score, MCC
        """

        y_pred = model.predict(X_test)
        f_one = f1_score(Y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
        return f_one, (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5


    def AL_strategy(self, X_train, X_test, Y_train, Y_test, k,
                    n_initial, n_queries,
                    cls=RandomForestClassifier(),
                    name='RandomForestClassifier',
                    q_strategy=uncertainty_sampling):
        """
        Subsample training dataset using AL strategies
        """

        initial_idx = np.random.choice(range(len(X_train)),
                                       size=n_initial, replace=False)
        if not isinstance(Y_train, np.ndarray):
            Y_train = Y_train.to_numpy()
        X = X_train[initial_idx]
        Y = Y_train[initial_idx]
        X_initial, y_initial = X_train[initial_idx], Y_train[initial_idx]
        X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), \
                         np.delete(Y_train, initial_idx, axis=0)

        learner = ActiveLearner(
            estimator=cls,
            query_strategy=q_strategy,
            X_training=X_initial, y_training=y_initial
        )
        AL_accuracy_scores = [learner.score(X_test, Y_test)]
        auc_d, (lb_d, ub_d) = self.auc_for_modAL(learner, X_test, Y_test)
        f_one, mcc = self.f_one_mcc_score(learner, X_test, Y_test)
        if not math.isnan(mcc):
            AL_mcc_scores = [mcc]
        else:
            AL_mcc_scores = [0]
        AL_auc_l_scores = [lb_d]
        AL_auc_scores = [auc_d]
        AL_auc_u_scores = [ub_d]
        AL_f_one_scores = [f_one]

        for i in range(n_queries):
            query_idx, query_inst = learner.query(X_pool)
            learner.teach(X_pool[query_idx], y_pool[query_idx])
            X = np.append(X, X_pool[query_idx], axis=0)
            Y = np.append(Y, y_pool[query_idx])
            X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), \
                             np.delete(y_pool, query_idx, axis=0)
            auc_d, (lb_d, ub_d) = self.auc_for_modAL(learner, X_test, Y_test)
            AL_auc_scores.append(auc_d)
            AL_auc_l_scores.append(lb_d)
            AL_auc_u_scores.append(ub_d)
            f_one, mcc = self.f_one_mcc_score(learner, X_test, Y_test)
            if not math.isnan(mcc):
                AL_mcc_scores.append(mcc)
            else:
                AL_mcc_scores.append(0)
            AL_f_one_scores.append(f_one)
            AL_accuracy_scores.append(learner.score(X_test, Y_test))
        max_mcc = max(AL_mcc_scores)
        ind_max_mcc = AL_mcc_scores.index(max_mcc)
        auc_l_m_mcc = AL_auc_l_scores[ind_max_mcc]
        auc_m_mcc = AL_auc_scores[ind_max_mcc]
        auc_u_m_mcc = AL_auc_u_scores[ind_max_mcc]
        accuracy_m_mcc = AL_accuracy_scores[ind_max_mcc]
        f_one_m_mcc = AL_f_one_scores[ind_max_mcc]
        performance_stats = [k, name, auc_l_m_mcc, auc_m_mcc, auc_u_m_mcc, accuracy_m_mcc, f_one_m_mcc, max_mcc]
        print('Test ROC AUC score for {} model with AL is {: .3}. Accuracy is {: .3}. MCC is {: .3}. F1 score is {: .3}'.format(name, auc_m_mcc,
                                                                                                                                accuracy_m_mcc,
                                                                                                                                max_mcc,
                                                                                                                                f_one_m_mcc))
        return performance_stats

    def fit_model_CV(self):
        """
        Run Cross Validation and calculate performance and metrics on test data
        """
        self.auc_stats = dict((key, [[], [], []]) for key in self.models.keys())
        performance_stats_n_AL = []
        performance_stats_AL = []
        for model_name, model_function in self.models.items():
            self.SCAMmer = model_function
            for i in range(self.cv_n):
                X_train, X_test, Y_train, Y_test = self.split_train_val(self.dataset,
                                                                        self.test_split_r)
                self.SCAMmer.fit(X_train, Y_train)
                test_predicted = self.SCAMmer.predict_proba(X_test)
                f_one, mcc = self.f_one_mcc_score(self.SCAMmer, X_test, Y_test)
                test_accuracy = accuracy_score(Y_test, self.SCAMmer.predict(X_test))
                auc_d, (lb_d, ub_d) = calc_auc_ci(Y_test, test_predicted[:, 1])
                performance_stats_n_AL.append([i, model_name, lb_d, auc_d, ub_d, test_accuracy, f_one, mcc])
                print('ROC AUC score for {} model {} is {: .3}. Accuracy is {: .3}. MCC is {: .3}. F1 score is {: .3}'.format(model_name, i, auc_d,
                                                                                                                              test_accuracy,
                                                                                                                              f_one, mcc))
                n_q = math.floor(self.dataset.shape[0]*(1-self.test_split_r) - self.initial)
                performance_stats_AL.append(self.AL_strategy(X_train, X_test, Y_train, Y_test, i,
                                                             self.initial, n_q,
                                                             cls=model_function, name=model_name))


            if self.non_AL_stats is None:
                self.non_AL_stats = pd.DataFrame(performance_stats_n_AL, columns=['Iteration', 'Method', 'AUC_LB', 'AUC',
                                                                             'AUC_UB', 'Accuracy', 'F1', 'MCC'])
            else:
                self.non_AL_stats = pd.concat([self.non_AL_stats,
                                            pd.DataFrame(performance_stats_n_AL,
                                                         columns=['Iteration', 'Method', 'AUC_LB', 'AUC',
                                                                  'AUC_UB', 'Accuracy', 'F1', 'MCC'])])
            self.non_AL_stats.to_csv('non_AL_stats.csv')

            if self.AL_stats is None:
                self.AL_stats = pd.DataFrame(performance_stats_AL, columns=['Iteration', 'Method', 'AUC_LB', 'AUC',
                                                                             'AUC_UB', 'Accuracy', 'F1', 'MCC'])
            else:
                self.AL_stats = pd.concat([self.AL_stats,
                                            pd.DataFrame(performance_stats_AL,
                                                         columns=['Iteration', 'Method', 'AUC_LB', 'AUC',
                                                                  'AUC_UB', 'Accuracy', 'F1', 'MCC'])])
            self.AL_stats.to_csv('AL_stats.csv')

    def calculate_t_test(self):
        """
        Calculate T-test for means of two independent samples
        """
        t_test_res = []  # List to save results
        for met in METRICS:  # For all calculates metrics
            met_mean_n_a = self.non_AL_stats[met].mean()  # Calculate mean for non AL
            met_mean_a = self.AL_stats[met].mean()  # Calculate mean for AL
            met_std_n_a = self.non_AL_stats[met].std() # Calculate std for non AL
            met_std_a = self.AL_stats[met].std()  # Calculate std for AL
            stat, p_value = ttest_ind_from_stats(mean1=met_mean_n_a, std1=met_std_n_a,
                                                 nobs1=self.non_AL_stats.shape[0],
                                                 mean2=met_mean_a, std2=met_std_a,
                                                 nobs2=self.AL_stats.shape[0]
                                                 )  # Calculate stat and p-value
            p_adj = p_value*len(METRICS)  # Bonferroni correction to multiple-testing
            t_test_res.append([met, met_mean_n_a, met_mean_a, stat, p_value, p_adj])

        self.t_test = pd.DataFrame(t_test_res, columns=['Metrics', 'Mean non AL', 'Mean AL',
                                                        't-test stat', 'p-value', 'p_adj'])  # Save results as a DataFrame
        print(self.t_test)
        self.t_test.to_csv('t-test_stats.csv')  # Save table with results





if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--path', required=True,
                    help='Path to directory with dataset')  # Specify patient name
    ap.add_argument('-f', '--file', required=True,
                    help='Dataset file name')  # Specify sample_type
    ap.add_argument('-a', '--activity_col', required=False,
                    help='Activity column name', default='agg?')
    ap.add_argument('-d', '--descriptor', required=False,  # Specify data type for plotting
                    help='Descriptor, MorganFingerprint or RDKFingerprint or MACCSkeys',
                    default='MorganFingerprint')
    ap.add_argument('-m', '--models', required=False,
                    help='Names of models to train separated by comma',
                    default='RandomForestClassifier')
    ap.add_argument('-fc', '--feature_selection', required=False,
                    help='Feature selection function, for example, mutual_info_classif',
                    default=mutual_info_classif)
    ap.add_argument('-sa', '--sampling', required=False,
                    help='Define feature sampling procedure', default=SMOTE())
    ap.add_argument('-sp', '--test_split_ratio', required=False,
                    default=0.3, type=float)

    args = ap.parse_args()
    models = {}
    for m in args.models.split(','):
        models[m] = MODELS[m]

    dataset_path = Path(args.path) / args.file
    if not dataset_path.is_file():
        raise FileNotFoundError('File {} not found in location {}. Please, inter valid path and file name'.format(args.path, args.file))

    dataset = pd.read_csv(dataset_path, index_col=0)

    ModelInstince = TrainModel(dataset=dataset, activity_colunm_name=args.activity_col,
                                descriptor=args.descriptor, models=models,
                                feature_selection=args.feature_selection,
                                test_split_r=args.test_split_ratio,
                                )
    ModelInstince.calculate_descriptors()
    ModelInstince.fit_model_CV()
    ModelInstince.calculate_t_test()
