from pathlib import Path
import argparse

import math
import pandas as pd
import numpy as np

import tqdm
import time

#  Importing packages for visualization
import plotly.graph_objects as go


#  Importing modules from sklearn
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, \
    f1_score, fbeta_score, pairwise_distances

from scipy.stats import ttest_ind
from scipy import stats
from scipy.signal import savgol_filter


#  Importing the modules from imbalanced learning
from imblearn.over_sampling import ADASYN, SMOTE  # up-sampling
from imblearn.under_sampling import CondensedNearestNeighbour  # down-sampling

#  Importing the modules for active learning
from modAL.models import ActiveLearner, Committee
from modAL.batch import uncertainty_batch_sampling
from modAL.uncertainty import uncertainty_sampling, classifier_uncertainty, classifier_entropy
from modAL.disagreement import vote_entropy_sampling
from modAL.batch import ranked_batch

# Importing modules to calculate confidence intervals and descriptors
from utilities import calc_auc_ci, butina_cluster, generate_scaffolds, _generate_scaffold
from utilities import DESCRIPTORS, MODELS, METRICS, SAMPLING, SELECTION_MODE, str2bool, rm_tree, file_doesnot_exist

# Importing cls models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import xgboost as xgb

#  Importing modules and packages for model tunning
import optuna
import optuna.integration.lightgbm as lgb
from hyperopt import tpe

#  Importing packages to enable processing of chemical structures
import rdkit
from rdkit import Chem

#  Importing RDLogger to filter out rdkit warnings
from rdkit import RDLogger

#  Importing package to filter out warnings
import warnings

# from collections import Counter
# from itertools import combinations
from mlxtend.evaluate import mcnemar_table, mcnemar



RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")


class TrainModel:
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

    # Added class variables
    N_BITS = 2048  # Number of bits
    M_R = 3  # Morgan Fingerprint's Radius
    N_L = 3  # Number of commitee learners
    S_L = 0.05  # p-value threshold
    P_R_MCC = 0.88  # The portion of data to reach the max MCC
    W_SG = 5
    H_SG = 3

    def __init__(self, data, validation_data, activity_colunm_name,
                 descriptor, models, test_split_r,
                 folder_name, selection,
                 scaler=StandardScaler(), sampling=SMOTE(),
                 n_features=300,
                 initial=10, run_butina=False,
                 run_scaf_split=False,
                 run_sampling=False,
                 committee=False,
                 batch_mode=True,
                 n_batch=3):

        self.dataset = data.copy()
        self.validation_data = validation_data.copy()
        self.activity_column_name = activity_colunm_name
        self.descriptor = {descriptor: DESCRIPTORS[descriptor]}
        self.models = models
        self.test_split_r = test_split_r
        self.folder_name = folder_name
        self.result_dir_path = Path.cwd() / 'Results' / self.folder_name
        self.scaler = scaler
        self.sampling = sampling
        self.n_features = n_features
        self.initial = initial
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.test_predicted = None
        self.non_AL_stats = None
        self.AL_stats = None
        self.cv_n = 10
        self.t_test = None
        self.run_butina = run_butina
        self.run_scaf_split = run_scaf_split
        self.run_sampling = run_sampling
        self.committee = committee
        self.max_mcc_data_percent = None
        self.batch_mode = batch_mode
        self.batch_n = n_batch
        self.final_cls = None
        self.SCAMsCls = None
        self.external_X = None
        self.external_Y = None
        self.external_val_res = None
        self.selection = selection

    def run(self):
        """
        Create directory to save results and run study
        """
        if Path.is_dir(self.result_dir_path):
            delete_existing_path = str2bool(input('Path exist. Do you want to delete the existing path and create the new? Please, enter yes or no: '))
            if delete_existing_path:
                rm_tree(self.result_dir_path)
            else:
                new_study = input('Please, enter new name: ')
                self.result_dir_path = Path.cwd() / 'Results' / new_study
        Path.mkdir(self.result_dir_path)
        time_zero = time.time()
        self.calculate_descriptors()
        time_descr = time.time()
        print('It took {} seconds to calculate descriptors'.format(int(time_descr - time_zero)))
        self.external_X = self.transform_X(self.validation_data[list(self.descriptor.keys())[0]])
        self.external_Y = self.transform_X(self.validation_data['DLS'])
        self.fit_model_CV()
        time_fit = time.time()
        print('It took {} seconds to fit models'.format(int(time_fit - time_descr)))
        self.calculate_t_test()
        # print(self.max_mcc_data_percent)
        self.make_radar_chart()


    @staticmethod
    def run_descriptors(dataset, descr_dictionary,
                       M_R, N_BITS):

        for descr_name, descr_func in descr_dictionary.items():

            dataset['mol_obj'] = dataset['Smiles String'].apply(
                lambda x: Chem.MolFromSmiles(x))
            if descr_name == 'MorganFingerprint':
                dataset['{}'.format(descr_name)] = dataset['mol_obj'].apply(lambda x: descr_func(x, M_R, nBits=N_BITS))  # Calculate Morgan descriptors
            else:
                dataset['{}'.format(descr_name)] = dataset['mol_obj'].apply(
                    lambda x: descr_func(x))  # Calculate other descriptors


    def calculate_descriptors(self):
        """
        Converts SMILES to mol_obj and calculate descriptors
        """
        self.run_descriptors(self.dataset, self.descriptor, M_R=self.M_R,
                             N_BITS=self.N_BITS)
        self.run_descriptors(self.validation_data, self.descriptor, M_R=self.M_R,
                             N_BITS=self.N_BITS)

    @staticmethod
    def transform_X(X):
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

        X_train = self.transform_X(
            X_train.iloc[:, 0])  # Select 1-st column with calculated descriptors and transform to numpy array
        X_test = self.transform_X(X_test.iloc[:, 0])
        Y_train = self.transform_X(Y_train)
        Y_test = self.transform_X(Y_test)

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

    @staticmethod
    def sampl(X_train, Y_train, sampling):
        """
        Up- or down-sample training dataset
        """
        sl = sampling
        X_train, Y_train = sl.fit_resample(np.array(X_train),
                                           Y_train.tolist())  # Up- or down-sample training set
        Y_train = pd.Series(Y_train)
        return X_train, Y_train

    @staticmethod
    def auc_for_modAL(model, X_test, Y_test, ):
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

    def make_committee(self, cls, X_initial, y_initial, n_learners=3):
        """
        Make a committee learner
        Parameters
        ----------
        :param cls: classifier model
        :param X_initial: np.array, initial features
        :param y_initial: np.array, initial labels
        :param n_learners: int, number of learners in the committee

        Returns
        -------
        committee

        """
        learner_list = []
        for _ in range(n_learners):
            learner = ActiveLearner(
                estimator=cls,
                X_training=X_initial, y_training=y_initial)
            learner_list.append(learner)

        # assembling the Committee
        committee = Committee(learner_list=learner_list,
                              query_strategy=vote_entropy_sampling)

        return committee

    @staticmethod
    def calculate_cls_balance(class_labels_np):
        """
        Calculate class balance for AL
        Parameters
        ----------
        class_labels_np: np.array with labels

        Returns
        -------
        class balance

        """
        cls, counts = np.unique(class_labels_np, return_counts=True)
        return counts[0] / counts[1]

    @staticmethod
    def make_plot(y, n, plot_title):
        """
        Make a plot

        Parameters
        ----------
        :param y: list, values on the ordinate
        :param n: int, number of points
        :param plot_title: str, plot title

        Returns
        -------
        plot
        """
        plot = go.Figure(go.Scatter(
            x=list(range(0, n)),
            y=y
        ))
        plot.update_layout(
            xaxis=dict(
                title='Iteration',
                tickmode='linear',
                tick0=0.5,
                dtick=0.75,
                showticklabels=False
            ),
            yaxis=dict(
                title=plot_title
            )
        )
        return plot

    @staticmethod
    def average_stat(stat_array, window, polyorder):
        """
        Calculate average performance

        Parameters
        ----------
        :param stat_array: np.array, AL run results

        Returns
        :param average: float, average value
        -------
        """
        stat_array = np.array(stat_array)
        stat_array[np.isnan(stat_array)] = 0
        savitzky_golay = savgol_filter(stat_array, window, polyorder)
        max_index = np.argmax(savitzky_golay)
        return savitzky_golay[max_index], max_index


    def AL_strategy(self, iteration, X_train, X_test, Y_train, Y_test,
                    n_initial, n_queries,
                    cls=RandomForestClassifier(),
                    name='RandomForestClassifier',
                    q_strategy=vote_entropy_sampling):
        """
        Subsample training dataset using AL strategies
        """
        class_balance = []  # list to save class balance

        def random_choise(X_train, n_initial):
            initial_idx = np.random.choice(range(len(X_train)),
                                           size=n_initial, replace=False)
            return initial_idx

        initial_idx = random_choise(X_train, n_initial)
        while len(set(Y_train[initial_idx])) != 2:  # Check if both classes are presented
            initial_idx = random_choise(X_train, n_initial)

        X, Y = X_train[initial_idx], Y_train[initial_idx]

        class_balance.append(self.calculate_cls_balance(Y))

        X_initial, y_initial = X_train[initial_idx], Y_train[initial_idx]
        X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), \
                         np.delete(Y_train, initial_idx, axis=0)

        if self.committee:
            learner = self.make_committee(cls, X_initial, y_initial, self.N_L)
        else:
            learner = ActiveLearner(
                estimator=cls,
                query_strategy=q_strategy,
                X_training=X_initial, y_training=y_initial
            )
        # Calculate initial performance metrics on the test set
        AL_accuracy_scores_test = [learner.score(X_test, Y_test)]
        auc_d_test, (lb_d_test, ub_d_test) = self.auc_for_modAL(learner, X_test, Y_test)
        AL_auc_l_scores_test = [lb_d_test]
        AL_auc_scores_test = [auc_d_test]
        AL_auc_u_scores_test = [ub_d_test]
        f_one_test, mcc_test = self.f_one_mcc_score(learner, X_test, Y_test)
        AL_f_one_scores_test = [f_one_test]
        if not math.isnan(mcc_test):
            AL_mcc_scores_test = [mcc_test]
        else:
            AL_mcc_scores_test = [0]

        # Calculate initial performance metrics on the external validation set
        AL_accuracy_scores_val = [learner.score(self.external_X, self.external_Y)]
        auc_d_val, (lb_d_val, ub_d_val) = self.auc_for_modAL(learner, self.external_X, self.external_Y)
        AL_auc_l_scores_val = [lb_d_val]
        AL_auc_scores_val = [auc_d_val]
        AL_auc_u_scores_val = [ub_d_val]
        f_one_val, mcc_val = self.f_one_mcc_score(learner, self.external_X, self.external_Y)
        AL_f_one_scores_val = [f_one_val]
        if not math.isnan(mcc_val):
            AL_mcc_scores_val = [mcc_val]
        else:
            AL_mcc_scores_val = [0]




        for i in range(int(n_queries / self.batch_n) - 1):
            if self.batch_mode:
                query_idx, query_inst = learner.query(X_pool, n_instances=self.batch_n)
            else:
                query_idx, query_inst = learner.query(X_pool)
            if self.committee:
                learner.teach(X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1, ))
            else:
                learner.teach(X_pool[query_idx], y_pool[query_idx])
            X = np.append(X, X_pool[query_idx], axis=0)
            Y = np.append(Y, y_pool[query_idx])
            class_balance.append(self.calculate_cls_balance(Y))
            X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)

            AL_accuracy_scores_test.append(learner.score(X_test, Y_test))
            auc_d_test, (lb_d_test, ub_d_test) = self.auc_for_modAL(learner, X_test, Y_test)
            AL_auc_scores_test.append(auc_d_test)
            AL_auc_l_scores_test.append(lb_d_test)
            AL_auc_u_scores_test.append(ub_d_test)
            f_one_test, mcc_test = self.f_one_mcc_score(learner, X_test, Y_test)
            AL_f_one_scores_test.append(f_one_test)
            if not math.isnan(mcc_test):
                AL_mcc_scores_test.append(mcc_test)
            else:
                AL_mcc_scores_test.append(0)

            AL_accuracy_scores_val.append(learner.score(self.external_X, self.external_Y))
            auc_d_val, (lb_d_val, ub_d_val) = self.auc_for_modAL(learner, self.external_X, self.external_Y)
            AL_auc_scores_val.append(auc_d_val)
            AL_auc_l_scores_val.append(lb_d_val)
            AL_auc_u_scores_val.append(ub_d_val)
            f_one_val, mcc_val = self.f_one_mcc_score(learner, self.external_X, self.external_Y)
            AL_f_one_scores_val.append(f_one_val)
            if not math.isnan(mcc_test):
                AL_mcc_scores_val.append(mcc_val)
            else:
                AL_mcc_scores_val.append(0)


        max_auc_l_test, _ = self.average_stat(AL_auc_l_scores_test, self.W_SG, self.H_SG)
        max_auc_m_test, _ = self.average_stat(AL_auc_scores_test, self.W_SG, self.H_SG)
        max_auc_u_test, _ = self.average_stat(AL_auc_u_scores_test, self.W_SG, self.H_SG)
        max_accuracy_test, _ = self.average_stat(AL_accuracy_scores_test, self.W_SG, self.H_SG)
        max_f_one_test, _ = self.average_stat(AL_f_one_scores_test, self.W_SG, self.H_SG)
        max_mcc_test, max_mcc_index = self.average_stat(AL_mcc_scores_test, self.W_SG, self.H_SG)
        performance_stats_test = [max_auc_l_test, max_auc_m_test, max_auc_u_test, max_accuracy_test, max_f_one_test, max_mcc_test]

        max_auc_l_val, _ = self.average_stat(AL_auc_l_scores_val, self.W_SG, self.H_SG)
        max_auc_m_val, _ = self.average_stat(AL_auc_scores_val, self.W_SG, self.H_SG)
        max_auc_u_val, _ = self.average_stat(AL_auc_u_scores_val, self.W_SG, self.H_SG)
        max_accuracy_val, _ = self.average_stat(AL_accuracy_scores_val, self.W_SG, self.H_SG)
        max_f_one_val, _ = self.average_stat(AL_f_one_scores_val, self.W_SG, self.H_SG)
        max_mcc_val, max_mcc_index = self.average_stat(AL_mcc_scores_val, self.W_SG, self.H_SG)
        performance_stats_val = [max_auc_l_val, max_auc_m_val, max_auc_u_val, max_accuracy_val, max_f_one_val, max_mcc_val]


        mcc_plot_test = self.make_plot(AL_mcc_scores_test, n_queries,
                                  'MCC plot, test')

        mcc_plot_val = self.make_plot(AL_mcc_scores_val, n_queries,
                                      'MCC plot, validation')
        class_balance_plot = self.make_plot(class_balance, n_queries,
                                            'Class balance')

        mcc_plot_test_path = self.result_dir_path / 'mcc_plot_iteration_test_{}.svg'.format(iteration)
        mcc_plot_test.write_image(str(mcc_plot_test_path))

        mcc_plot_val_path = self.result_dir_path / 'mcc_plot_iteration_val_{}.svg'.format(iteration)
        mcc_plot_val.write_image(str(mcc_plot_val_path))

        class_balance_plot_path = self.result_dir_path / 'class_balance_plot_iteration_{}.svg'.format(iteration)
        class_balance_plot.write_image(str(class_balance_plot_path))



        # max_mcc_index = np.argmax(AL_mcc_scores)
        final_X_train, final_Y_train = X[0: max_mcc_index * self.batch_n + n_initial, ], Y[0: max_mcc_index * self.batch_n + n_initial, ]
        if self.max_mcc_data_percent is None:
            self.max_mcc_data_percent = []

        self.max_mcc_data_percent.append((final_X_train.shape[0] / X_train.shape[0]) * 100)

        self.final_cls = cls
        self.final_cls.fit(final_X_train, final_Y_train)
        f_one_final, mcc_final = self.f_one_mcc_score(self.final_cls, X_test, Y_test)
        predicted_labels = self.final_cls.predict(X_test)
        f1_ext, mcc_ext = self.f_one_mcc_score(self.final_cls, self.external_X, self.external_Y)
        performance_stats_external = [f1_ext, mcc_ext]
        print('AL, DeepScams ds: ', *performance_stats_val)
        return performance_stats_test, predicted_labels, performance_stats_val

    @staticmethod
    def calculate_iter_AL(data, spit_ratio,
                          n_initial):
        """
        Calculate maximun number of
        queries for AL model
        :param data:
        :param spit_ratio:
        :param n_initial:
        :return:
        """
        n_q = math.floor(data.shape[0] * (1 - spit_ratio) - n_initial)
        return n_q

    @staticmethod
    def make_df_with_stats(df_with_stats, performance_stats):
        """
        Make dataframe with performance stats

        Parameters
        ----------
        df_with_stats: A dataframe or None object with performance stats over passed iterations
        performance_stats: iteration stats


        """
        if df_with_stats is None:
            df_with_stats = pd.DataFrame(performance_stats,
                                         columns=['Iteration', 'Method', 'AUC_LB_test', 'AUC_test',
                                                  'AUC_UB_test', 'Accuracy_test', 'F1_test', 'MCC_test',
                                                  'AUC_LB_validation', 'AUC_validation',
                                                  'AUC_UB_validation', 'Accuracy_validation', 'F1_validation', 'MCC_validation'])
        else:
            df_with_stats = pd.concat([df_with_stats,
                                       pd.DataFrame(performance_stats,
                                                    columns=['Iteration', 'Method', 'AUC_LB_test', 'AUC_test',
                                                             'AUC_UB_test', 'Accuracy_test', 'F1_test', 'MCC_test',
                                                             'AUC_LB_validation', 'AUC_validation',
                                                             'AUC_UB_validation', 'Accuracy_validation',
                                                             'F1_validation', 'MCC_validation']
                                                    )])
        # F1_external', 'MCC_external'
        return df_with_stats

    def split_with_butina(self, df_with_mol_obj, split_r, x_column_name='MorganFingerprint',
                          y_column_name='agg?', id_name='Cdiv ID'):
        """
        Split dataset using butina clustering algorithm

        Parameters
        ----------
        df_with_mol_obj: pd.DataFrame with ids, mol_objects, calculated descriptors and class labels
        split_r: int, train/test split ratio
        id_name: str, id column name
        x_column_name: str, descriptor column name
        y_column_name str, class column name

        Returns
        _______
        X_train: np.array with train data
        X_test: np.array with test data
        Y_train: np.array with train class labels
        Y_test: np.array with test class labels

        """
        n_samples_test = int(df_with_mol_obj.shape[0] * split_r)
        df_with_mol_obj['cluster'] = butina_cluster(df_with_mol_obj['mol_obj'])
        uniq_cluster_ids = df_with_mol_obj['cluster'].value_counts().loc[lambda x: x == 1].index.tolist()
        if len(uniq_cluster_ids) < int(df_with_mol_obj.shape[0] * split_r):
            print('Unable to split dataset based on butina clustering')
        test_set = df_with_mol_obj[df_with_mol_obj.cluster.isin(uniq_cluster_ids)].sample(n_samples_test)
        train_set = df_with_mol_obj[~df_with_mol_obj[id_name].isin(test_set[id_name].tolist())]  # Remove hard coded label later
        X_test, Y_test = self.transform_X(test_set[x_column_name]), \
                         self.transform_X(test_set[y_column_name])
        X_train, Y_train = self.transform_X(train_set[x_column_name]), \
                           self.transform_X(train_set[y_column_name])
        return X_train, X_test, Y_train, Y_test

    def split_with_scaffold_splitter(self, scams_df, split_r, x_column_name='MorganFingerprint',
                                     y_column_name='agg?'):
        """
        Split dataset using scaffold splitter
        """
        n_samples_test = int(scams_df.shape[0] * split_r)
        scaffold_sets = generate_scaffolds(scams_df)
        train_cutoff = (1 - self.test_split_r) * scams_df.shape[0]
        train_inds = []
        test_inds = []

        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                test_inds += scaffold_set
            else:
                train_inds += scaffold_set
        X_train = self.transform_X(scams_df.iloc[train_inds][x_column_name])
        X_test = self.transform_X(scams_df.iloc[test_inds][x_column_name])
        Y_train = self.transform_X(scams_df.iloc[train_inds][y_column_name])
        Y_test = self.transform_X(scams_df.iloc[test_inds][y_column_name])
        return X_train, X_test, Y_train, Y_test

    def non_AL_strategy(self, model_name, model_function,
                        X_train, Y_train, X_test, Y_test):
        """
        Run non AL strategy
        """

        self.SCAMsCls = model_function
        if self.run_sampling:
            X_train, Y_train = self.sampl(X_train, Y_train, self.sampling)
        self.SCAMsCls.fit(X_train, Y_train)
        test_predicted = self.SCAMsCls.predict_proba(X_test)
        f_one_test, mcc_test = self.f_one_mcc_score(self.SCAMsCls, X_test, Y_test)
        test_accuracy = accuracy_score(Y_test, self.SCAMsCls.predict(X_test))
        auc_d_test, (lb_d_test, ub_d_test) = calc_auc_ci(Y_test, test_predicted[:, 1])
        performance_stats_test = [lb_d_test, auc_d_test, ub_d_test, test_accuracy, f_one_test, mcc_test]
        predicted_labels = self.SCAMsCls.predict(X_test)


        val_predicted = self.SCAMsCls.predict_proba(self.external_X)
        f_one_val, mcc_val = self.f_one_mcc_score(self.SCAMsCls, self.external_X, self.external_Y)
        auc_d_val, (lb_d_val, ub_d_val) = calc_auc_ci(self.external_Y, val_predicted[:, 1])
        val_accuracy = accuracy_score(self.external_Y, self.SCAMsCls.predict(self.external_X))
        performance_stats_external = [lb_d_val, auc_d_val, ub_d_val, val_accuracy, f_one_val, mcc_val]
        print('non AL, DeepScams ds: ', *performance_stats_external)
        return performance_stats_test, predicted_labels, performance_stats_external

    def fit_model_CV(self):
        """
        Run Cross Validation and calculate performance and metrics on test data
        """
        performance_stats_n_AL = []
        performance_stats_AL = []
        # performance_stats_ext_AL = []
        # performance_stats_ext_non_AL = []
        mc_nemar_stats = []
        for model_name, model_function in self.models.items():
            for i in tqdm.tqdm(range(self.cv_n)):
                if self.run_butina:
                    X_train, X_test, Y_train, Y_test = self.split_with_butina(self.dataset, self.test_split_r)
                if self.run_scaf_split:
                    X_train, X_test, Y_train, Y_test = self.split_with_scaffold_splitter(self.dataset,
                                                                                         self.test_split_r)
                else:
                    X_train, X_test, Y_train, Y_test = self.split_train_val(self.dataset, self.test_split_r)
                # Run non-AL model and save the results
                [lb_d_n_al_t, auc_d_n_al_t, ub_d_n_al_t, accuracy_n_al_t, f_one_n_al_t,
                 mcc_n_al_t], y_non_AL_model, \
                [lb_d_n_al_v, auc_d_n_al_v, ub_d_n_al_v, accuracy_n_al_v, f1_non_AL_v, mcc_non_AL_v] = self.non_AL_strategy(model_name,
                                                                  model_function,
                                                                  X_train, Y_train,
                                                                  X_test, Y_test)

                performance_stats_n_AL.append(
                    [i, model_name, lb_d_n_al_t, auc_d_n_al_t, ub_d_n_al_t, accuracy_n_al_t, f_one_n_al_t, mcc_n_al_t,
                     lb_d_n_al_v, auc_d_n_al_v, ub_d_n_al_v, accuracy_n_al_v, f1_non_AL_v, mcc_non_AL_v]) #  f1_ext_non_AL, mcc_ext_non_AL

                # Run AL model and save the results
                n_q = int(self.P_R_MCC * self.calculate_iter_AL(self.dataset, self.test_split_r, self.initial))
                [lb_d_al_t, auc_d_al_t, ub_d_al_t, accuracy_al_t, f_one_al_t, mcc_al_t], y_AL_model, \
                [lb_d_al_v, auc_d_al_v, ub_d_al_v, accuracy_al_v, f1_ext_AL_v, mcc_ext_AL_v] = self.AL_strategy(i, X_train, X_test, Y_train, Y_test,
                                                               self.initial, n_q, cls=model_function,
                                                               name=model_name, q_strategy=self.selection)
                performance_stats_AL.append(
                    [i, model_name, lb_d_al_t, auc_d_al_t, ub_d_al_t, accuracy_al_t, f_one_al_t, mcc_al_t,
                     lb_d_al_v, auc_d_al_v, ub_d_al_v, accuracy_al_v, f1_ext_AL_v, mcc_ext_AL_v]) # f1_ext_AL, mcc_ext_AL

                cont_tb = mcnemar_table(y_target=Y_test,
                                        y_model1=y_non_AL_model,
                                        y_model2=y_AL_model)
                chi2, p = mcnemar(ary=cont_tb, exact=True)
                mc_nemar_stats.append([model_name, i, chi2, p])

            self.non_AL_stats = self.make_df_with_stats(self.non_AL_stats, performance_stats_n_AL)

            self.AL_stats = self.make_df_with_stats(self.AL_stats, performance_stats_AL)

        mc_nemar_stats = pd.DataFrame(mc_nemar_stats, columns=['Model name', 'Iteration',
                                                               'chi-squared', 'p-value'])
        self.non_AL_stats.to_csv(self.result_dir_path / 'non_AL_stats.csv')
        self.AL_stats.to_csv(self.result_dir_path / 'AL_stats.csv')
        mc_nemar_stats.to_csv(self.result_dir_path / 'mc_nemar_stats.csv')

    def calculate_t_test(self):
        """
        Calculate T-test for means of two independent samples
        """
        t_test_res = []  # List to save results
        for met in METRICS:  # For all calculates metrics
            met_mean_n_a = self.non_AL_stats[met].mean()  # Calculate mean for non AL
            met_mean_a = self.AL_stats[met].mean()  # Calculate mean for AL
            stat, p_value = ttest_ind(self.non_AL_stats[met], self.AL_stats[met])  # Calculate stat and p-value
            p_adj = p_value * len(METRICS)  # Bonferroni correction to multiple-testing
            sig = p_adj < self.S_L
            t_test_res.append([met, met_mean_n_a, met_mean_a, stat, p_value, p_adj, sig])

        self.t_test = pd.DataFrame(t_test_res, columns=['Metrics', 'Mean non AL', 'Mean AL',
                                                        't-test stat', 'p-value', 'p_adj',
                                                        'is_significant'])  # Save results as a DataFrame
        print(self.t_test)
        self.t_test.to_csv(self.result_dir_path / 't-test_stats.csv')  # Save table with results

    def make_radar_chart(self):
        """
        Make a radar chart with performance of AL and non-AL models
        """
        AL = self.t_test['Mean AL']
        non_AL = self.t_test['Mean non AL']
        # Make adaptive max score
        max_perf = max(max(AL), max(non_AL))
        if max_perf + max_perf*0.1 > 1:
            max_perf = 1
        else:
            max_perf = max_perf + max_perf * 0.1


        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=AL,
            theta=METRICS,
            fill='toself',
            name='AL strategy'
        ))
        radar.add_trace(go.Scatterpolar(
            r=non_AL,
            theta=METRICS,
            fill='toself',
            name='Non-AL strategy'
        ))
        radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_perf]
                )),
            showlegend=True
        )
        radar_plot_path = self.result_dir_path / 'AL_non_AL_performance.svg'
        radar.write_image(str(radar_plot_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('-p', '--path', required=True,
                    help='Path to directory with dataset')  # Specify path
    ap.add_argument('-f', '--file', required=True,
                    help='Dataset file name')  # Specify dataset file name
    ap.add_argument('-sn', '--study_name', required=True, type=str,
                    help='Study name')
    ap.add_argument('-e_v', '--external_validation_dataset', default='test_DLS.txt',
                    help='Dataset for external validation')

    ap.add_argument('-sp', '--test_split_ratio', required=False,
                    default=0.3, type=float)

    ap.add_argument('-a', '--activity_col', required=False,
                    help='Activity column name', default='agg?')
    ap.add_argument('-d', '--descriptor', required=False,  # Specify descriptors
                    help='Descriptor, MorganFingerprint or RDKFingerprint or MACCSkeys',
                    default='MorganFingerprint')

    ap.add_argument('-m', '--models', required=False, nargs='+',
                    help='Names of models to train separated by comma',
                    default=['RandomForestClassifier'])

    ap.add_argument('-rs', '--run_sampling', required=False, default=False,
                    help='Run up- or downsampling', type=bool)
    ap.add_argument('-sa', '--sampling', required=False,
                    help='Define feature sampling procedure', default='SMOTE')


    ap.add_argument('-b', '--butina', default=False,
                    help='Run butina algorithm, False or True', type=bool)
    ap.add_argument('-ss', '--scaf_split', default=False,
                    help='Run scaffold split, False or True', type=bool)

    ap.add_argument('-c', '--committee', default=False,
                    help='Make committee learner, False or True', type=bool)
    ap.add_argument('-sl_m', '--selection_mode', required=False, default='uncertainty_sampling',
                    help='Selection mode', type=str)
    ap.add_argument('-b_n', '--n_batch', required=False, default=3,
                    help='Number of samples in the bath', type=int)

    args = ap.parse_args()
    models = {}

    for m in args.models:
        models[m] = MODELS[m]
        print(MODELS[m])

    selection = SELECTION_MODE[args.selection_mode]

    if args.selection_mode == 'uncertainty_batch_sampling':
        n_batch = args.n_batch
        batch_mode = True
    else:
        n_batch = 1
        batch_mode = False

    sampling_u = SAMPLING[args.sampling]

    dataset_path = file_doesnot_exist(args.path, args.file)

    dataset = pd.read_csv(dataset_path, index_col=0)

    ext_val_dataset_path = file_doesnot_exist(args.path, args.external_validation_dataset)
    ext_val_dataset = pd.read_csv(ext_val_dataset_path, sep='\t')


    ModelInstance = TrainModel(data=dataset, validation_data=ext_val_dataset,
                               activity_colunm_name=args.activity_col,
                               descriptor=args.descriptor, models=models,
                               test_split_r=args.test_split_ratio,
                               folder_name=args.study_name,
                               sampling=sampling_u,
                               run_butina=args.butina,
                               run_scaf_split=args.scaf_split,
                               run_sampling=args.run_sampling,
                               committee=args.committee,
                               batch_mode=batch_mode,
                               selection=selection,
                               n_batch=n_batch)
    ModelInstance.run()
