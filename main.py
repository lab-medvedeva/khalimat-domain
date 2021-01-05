from pathlib import Path
import argparse
import math

import pandas as pd
import numpy as np

#  Importing packages for visualization
# import seaborn as sns
# import matplotlib
# import matplotlib.pyplot as plt
# from IPython import display
import plotly.graph_objects as go


#  Importing modules from sklearn
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, \
    f1_score, fbeta_score, pairwise_distances

from scipy.stats import ttest_ind
from scipy import stats

#  Importing module for imbalanced learning
from imblearn.over_sampling import ADASYN, SMOTE  # up-sampling
from imblearn.under_sampling import CondensedNearestNeighbour  # down-sampling

#  Importing module for active learning
from modAL.models import ActiveLearner, Committee
from modAL.batch import uncertainty_batch_sampling
from modAL.uncertainty import uncertainty_sampling, classifier_uncertainty
from modAL.disagreement import vote_entropy_sampling
from modAL.batch import ranked_batch

# Importing modules to calculate confidence intervals and descriptors
from utilities import calc_auc_ci, butina_cluster, generate_scaffolds, _generate_scaffold
from utilities import DESCRIPTORS, MODELS, METRICS, SAMPLING, BATCH_MODE
# Importing modules to calculate confidence intervals and descriptors
from utilities import calc_auc_ci, butina_cluster, generate_scaffolds, _generate_scaffold
from utilities import DESCRIPTORS, MODELS, METRICS, SAMPLING

# Importing lightgbm to train classifier
from lightgbm import LGBMClassifier
import optuna.integration.lightgbm as lgb
from xgboost import XGBClassifier

import xgboost as xgb

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

#  Importing modules and packages for model tunning
import optuna
from hyperopt import tpe
from scipy import stats
import time

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
    P_R_MCC = 0.88 # The portion of data to reach the max MCC

    def __init__(self, data, activity_colunm_name,
                 descriptor, models, test_split_r,
                 scaler=StandardScaler(),
                 sampling=SMOTE(), n_features=300,
                 initial=10, run_butina=False,
                 run_scaf_split=False,
                 run_sampling=False,
                 committee=False,
                 batch_mode=True,
                 n_batch=3):

        self.dataset = data.copy()
        self.activity_column_name = activity_colunm_name
        self.descriptor = {descriptor: DESCRIPTORS[descriptor]}
        self.models = models
        self.test_split_r = test_split_r
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
        self.class_balance = None
        print(self.batch_mode)

    def run(self):
        time_zero = time.time()
        self.calculate_descriptors()
        time_descr = time.time()
        print('It took {} seconds to calculate descriptors'.format(int(time_descr - time_zero)))
        self.fit_model_CV()
        time_fit = time.time()
        print('It took {} seconds to fit models'.format(int(time_fit - time_descr)))
        self.calculate_t_test()
        # print(self.max_mcc_data_percent)
        self.make_radar_chart()

    def calculate_descriptors(self):
        """
        Converts SMILES to mol_obj and calculate descriptors
        """
        for d_nm, d_fn in self.descriptor.items():
            self.dataset['mol_obj'] = self.dataset['Smiles String'].apply(
                lambda x: Chem.MolFromSmiles(x))  # Make mol_obj from SMILES
            if d_nm == 'MorganFingerprint':
                self.dataset['{}'.format(d_nm)] = self.dataset['mol_obj'].apply(lambda x: d_fn(x, self.M_R,
                                                                                               nBits=self.N_BITS))  # Calculate Morgan descriptors
            else:
                self.dataset['{}'.format(d_nm)] = self.dataset['mol_obj'].apply(
                    lambda x: d_fn(x))  # Calculate other descriptors

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
        Select self.n_features most informati   ve features
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
        cls, counts = np.unique(class_labels_np, return_counts=True)
        return counts[0] / counts[1]

    def AL_strategy(self, X_train, X_test, Y_train, Y_test,
                    n_initial, n_queries,
                    cls=RandomForestClassifier(),
                    name='RandomForestClassifier',
                    q_strategy=uncertainty_batch_sampling):
        """
        Subsample training dataset using AL strategies
        """
        if self.class_balance is None:
            self.class_balance = []

        def random_choise(X_train, n_initial):
            initial_idx = np.random.choice(range(len(X_train)),
                                           size=n_initial, replace=False)
            return initial_idx

        initial_idx = random_choise(X_train, n_initial)
        while len(set(Y_train[initial_idx])) != 2:  # Check if both classes are presented
            initial_idx = random_choise(X_train, n_initial)

        X, Y = X_train[initial_idx], Y_train[initial_idx]

        self.class_balance.append(self.calculate_cls_balance(Y))

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
            self.class_balance.append(self.calculate_cls_balance(Y))
            X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
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

        max_auc_l = max(AL_auc_l_scores)
        max_auc_m = max(AL_auc_scores)
        max_auc_u = max(AL_auc_u_scores)
        max_accuracy = max(AL_accuracy_scores)
        max_f_one = max(AL_f_one_scores)
        max_mcc = max(AL_mcc_scores)
        performance_stats = [max_auc_l, max_auc_m, max_auc_u, max_accuracy, max_f_one, max_mcc]
        max_mcc_index = np.argmax(AL_mcc_scores)
        final_X_train, final_Y_train = X[0: max_mcc_index * self.batch_n + n_initial, ], Y[0: max_mcc_index * self.batch_n + n_initial, ]
        final_X_train, final_Y_train = X[0: max_mcc_index + n_initial, ], Y[0: max_mcc_index + n_initial, ]
        if self.max_mcc_data_percent is None:
            self.max_mcc_data_percent = []
        self.max_mcc_data_percent.append((final_X_train.shape[0] / X_train.shape[0]) * 100)

        # print(final_X_train.shape[0])
        final_cls = cls
        final_cls.fit(final_X_train, final_Y_train)
        predicted_labels = final_cls.predict(X_test)

        return performance_stats, predicted_labels

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
        if df_with_stats is None:
            df_with_stats = pd.DataFrame(performance_stats,
                                         columns=['Iteration', 'Method', 'AUC_LB', 'AUC',
                                                  'AUC_UB', 'Accuracy', 'F1', 'MCC'])
        else:
            df_with_stats = pd.concat([df_with_stats,
                                       pd.DataFrame(performance_stats,
                                                    columns=['Iteration', 'Method', 'AUC_LB', 'AUC',
                                                             'AUC_UB', 'Accuracy', 'F1', 'MCC'])])
        return df_with_stats

    def split_with_butina(self, df, split_r, x_column_name='MorganFingerprint',
                          y_column_name='agg?'):
        n_samples_test = int(df.shape[0] * split_r)
        df['cluster'] = butina_cluster(df['mol_obj'])
        uniq_cluster_ids = df['cluster'].value_counts().loc[lambda x: x == 1].index.tolist()
        if len(uniq_cluster_ids) < int(df.shape[0] * split_r):
            print('Unable to split dataset based on butina clustering')
        test_set = df[df.cluster.isin(uniq_cluster_ids)].sample(n_samples_test)
        train_set = df[~df['Cdiv ID'].isin(test_set['Cdiv ID'].tolist())]  # Remove hard coded label later
        X_test, Y_test = self.transform_X(test_set[x_column_name]), \
                         self.transform_X(test_set[y_column_name])
        X_train, Y_train = self.transform_X(train_set[x_column_name]), \
                           self.transform_X(train_set[y_column_name])
        return X_train, X_test, Y_train, Y_test

    def split_with_scaffold_splitter(self, scams_df, split_r, x_column_name='MorganFingerprint',
                                     y_column_name='agg?'):
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

        SCAMsCls = model_function
        if self.run_sampling:
            X_train, Y_train = self.sampl(X_train, Y_train, self.sampling)
        SCAMsCls.fit(X_train, Y_train)
        test_predicted = SCAMsCls.predict_proba(X_test)
        predicted_labels = SCAMsCls.predict(X_test)
        f_one, mcc = self.f_one_mcc_score(SCAMsCls, X_test, Y_test)
        test_accuracy = accuracy_score(Y_test, SCAMsCls.predict(X_test))
        auc_d, (lb_d, ub_d) = calc_auc_ci(Y_test, test_predicted[:, 1])

        # print('AUC score for {} model is {: .3}. Accuracy is {: .3}. MCC is {: .3}. F1 score is {: .3}'.format(
        #     model_name, auc_d, test_accuracy, mcc, f_one))
        performance_stats = [lb_d, auc_d, ub_d, test_accuracy, f_one, mcc]

        return performance_stats, predicted_labels

    def fit_model_CV(self):
        """
        Run Cross Validation and calculate performance and metrics on test data
        """
        performance_stats_n_AL = []
        performance_stats_AL = []
        mc_nemar_stats = []
        for model_name, model_function in self.models.items():
            for i in range(self.cv_n):
                if self.run_butina:
                    X_train, X_test, Y_train, Y_test = self.split_with_butina(self.dataset, self.test_split_r)
                if self.run_scaf_split:
                    X_train, X_test, Y_train, Y_test = self.split_with_scaffold_splitter(self.dataset,
                                                                                         self.test_split_r)
                else:
                    X_train, X_test, Y_train, Y_test = self.split_train_val(self.dataset, self.test_split_r)
                # Run non-AL model and save the results
                [lb_d_n_al, auc_d_n_al, ub_d_n_al, accuracy_n_al, f_one_n_al,
                 mcc_n_al], y_non_AL_model = self.non_AL_strategy(model_name,
                                                                  model_function,
                                                                  X_train, Y_train,
                                                                  X_test, Y_test)

                performance_stats_n_AL.append(
                    [i, model_name, lb_d_n_al, auc_d_n_al, ub_d_n_al, accuracy_n_al, f_one_n_al, mcc_n_al])

                # Run AL model and save the results
                n_q = int(self.P_R_MCC * self.calculate_iter_AL(self.dataset, self.test_split_r, self.initial))
                [lb_d_al, auc_d_al, ub_d_al, accuracy_al, f_one_al, mcc_al], y_AL_model = self.AL_strategy(X_train,
                                                                                                           X_test,
                                                                                                           Y_train,
                                                                                                           Y_test,
                                                                                                           self.initial,
                                                                                                           n_q,
                                                                                                           cls=model_function,
                                                                                                           name=model_name,
                                                                                                           q_strategy=
                                                                                                           BATCH_MODE[
                                                                                                               self.batch_mode])
                performance_stats_AL.append(
                    [i, model_name, lb_d_al, auc_d_al, ub_d_al, accuracy_al, f_one_al, mcc_al])

                cont_tb = mcnemar_table(y_target=Y_test,
                                        y_model1=y_non_AL_model,
                                        y_model2=y_AL_model)
                chi2, p = mcnemar(ary=cont_tb, exact=True)
                mc_nemar_stats.append([model_name, i, chi2, p])

            self.non_AL_stats = self.make_df_with_stats(self.non_AL_stats, performance_stats_n_AL)

            self.AL_stats = self.make_df_with_stats(self.AL_stats, performance_stats_AL)

        mc_nemar_stats = pd.DataFrame(mc_nemar_stats, columns=['Model name', 'Iteration',
                                                               'chi-squared', 'p-value'])
        self.non_AL_stats.to_csv('non_AL_stats.csv')
        self.AL_stats.to_csv('AL_stats.csv')
        mc_nemar_stats.to_csv('mc_nemar_stats.csv')

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
        self.t_test.to_csv('t-test_stats.csv')  # Save table with results

    def make_radar_chart(self):
        AL = self.t_test['Mean AL']
        non_AL = self.t_test['Mean non AL']

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
                    range=[0, 1]
                )),
            showlegend=True
        )
        radar.write_image("AL_non_AL_performance.svg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--path', required=True,
                    help='Path to directory with dataset')  # Specify path
    ap.add_argument('-f', '--file', required=True,
                    help='Dataset file name')  # Specify dataset file name
    ap.add_argument('-a', '--activity_col', required=False,
                    help='Activity column name', default='agg?')
    ap.add_argument('-d', '--descriptor', required=False,  # Specify descriptors
                    help='Descriptor, MorganFingerprint or RDKFingerprint or MACCSkeys',
                    default='MorganFingerprint')
    ap.add_argument('-m', '--models', required=False, nargs='+',
                    help='Names of models to train separated by comma',
                    default=['RandomForestClassifier'])
    ap.add_argument('-sa', '--sampling', required=False,
                    help='Define feature sampling procedure', default='SMOTE')
    ap.add_argument('-sp', '--test_split_ratio', required=False,
                    default=0.3, type=float)
    ap.add_argument('-b', '--butina', default=False,
                    help='Run butina algorithm, False or True', type=bool)
    ap.add_argument('-rs', '--run_sampling', required=False, default=False,
                    help='Run up- or downsampling', type=bool)
    ap.add_argument('-ss', '--scaf_split', default=False,
                    help='Run scaffold split, False or True', type=bool)
    ap.add_argument('-c', '--committee', default=False,
                    help='Make committee learner, False or True', type=bool)
    ap.add_argument('-b_m', '--batch_mode', required=False, default=True,
                    help='Run batch selection, False or True', type=bool)
    ap.add_argument('-b_n', '--n_batch', required=False, default=3,
                    help='Number of samples in the bath', type=int)
    args = ap.parse_args()
    models = {}
    for m in args.models:
        models[m] = MODELS[m]
    if args.batch_mode:
        n_batch = args.n_batch
    else:
        n_batch = 1

    sampling_u = SAMPLING[args.sampling]

    dataset_path = Path(args.path) / args.file
    if not dataset_path.is_file():
        raise FileNotFoundError(
            'File {} not found in location {}. Please, inter valid path and file name'.format(args.file, args.path))

    dataset = pd.read_csv(dataset_path, index_col=0)

    ModelInstance = TrainModel(data=dataset, activity_colunm_name=args.activity_col,
                               descriptor=args.descriptor, models=models,
                               test_split_r=args.test_split_ratio,
                               sampling=sampling_u,
                               run_butina=args.butina,
                               run_scaf_split=args.scaf_split,
                               run_sampling=args.run_sampling,
                               committee=args.committee,
                               batch_mode=args.batch_mode,
                               n_batch=n_batch)
    ModelInstance.run()
