import os
import warnings
import statistics
from pathlib import Path

import numpy as np
# from matplotlib import pyplot
import matplotlib.pyplot as plt

from sklearn import impute
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
from sklearn import dummy
from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
from sklearn import neighbors
from sklearn import linear_model
from sklearn import neural_network

import pandas as pd
from tqdm import tqdm

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

tqdm.pandas()

pd.options.display.width = None
pd.set_option('display.max_columns', None)


def folder_nb_2_position(n):
    if n == 0:
        return 0, 0
    elif n == 1:
        return 0, 1
    elif n == 2:
        return 0, 2
    elif n == 3:
        return 1, 0
    elif n == 4:
        return 1, 1

rs = 42
df = pd.read_csv('datasets/training_dataset_featured.csv')

# print(df.columns)
# print('Nb churn:', len(df[df['churn'] == 1]))
# print('Nb no churn:', len(df[df['churn'] == 0]))

# - Get the number of NULL values per column
# print(df.isnull().sum())

df.drop(columns=[
    'polkey', 'f_gender_is_man'
], inplace=True)

X = df.drop(columns=['churn'])
y = df['churn']

NB_FOLDS = 5
skf = model_selection.StratifiedKFold(n_splits=NB_FOLDS, random_state=rs, shuffle=True)

classifiers = {
    'dummy': [
        {
            'classifier': dummy.DummyClassifier(strategy='stratified', random_state=rs),
            'folder': 'stratified'
        },
        {
            'classifier': dummy.DummyClassifier(strategy='most_frequent', random_state=rs),
            'folder': 'most_frequent'
        }
    ]
}

PATH_TO_RESULTS = 'results/classification'
# for folder_name, clf_instances in tqdm(classifiers.items(), total=len(classifiers), desc='Classifiers', position=0, leave=False, ncols=80):
for folder_name, clf_instances in classifiers.items():

    print(f'[Classifier] - {folder_name}')
    destination_folder = os.path.join(PATH_TO_RESULTS, folder_name)
    Path(destination_folder).mkdir(parents=True, exist_ok=True)

    # for clf_instance in tqdm(clf_instances, total=len(clf_instances), desc=folder_name, position=1, leave=False, ncols=80):
    for clf_instance in clf_instances:
        clf = clf_instance['classifier']
        subfolder = clf_instance['folder']

        print(f'>>> {subfolder}')

        destination_subfolder = os.path.join(destination_folder, subfolder)
        Path(destination_subfolder).mkdir(parents=True, exist_ok=True)

        overall_metrics = {
            'confusion_matrix': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'auc': []
        }
        fig_roc, axs_roc = plt.subplots(2, 3)
        fig_prc, axs_prc = plt.subplots(2, 3)
        fold_nb = 0
        for train_index, test_index in tqdm(skf.split(X, y), total=NB_FOLDS):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #############################################################################
            # - Impute missing values // TRAIN
            #############################################################################
            # - Customer Age
            customer_age_group_imputer = impute.SimpleImputer(strategy='mean')
            customer_age_group_imputer.fit(X_train[['f_customer_age_group']])
            X_train['f_customer_age_group'] = customer_age_group_imputer.transform(X_train[['f_customer_age_group']])
            # - Policy Age
            policy_age_group_imputer = impute.SimpleImputer(strategy='mean')
            policy_age_group_imputer.fit(X_train[['f_policy_age']])
            X_train['f_policy_age'] = policy_age_group_imputer.transform(X_train[['f_policy_age']])
            # - Claim Completion Time
            X_train['f_claim_completion_time_3m'].fillna(0, inplace=True)
            X_train['f_claim_completion_time_6m'].fillna(0, inplace=True)
            X_train['f_claim_completion_time_12m'].fillna(0, inplace=True)

            # Normalize data

            # Cluster data based on age/product_type

            #############################################################################
            # - Impute missing values // TEST
            #############################################################################
            X_test['f_customer_age_group'] = customer_age_group_imputer.transform(X_test[['f_customer_age_group']])
            X_test['f_policy_age'] = policy_age_group_imputer.transform(X_test[['f_policy_age']])
            X_test['f_claim_completion_time_3m'].fillna(0, inplace=True)
            X_test['f_claim_completion_time_6m'].fillna(0, inplace=True)
            X_test['f_claim_completion_time_12m'].fillna(0, inplace=True)

            #############################################################################
            # - Training
            #############################################################################
            clf.fit(X_train, y_train)
            feat_imp = clf.feature_importances_
            print(feat_imp)

            #############################################################################
            # - Prediction
            #############################################################################
            y_pred = clf.predict(X_test)
            y_pred_probs = clf.predict_proba(X_test)[:, 1]

            #############################################################################
            # - Evaluation
            #############################################################################
            plt_pos_x, plt_pos_y = folder_nb_2_position(fold_nb)

            # - Confusion Matrix
            cm = metrics.confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # - AUC
            auc = metrics.roc_auc_score(y_test, y_pred_probs)

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            overall_metrics['confusion_matrix'].append(cm)
            overall_metrics['accuracy'].append(accuracy)
            overall_metrics['precision'].append(precision)
            overall_metrics['recall'].append(recall)
            overall_metrics['auc'].append(auc)

            write_option = 'a' if fold_nb > 0 else 'w'
            with open(os.path.join(destination_subfolder, 'results_raw.txt'), write_option) as f:
                f.write('#' + '-' * 79 + '\n')
                f.write(f'# [Fold] - #{fold_nb + 1}\n')
                f.write('#' + '-' * 79 + '\n')
                f.write(f'{str(cm)}\n')
                f.write(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n\n')
                f.write(f'Accuracy:  {accuracy}\n')
                f.write(f'Precision: {precision}\n')
                f.write(f'Recall:    {recall}\n')
                f.write(f'AUC:       {auc}\n')

            # Save confusion matrix as heatmap
            # fig = plt.figure()
            # plt.matshow(cm)
            # plt.colorbar()
            # plt.ylabel('True Label')
            # plt.xlabel('Predicated Label')
            # plt.savefig(os.path.join(destination_subfolder, 'confusion_matrix.jpg'))

            # - ROC Curve
            fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_probs)
            axs_roc[plt_pos_x, plt_pos_y].plot(fpr, tpr, marker='.')
            axs_roc[plt_pos_x, plt_pos_y].set_aspect(1)
            axs_roc[plt_pos_x, plt_pos_y].set_xlabel('False Positive Rate')
            axs_roc[plt_pos_x, plt_pos_y].set_ylabel('True Positive Rate')
            axs_roc[plt_pos_x, plt_pos_y].set_xticks([])
            axs_roc[plt_pos_x, plt_pos_y].set_yticks([])
            axs_roc[plt_pos_x, plt_pos_y].set_title(f'K={fold_nb+1}')

            # - Precision/Recall Curve
            fpr, tpr, _ = metrics.precision_recall_curve(y_test, y_pred_probs)
            axs_prc[plt_pos_x, plt_pos_y].plot(fpr, tpr, marker='.')
            axs_prc[plt_pos_x, plt_pos_y].set_aspect(0.8)
            axs_prc[plt_pos_x, plt_pos_y].set_xlabel('Recall')
            axs_prc[plt_pos_x, plt_pos_y].set_ylabel('Precision')
            axs_prc[plt_pos_x, plt_pos_y].set_xticks([])
            axs_prc[plt_pos_x, plt_pos_y].set_yticks([])
            axs_prc[plt_pos_x, plt_pos_y].set_title(f'K={fold_nb + 1}')

            fold_nb += 1

        # Save ROC
        axs_roc[1, 2].set_visible(False)
        fig_roc.savefig(os.path.join(destination_subfolder, 'ROC.png'))

        # Save Precision Recall Curve
        axs_prc[1, 2].set_visible(False)
        fig_prc.savefig(os.path.join(destination_subfolder, 'PRC.png'))

        overall_confusion_matrix = sum(overall_metrics['confusion_matrix'])
        overall_tn, overall_fp, overall_fn, overall_tp = overall_confusion_matrix.ravel()

        overall_accuracy = (overall_tp + overall_tn) / (overall_tp + overall_tn + overall_fp + overall_fn)
        overall_precision = overall_tp / (overall_tp + overall_fp)
        overall_recall = overall_tp / (overall_tp + overall_fn)
        overall_auc = statistics.mean(overall_metrics['auc'])
        stdev_accuracy = statistics.stdev(overall_metrics['accuracy'])
        stdev_precision = statistics.stdev(overall_metrics['precision'])
        stdev_recall = statistics.stdev(overall_metrics['recall'])
        stdev_auc = statistics.stdev(overall_metrics['auc'])
        with open(os.path.join(destination_subfolder, 'results_raw.txt'), 'a') as f:
            f.write('#' + '-' * 79 + '\n')
            f.write(f'# OVERALL\n')
            f.write('#' + '-' * 79 + '\n')
            f.write(f'{str(overall_confusion_matrix)}\n')
            f.write(f'TN: {overall_tn}, FP: {overall_fp}, FN: {overall_fn}, TP: {overall_tp}\n\n')
            f.write(f'Accuracy:  {overall_accuracy}\t[std-dev: {stdev_accuracy}]\n')
            f.write(f'Precision: {overall_precision}\t[std-dev: {stdev_precision}]\n')
            f.write(f'Recall:    {overall_recall}\t[std-dev: {stdev_recall}]\n')
            f.write(f'AUC:       {overall_auc}\t[std-dev: {stdev_auc}]\n')

