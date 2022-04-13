import numpy as np
import pandas as pd
import pickle
import random
from specific_nucbind.classifier import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score, classification_report, auc, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import joblib

# Processing original labels to binarized labels
def binary_label(label_list, class_id):
    output_label = []
    for i in range(len(label_list)):
        if label_list[i] == class_id:
            output_label.append(1)
        else:
            output_label.append(0)
    return output_label

if __name__ == '__main__':
    """
    Generate binary classifiers for each binding nucleic acid.
    For each type of binding nucleic acid, generate protein sequence feature from the corresponding training subset by feature_ml.py,
    and then change the file path to where you saved it.
    """
    traindata_pickle = open('./feature/train/trainfeature_lgb_eachclass.pickle', 'rb')
    traindata = pickle.load(traindata_pickle)
    trainlabel_pickle = open('./feature/train/trainlabel_lgb_eachclass.pickle', 'rb')
    trainlabel = pickle.load(trainlabel_pickle)
    # 0: nonbind; 1: A-DNA bind; 2: B-DNA bind; 3: ssDNA bind; 4: mRNA bind; 5: tRNA bind; 6: rRNA bind.
    ADNA_traindata = traindata['1']
    ADNA_trainlabel = trainlabel['1']
    ADNA_binary_trainlabel = binary_label(ADNA_trainlabel, 1)
    ADNA_traindata = np.array(ADNA_traindata)
    ADNA_binary_trainlabel = np.array(ADNA_binary_trainlabel)

    x_train = ADNA_traindata
    y_train = ADNA_binary_trainlabel

    # 确定n_estimators
    # params = {'boosting': 'gbdt',
    #           'objective': 'binary',
    #           'metric': 'auc',
    #           'max_depth': 7,
    #           'num_leaves': 50,
    #           'bagging_fraction': 0.8,
    #           'feature_fraction': 0.8,
    #           'learning_rate': 0.1,
    #           'nthread': 4
    #           }
    # data_train = lgb.Dataset(x_train, y_train)
    # cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True,
    #                     metrics='auc', early_stopping_rounds=50, seed=0)
    #
    # print('best n_estimators:', len(cv_results['auc-mean']))
    # print('best cv score:', pd.Series(cv_results['auc-mean']).max())

    # grid search to find optimal hyper-parameter
    parameters = {
        'max_depth': range(3, 8, 1),
        'num_leaves': range(5, 100, 5),
        'min_child_samples': [18, 19, 20, 21, 22],
        'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
        'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0,0.001,0.01,0.03,0.05,0.08,0.1,0.3,0.5],
        'reg_lambda': [0,0.001,0.01,0.03,0.05,0.08,0.1,0.3,0.5],
        'learning_rate': [0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]        
    }

    gbm = lgb.LGBMClassifier(bagging_fraction=0.6, feature_fraction=0.8, is_unbalance=True,
               learning_rate=0.1, max_depth=5, metric='binary_logloss,auc',
               min_child_samples=20, n_estimators=400, num_leaves=20,
               objective='binary', reg_alpha=0, reg_lambda=0.5)
    gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='roc_auc', cv=5)
    gsearch.fit(x_train, y_train)
    print('optimal hyper-parameter:{0}'.format(gsearch.best_params_))
    print('best model:{0}'.format(gsearch.best_score_))
    print(gsearch.cv_results_['mean_test_score'])
    print(gsearch.cv_results_['params'])
    
    gbm.fit(x_train, y_train)

    joblib.dump(gbm, './model/LightGBM/lgb_ADNA.pkl')























