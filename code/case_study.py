import pickle
import numpy as np
import joblib
from excution import lightgmprediction_nn, capsnetprediction_nn
from sklearn.metrics import roc_curve, accuracy_score, auc, f1_score, confusion_matrix, matthews_corrcoef, recall_score, precision_score, matthews_corrcoef
from dnagenie_result import *


if __name__ == '__main__':
    # feature for LightGBM
    testfeaturelgb_pickle = open(
        '~/feature/case_study/O69644_feature.pickle', 'rb')
    testfeaturelgb = pickle.load(testfeaturelgb_pickle)
    testfeaturelgb = np.array(testfeaturelgb)
    testlabellgb_pickle = open(
        '~/feature/case_study/O69644_label.pickle', 'rb')
    testlabellgb = pickle.load(testlabellgb_pickle)
    testlabellgb = np.array(testlabellgb)

    # feature for CapsNet
    testfeature_pickle = open(
        '~/feature/case_study/O69644_feature_cpasnet_21.pickle', 'rb')
    testfeature = pickle.load(testfeature_pickle)
    testlabel_pickle = open(
        '~/feature/case_study/O69644_label_cpasnet_21.pickle', 'rb')
    testlabel = pickle.load(testlabel_pickle)
    testlabel = np.array(testlabel)

    # 三分类--label处理
    tri_testlabel_capsnet = []
    tri_testlabel_lgb = []
    for i in range(len(testlabel)):
        if testlabel[i] == 1 or testlabel[i] == 2 or testlabel[i] == 3:
            tri_testlabel_capsnet.append(1)
            tri_testlabel_lgb.append(1)
        elif testlabel[i] == 4 or testlabel[i] == 5 or testlabel[i] == 6:
            tri_testlabel_capsnet.append(2)
            tri_testlabel_lgb.append(2)
        else:
            tri_testlabel_capsnet.append(testlabel[i])
            tri_testlabel_lgb.append(testlabel[i])

    lgbprediction_result = lightgmprediction_nn(testfeaturelgb, testlabellgb)
    print(lgbprediction_result)

    caspnetprediction_result = capsnetprediction_nn(testfeature, testlabel)
    print(caspnetprediction_result)

    ensembleprediction = 0.5 * lgbprediction_result + 0.5 * caspnetprediction_result
    ensembleprediction = np.array(ensembleprediction)
    print(ensembleprediction)

    y_test = label_binarize(testlabel, classes=[i for i in range(7)])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    mcc = dict()
    recall = dict()
    target = ['nonbind', 'ADNA bind', 'BDNA bind', 'ssDNA bind', 'mRNA bind', 'tRNA bind', 'rRNA bind']
    plt.figure()
    for i in range(len(target)):
        fpr[target[i]], tpr[target[i]], thresholds[target[i]] = roc_curve(y_test[:, i], ensembleprediction[:, i])
        roc_auc[target[i]] = auc(fpr[target[i]], tpr[target[i]])

    threshold_prob = dict()
    for i in range(len(target)):
        threshold_prob[target[i]] = []
        for j in range(len(fpr[target[i]])):
            if fpr[target[i]][j] >= 0.05 and flag == 0:
                threshold_prob[target[i]].append(thresholds[target[i]][j])
                break
    # print(threshold_prob)
    prediction_fpr5 = []
    for y in ensembleprediction:
        eachpred_5 = []
        for j in range(len(target)):
            if y[j] >= threshold_prob[target[j]][0]:
                eachpred_5.append(1)
            else:
                eachpred_5.append(0)
        prediction_fpr5.append(eachpred_5)
    prediction_fpr5 = np.array(prediction_fpr5)


    type = 0
    for i in range(1, len(target)):
        y = prediction_fpr5[:, i]
        if 0 not in y:
            continue
        else:
            print('the protein contains %s binding residues.' % (target[i]))
            type = i
            prediction = list(y)
            print('predicted binding label: %s' % (str(prediction)))

    np.savetxt('prediction_fpr5.txt', prediction, fmt='%d')

    print("\n FPR at 5%:")
    # print(prediction_fpr5)
    cnf_matrix_fpr_5 = confusion_matrix(y_test[:, type], prediction_fpr5[:, type])
    print(accuracy_score(y_test[:, type], prediction_fpr5[:, type]))
    print(precision_score(y_test[:, type], prediction_fpr5[:, type]))

    fpr_type, tpr_type, _ = roc_curve(y_test[:, type], ensembleprediction[:, type])
    roc_auc = auc(fpr_type, tpr_type)
    plt.plot(fpr_type, tpr_type, color='r', linestyle='--', label='ROC curve(area=%0.4f)' %(roc_auc))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('O69644')
    plt.legend(loc="lower right")
    plt.savefig('AUC_O69644.pdf')
    plt.show()







