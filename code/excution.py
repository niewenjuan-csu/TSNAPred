import pickle
import numpy as np
import joblib
from sklearn.metrics import roc_curve, accuracy_score, auc, f1_score, confusion_matrix, matthews_corrcoef, recall_score, precision_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import random


def create_network():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=14))
    model.add(Dropout(0.7))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def lightgmprediction_nn(testfeature, testlabel):
    model0 = joblib.load('./model/dl/lgb_nonbind.pkl')
    model1 = joblib.load('./model/dl/lgb_ADNA.pkl')
    model2 = joblib.load('./model/dl/lgb_BDNA.pkl')
    model3 = joblib.load('./model/dl/lgb_ssDNA.pkl')
    model4 = joblib.load('./model/dl/lgb_mRNA.pkl')
    model5 = joblib.load('./model/dl/lgb_tRNA.pkl')
    model6 = joblib.load('./model/dl/lgb_rRNA.pkl')
    # load train data
    trainfeature_pickle = open(
        './feature/train/trainfeature_lgb.pickle', 'rb')
    trainfeature = pickle.load(trainfeature_pickle)
    trainlabel_pickle = open(
        './feature/train/trainlabel_lgb.pickle', 'rb')
    trainlabel = pickle.load(trainlabel_pickle)

    trainfeature = np.array(trainfeature)
    trainlabel = np.array(trainlabel)
    trainlabel_binary = np_utils.to_categorical(trainlabel, num_classes=7)

    nonbind_prob = model0.predict_proba(trainfeature)
    ANDA_prob = model1.predict_proba(trainfeature)
    BDNA_prob = model2.predict_proba(trainfeature)
    ssDNA_prob = model3.predict_proba(trainfeature)
    mRNA_prob = model4.predict_proba(trainfeature)
    tRNA_prob = model5.predict_proba(trainfeature)
    rRNA_prob = model6.predict_proba(trainfeature)

    temp = np.hstack((nonbind_prob, ANDA_prob))
    temp = np.hstack((temp, BDNA_prob))
    temp = np.hstack((temp, ssDNA_prob))
    temp = np.hstack((temp, mRNA_prob))
    temp = np.hstack((temp, tRNA_prob))
    feature = np.hstack((temp, rRNA_prob))
    print(feature)

    nonbind_prob_test = model0.predict_proba(testfeature)
    ANDA_prob_test = model1.predict_proba(testfeature)
    BDNA_prob_test = model2.predict_proba(testfeature)
    ssDNA_prob_test = model3.predict_proba(testfeature)
    mRNA_prob_test = model4.predict_proba(testfeature)
    tRNA_prob_test = model5.predict_proba(testfeature)
    rRNA_prob_test = model6.predict_proba(testfeature)

    temp = np.hstack((nonbind_prob_test, ANDA_prob_test))
    temp = np.hstack((temp, BDNA_prob_test))
    temp = np.hstack((temp, ssDNA_prob_test))
    temp = np.hstack((temp, mRNA_prob_test))
    temp = np.hstack((temp, tRNA_prob_test))
    feature_test = np.hstack((temp, rRNA_prob_test))
    print(feature_test)

    network = create_network()
    network.fit(feature, trainlabel_binary, batch_size=256, epochs=500, verbose=1)
    y_prob = network.predict_proba(feature_test)
    y_prob = np.array(y_prob)

    return y_prob

def capsnetprediction_nn(testfeature, testlabel):
    testfeature = np.array(testfeature).reshape(-1, 19, 30, 1)

    model0 = CaspNet(input_shape=(19, 30, 1), n_class=2, num_routing=3)
    model0.load_weights('./model/dl/capsnet_nonbind.h5')
    model1 = CaspNet(input_shape=(19, 30, 1), n_class=2, num_routing=3)
    model1.load_weights('./model/dl/capsnet_ADNA.h5')
    model2 = CaspNet(input_shape=(19, 30, 1), n_class=2, num_routing=3)
    model2.load_weights('./model/dl/capsnet_BDNA.h5')
    model3 = CaspNet(input_shape=(19, 30, 1), n_class=2, num_routing=3)
    model3.load_weights('./model/dl/capsnet_ssDNA.h5')
    model4 = CaspNet(input_shape=(19, 30, 1), n_class=2, num_routing=3)
    model4.load_weights('./model/dl/capsnet_mRNA.h5')
    model5 = CaspNet(input_shape=(19, 30, 1), n_class=2, num_routing=3)
    model5.load_weights('./model/dl/capsnet_tRNA.h5')
    model6 = CaspNet(input_shape=(19, 30, 1), n_class=2, num_routing=3)
    model6.load_weights('./model/dl/capsnet_rRNA.h5')

    trainfeature_pickle = open(
        './feature/train/trainfeature_capsnet.pickle', 'rb')
    trainfeature = pickle.load(trainfeature_pickle)
    trainlabel_pickle = open('./feature/train/trainlabel_capsnet.pickle', 'rb')
    trainlabel = pickle.load(trainlabel_pickle)
    trainfeature = np.array(trainfeature).reshape(-1, 19, 30, 1)
    trainlabel = np.array(trainlabel)
    trainlabel_binary = np_utils.to_categorical(trainlabel, num_classes=7)

    nonbind_prob = model0.predict(trainfeature)
    ANDA_prob = model1.predict(trainfeature)
    BDNA_prob = model2.predict(trainfeature)
    ssDNA_prob = model3.predict(trainfeature)
    mRNA_prob = model4.predict(trainfeature)
    tRNA_prob = model5.predict(trainfeature)
    rRNA_prob = model6.predict(trainfeature)

    temp = np.hstack((nonbind_prob, ANDA_prob))
    temp = np.hstack((temp, BDNA_prob))
    temp = np.hstack((temp, ssDNA_prob))
    temp = np.hstack((temp, mRNA_prob))
    temp = np.hstack((temp, tRNA_prob))
    feature = np.hstack((temp, rRNA_prob))
    print(feature)

    nonbind_prob_test = model0.predict(testfeature)
    ANDA_prob_test = model1.predict(testfeature)
    BDNA_prob_test = model2.predict(testfeature)
    ssDNA_prob_test = model3.predict(testfeature)
    mRNA_prob_test = model4.predict(testfeature)
    tRNA_prob_test = model5.predict(testfeature)
    rRNA_prob_test = model6.predict(testfeature)

    temp = np.hstack((nonbind_prob_test, ANDA_prob_test))
    temp = np.hstack((temp, BDNA_prob_test))
    temp = np.hstack((temp, ssDNA_prob_test))
    temp = np.hstack((temp, mRNA_prob_test))
    temp = np.hstack((temp, tRNA_prob_test))
    feature_test = np.hstack((temp, rRNA_prob_test))
    print(feature_test)

    network = create_network()
    network.fit(feature, trainlabel_binary, batch_size=256, epochs=500, verbose=1)
    y_prob = network.predict_proba(feature_test)
    y_prob = np.array(y_prob)

    return y_prob

if __name__ == '__main__':
    # feature for machine learning
    testfeaturelgb_pickle = open(
        './feature/test/testfeature_lgb.pickle', 'rb')
    testfeature_lgb = pickle.load(testfeaturelgb_pickle)
    testfeature_lgb = np.array(testfeaturelgb)
    testlabellgb_pickle = open(
        './feature/test/testlabel_lgb.pickle', 'rb')
    testlabel_lgb = pickle.load(testlabellgb_pickle)
    testlabel_lgb = np.array(testlabellgb)

    # feature for deep learning
    testfeature_pickle = open('./feature/test/testfeature_capsnet.pickle', 'rb')
    testfeature_capsnet = pickle.load(testfeature_pickle)
    testlabel_pickle = open('./feature/test/testlabel_capsnet.pickle', 'rb')
    testlabel = pickle.load(testlabel_pickle)
    testlabel_capsnet = np.array(testlabel)


    # 三分类--label处理
    # 0:nonbind 1:DNA-binding 2:RNA-binding
    tri_testlabel_capsnet = []
    tri_testlabel_lgb = []
    for i in range(len(testlabel_capsnet)):
        if testlabel_capsnet[i] == 1 or testlabel_capsnet[i] == 2 or testlabel_capsnet[i] == 3:
            tri_testlabel_capsnet.append(1)
        elif testlabel_capsnet[i] == 4 or testlabel_capsnet[i] == 5 or testlabel_capsnet[i] == 6:
            tri_testlabel_capsnet.append(2)
        else:
            tri_testlabel_capsnet.append(testlabel_capsnet[i])

    for i in range(len(testlabel_lgb)):
        if testlabel_lgb[i] == 1 or testlabel_lgb[i] == 2 or testlabel_lgb[i] == 3:
            tri_testlabel_lgb.append(1)
        elif testlabel_lgb[i] == 4 or testlabel_lgb[i] == 5 or testlabel_lgb[i] == 6:
            tri_testlabel_lgb.append(2)
        else:
            tri_testlabel_lgb.append(testlabel_lgb[i])


    lgbprediction_result = lightgmprediction_nn(testfeature_lgb, testlabel_lgb)
    print(lgbprediction_result)

    caspnetprediction_result = capsnetprediction_nn(testfeature_capsnet, testlabel_capsnet)
    print(caspnetprediction_result)

    ensembleprediction = 0.5 * lgbprediction_result + 0.5 * caspnetprediction_result
    ensembleprediction = np.array(ensembleprediction)

    # """
    # prediction = [np.argmax(y) for y in caspnetprediction_result]
    # prediction = np.array(prediction)

    y_test = label_binarize(testlabel_capsnet, classes=[i for i in range(7)])
    # y_pred = label_binarize(prediction, classes=[i for i in range(7)])

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

    colors = cycle(['darkgray', 'skyblue', 'bisque', 'violet', 'indianred', 'moccasin', 'salmon'])
    for i, color in zip(target, colors):
        plt.plot(fpr[i], tpr[i], color=color, linestyle='--',
                 label='ROC curve of %s (area=%0.4f)' % (i, roc_auc[i]))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC for TSNAPred')
    plt.legend(loc="lower right")
    plt.show()

    """
    # 阈值划分
    threshold_prob = dict()
    for i in range(len(target)):
        threshold_prob[target[i]] = []
        flag = 0
        for j in range(len(fpr[target[i]])):
            if fpr[target[i]][j] >= 0.05 and flag == 0:
                threshold_prob[target[i]].append(thresholds[target[i]][j])
                flag += 1
            if fpr[target[i]][j] >= 0.1 and flag == 1:
                threshold_prob[target[i]].append(thresholds[target[i]][j])
                flag += 1
            if fpr[target[i]][j] >= 0.15 and flag == 2:
                threshold_prob[target[i]].append(thresholds[target[i]][j])
                flag += 1
            if fpr[target[i]][j] >= 0.2 and flag == 3:
                threshold_prob[target[i]].append(thresholds[target[i]][j])
    # print(threshold_prob)
    # min_threshold = 100
    # for i in range(1, len(target)):
    #     threshold = threshold_prob[target[i]][0]
    #     if threshold < min_threshold:
    #         min_threshold = threshold
    prediction_fpr5 = []
    prediction_fpr10 = []
    prediction_fpr15 = []
    prediction_fpr20 = []
    prediction_sen_equal_spe = []
    for y in ensembleprediction:
        eachpred_5 = []
        eachpred_10 = []
        eachpred_15 = []
        eachpred_20 = []
        eachpred_sen_equal_spe = []
        for j in range(len(target)):
            if y[j] >= threshold_prob[target[j]][0]:
                eachpred_5.append(1)
            else:
                eachpred_5.append(0)
            if y[j] >= threshold_prob[target[j]][1]:
                eachpred_10.append(1)
            else:
                eachpred_10.append(0)
            if y[j] >= threshold_prob[target[j]][2]:
                eachpred_15.append(1)
            else:
                eachpred_15.append(0)
            if y[j] >= threshold_prob[target[j]][3]:
                eachpred_20.append(1)
            else:
                eachpred_20.append(0)

        prediction_fpr5.append(eachpred_5)
        prediction_fpr10.append(eachpred_10)
        prediction_fpr15.append(eachpred_15)
        prediction_fpr20.append(eachpred_20)
    prediction_fpr5 = np.array(prediction_fpr5)
    prediction_fpr10 = np.array(prediction_fpr10)
    prediction_fpr15 = np.array(prediction_fpr15)
    prediction_fpr20 = np.array(prediction_fpr20)

    print("\n FPR at 5%:")
    for i in range(len(target)):
        recall[target[i]] = recall_score(y_test[:, i], prediction_fpr5[:, i])
        mcc[target[i]] = matthews_corrcoef(y_test[:, i], prediction_fpr5[:, i])
        print('the class %s  - recall:%s -mcc:%s' % (
        str(target[i]), str(round(recall[target[i]], 4)), str(round(mcc[target[i]], 4))))

    print("\n FPR at 10%:")
    for i in range(len(target)):
        recall[target[i]] = recall_score(y_test[:, i], prediction_fpr10[:, i])
        mcc[target[i]] = matthews_corrcoef(y_test[:, i], prediction_fpr10[:, i])
        print('the class %s  - recall:%s -mcc:%s' % (
            str(target[i]), str(round(recall[target[i]], 4)), str(round(mcc[target[i]], 4))))

    print("\n FPR at 15%:")
    for i in range(len(target)):
        recall[target[i]] = recall_score(y_test[:, i], prediction_fpr15[:, i])
        mcc[target[i]] = matthews_corrcoef(y_test[:, i], prediction_fpr15[:, i])
        print('the class %s  - recall:%s -mcc:%s' % (
            str(target[i]), str(round(recall[target[i]], 4)), str(round(mcc[target[i]], 4))))

    print("\n FPR at 20%:")
    for i in range(len(target)):
        recall[target[i]] = recall_score(y_test[:, i], prediction_fpr20[:, i])
        mcc[target[i]] = matthews_corrcoef(y_test[:, i], prediction_fpr20[:, i])
        print('the class %s  - recall:%s -mcc:%s' % (
            str(target[i]), str(round(recall[target[i]], 4)), str(round(mcc[target[i]], 4))))
    """


    """
    # seach for the optimal weight from [0,1]
    accuracy = []
    AUC = []
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    target = ['nonbind', 'ADNA bind', 'BDNA bind', 'ssDNA bind', 'mRNA bind', 'tRNA bind', 'rRNA bind']
    TPR_array = []
    FRP_array = []
    ACC_array = []
    for i in range(0, 105, 5):
        a1 = float(i / 100)
        a2 = float(1 - a1)
        ensembleprediction = a1 * lgbprediction_result + a2 * caspnetprediction_result
        prediction = [np.argmax(y) for y in ensembleprediction]

        prediction = np.array(prediction)
        y_test = label_binarize(testlabel_capsnet, classes=[i for i in range(7)])
        y_pred = label_binarize(prediction, classes=[i for i in range(7)])
        eachAUC = []
        for k in range(len(target)):
            fpr[target[k]], tpr[target[k]], _ = roc_curve(y_test[:, k], ensembleprediction[:, k])
            roc_auc[target[k]] = auc(fpr[target[k]], tpr[target[k]])
            eachAUC.append(roc_auc[target[k]])

        AUC.append(eachAUC)
        acc = accuracy_score(testlabel, prediction)
        accuracy.append(acc)
        f1 = f1_score(testlabel, prediction, average='weighted')
        print("a1:%f, a2:%f accuracy:%f f1:%f" % (a1, a2, acc, f1))

        cnf_matrix = confusion_matrix(testlabel, prediction)
        print(cnf_matrix)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)
        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)

        TPR_array.append(TPR)
        FRP_array.append(FPR)
        ACC_array.append(ACC)
    # """

    # """
    # 三分类
    tri_predictionprob = []
    for i in range(len(ensembleprediction)):
        eachprediction = []
        eachres_prob = ensembleprediction[i]
        eachprediction.append(eachres_prob[0])
        eachprediction.append(np.max(eachres_prob[1:4]))
        eachprediction.append(np.max(eachres_prob[4:7]))
        tri_predictionprob.append(eachprediction)
    target = ['nonbind', 'DNA binding', 'RNA binding']
    prediction = [np.argmax(y) for y in tri_predictionprob]

    tri_predictionprob = np.array(tri_predictionprob)
    tri_testlabel = np.array(tri_testlabel_capsnet)
    tri_prediction = np.array(prediction)
    tri_test = label_binarize(tri_testlabel, classes=[i for i in range(3)])
    tri_pred = label_binarize(tri_prediction, classes=[i for i in range(3)])


    fpr_tri = dict()
    tpr_tri = dict()
    roc_auc_tri = dict()
    f1_value = dict()
    recall = dict()
    precision = dict()
    accuracy = dict()
    mcc = dict()
    thresholds_tri = dict()

    plt.figure()
    for i in range(len(target)):
        fpr_tri[target[i]], tpr_tri[target[i]], thresholds_tri[target[i]] = roc_curve(tri_test[:, i], tri_predictionprob[:, i])
        roc_auc_tri[target[i]] = auc(fpr_tri[target[i]], tpr_tri[target[i]])
        accuracy[target[i]] = accuracy_score(tri_test[:, i], tri_pred[:, i])
        f1_value[target[i]] = f1_score(tri_test[:, i], tri_pred[:, i])
        recall[target[i]] = recall_score(tri_test[:, i], tri_pred[:, i])
        precision[target[i]] = precision_score(tri_test[:, i], tri_pred[:, i])
        print('the class %s - acc:%s - f1:%s - recall:%s - precision:%s' % (str(target[i]),
                                                                            str(round(accuracy[target[i]], 4)),
                                                                            str(round(f1_value[target[i]], 4)),
                                                                            str(round(recall[target[i]], 4)),
                                                                            str(round(precision[target[i]],
                                                                                      4))))

    colors = cycle(['darkgray', 'skyblue', 'salmon'])
    for i, color in zip(target, colors):
        plt.plot(fpr_tri[i], tpr_tri[i], color=color, linestyle='--',
                 label='ROC curve of %s (area=%0.4f)' % (i, roc_auc[i]))
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC for DNA and RNA binding')
    plt.legend(loc="lower right")
    plt.show()
    # """

    # """
    # 阈值划分
    threshold_prob = dict()
    for i in range(len(target)):
        threshold_prob[target[i]] = []
        flag = 0
        for j in range(len(fpr_tri[target[i]])):
            if fpr_tri[target[i]][j] >= 0.05 and flag == 0:
                threshold_prob[target[i]].append(thresholds_tri[target[i]][j])
                flag += 1
            if fpr_tri[target[i]][j] >= 0.1 and flag == 1:
                threshold_prob[target[i]].append(thresholds_tri[target[i]][j])
                flag += 1
            if fpr_tri[target[i]][j] >= 0.15 and flag == 2:
                threshold_prob[target[i]].append(thresholds_tri[target[i]][j])
                flag += 1
            if fpr_tri[target[i]][j] >= 0.2 and flag == 3:
                threshold_prob[target[i]].append(thresholds_tri[target[i]][j])
                flag += 1
    # print(threshold_prob)
   
    prediction_fpr5 = []
    prediction_fpr10 = []
    prediction_fpr15 = []
    prediction_fpr20 = []
    for y in tri_predictionprob:
        eachpred_5 = []
        eachpred_10 = []
        eachpred_15 = []
        eachpred_20 = []
        for j in range(len(target)):
            if y[j] >= threshold_prob[target[j]][0]:
                eachpred_5.append(1)
            else:
                eachpred_5.append(0)
            if y[j] >= threshold_prob[target[j]][1]:
                eachpred_10.append(1)
            else:
                eachpred_10.append(0)
            if y[j] >= threshold_prob[target[j]][2]:
                eachpred_15.append(1)
            else:
                eachpred_15.append(0)
            if y[j] >= threshold_prob[target[j]][3]:
                eachpred_20.append(1)
            else:
                eachpred_20.append(0)
        prediction_fpr5.append(eachpred_5)
        prediction_fpr10.append(eachpred_10)
        prediction_fpr15.append(eachpred_15)
        prediction_fpr20.append(eachpred_20)
    prediction_fpr5 = np.array(prediction_fpr5)
    prediction_fpr10 = np.array(prediction_fpr10)
    prediction_fpr15 = np.array(prediction_fpr15)
    prediction_fpr20 = np.array(prediction_fpr20)


    print("\n FPR at 5%:")
    for i in range(len(target)):
        accuracy[target[i]] = accuracy_score(tri_test[:, i], prediction_fpr5[:, i])
        f1_value[target[i]] = f1_score(tri_test[:, i], prediction_fpr5[:, i])
        recall[target[i]] = recall_score(tri_test[:, i], prediction_fpr5[:, i])
        precision[target[i]] = precision_score(tri_test[:, i], prediction_fpr5[:, i])
        mcc[target[i]] = matthews_corrcoef(tri_test[:, i], prediction_fpr5[:, i])
        print('the class %s - acc:%s - f1:%s - recall:%s - precision:%s -mcc:%s' % (str(target[i]),
                                                                            str(round(accuracy[target[i]], 4)),
                                                                            str(round(f1_value[target[i]], 4)),
                                                                            str(round(recall[target[i]], 4)),
                                                                            str(round(precision[target[i]],
                                                                                      4)),
                                                                                    str(round(mcc[target[i]], 4))))
    print("\n FPR at 10%:")
    for i in range(len(target)):
        accuracy[target[i]] = accuracy_score(tri_test[:, i], prediction_fpr10[:, i])
        f1_value[target[i]] = f1_score(tri_test[:, i], prediction_fpr10[:, i])
        recall[target[i]] = recall_score(tri_test[:, i], prediction_fpr10[:, i])
        precision[target[i]] = precision_score(tri_test[:, i], prediction_fpr10[:, i])
        mcc[target[i]] = matthews_corrcoef(tri_test[:, i], prediction_fpr10[:, i])
        print('the class %s - acc:%s - f1:%s - recall:%s - precision:%s -mcc:%s' % (str(target[i]),
                                                                                    str(round(accuracy[target[i]], 4)),
                                                                                    str(round(f1_value[target[i]], 4)),
                                                                                    str(round(recall[target[i]], 4)),
                                                                                    str(round(precision[target[i]],
                                                                                              4)),
                                                                                    str(round(mcc[target[i]], 4))))

    print("\n FPR at 15%:")
    for i in range(len(target)):
        accuracy[target[i]] = accuracy_score(tri_test[:, i], prediction_fpr15[:, i])
        f1_value[target[i]] = f1_score(tri_test[:, i], prediction_fpr15[:, i])
        recall[target[i]] = recall_score(tri_test[:, i], prediction_fpr15[:, i])
        precision[target[i]] = precision_score(tri_test[:, i], prediction_fpr15[:, i])
        mcc[target[i]] = matthews_corrcoef(tri_test[:, i], prediction_fpr15[:, i])
        print('the class %s - acc:%s - f1:%s - recall:%s - precision:%s -mcc:%s' % (str(target[i]),
                                                                                    str(round(accuracy[target[i]], 4)),
                                                                                    str(round(f1_value[target[i]], 4)),
                                                                                    str(round(recall[target[i]], 4)),
                                                                                    str(round(precision[target[i]],
                                                                                              4)),
                                                                                    str(round(mcc[target[i]], 4))))
    print("\n FPR at 20%:")
    for i in range(len(target)):
        accuracy[target[i]] = accuracy_score(tri_test[:, i], prediction_fpr20[:, i])
        f1_value[target[i]] = f1_score(tri_test[:, i], prediction_fpr20[:, i])
        recall[target[i]] = recall_score(tri_test[:, i], prediction_fpr20[:, i])
        precision[target[i]] = precision_score(tri_test[:, i], prediction_fpr20[:, i])
        mcc[target[i]] = matthews_corrcoef(tri_test[:, i], prediction_fpr20[:, i])
        print('the class %s - acc:%s - f1:%s - recall:%s - precision:%s -mcc:%s' % (str(target[i]),
                                                                                    str(round(accuracy[target[i]], 4)),
                                                                                    str(round(f1_value[target[i]], 4)),
                                                                                    str(round(recall[target[i]], 4)),
                                                                                    str(round(precision[target[i]],
                                                                                              4)),
                                                                                    str(round(mcc[target[i]], 4))))
    # """







