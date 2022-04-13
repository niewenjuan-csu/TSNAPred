import numpy as np
import os
import linecache
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pickle
import random
import joblib

# Normalize numeric values to [0,1]
def normalize(value):
    a = 1 + math.exp(value)
    b = 1 / a
    return b

def dictolist(dic):
    newlist = []
    for key in dic:
        newlist.extend(dic[key])
    return newlist

def PSSMfeature(PSSMpath, uniport_id):
    pssmfilelines = linecache.getlines(PSSMpath + '/' + uniport_id + '.pssm')
    pssmDic = {}
    for line in pssmfilelines:
        content = line.split()
        residuePosition = int(content[0]) - 1
        pssmDic[residuePosition] = []
        for i in range(2, 22):
            pssmDic[residuePosition].append(normalize(int(content[i])))
    # print(pssmDic)
    return pssmDic

def psipredfeature(psipredoutpath, uniport_id):
    psipredDic = {}
    filelines = linecache.getlines(psipredoutpath + '/' + uniport_id + '/' + uniport_id + '.ss2')
    length = len(filelines)
    for i in range(2, length):
        residuenum = int(filelines[i].split()[0]) - 1
        psipredDic[residuenum] = []
        psipredDic[residuenum].append(float(filelines[i].split()[3]))
        psipredDic[residuenum].append(float(filelines[i].split()[4]))
        psipredDic[residuenum].append(float(filelines[i].split()[5]))
    # print(psipredDic)
    return psipredDic

def onehotfeature(fastaline, uniport_id):
    """
    :param fastaline: sequence, string
    :param uniport_id: protein_name
    :return:
    """
    onehotdic = {}
    length = len(fastaline)
    for i in range(length):
        residuename = fastaline[i]
        if residuename == 'A' or residuename == 'G' or residuename == 'V':
            onehotdic[i] = [0, 0, 0, 0, 0, 0, 1]
        elif residuename == 'I' or residuename == 'L' or residuename == 'F' or residuename == 'P':
            onehotdic[i] = [0, 0, 0, 0, 0, 1, 0]
        elif residuename == 'H' or residuename == 'N' or residuename == 'Q' or residuename == 'W':
            onehotdic[i] = [0, 0, 0, 0, 1, 0, 0]
        elif residuename == 'Y' or residuename == 'M' or residuename == 'T' or residuename == 'S':
            onehotdic[i] = [0, 0, 0, 1, 0, 0, 0]
        elif residuename == 'R' or residuename == 'K':
            onehotdic[i] = [0, 0, 1, 0, 0, 0, 0]
        elif residuename == 'D' or residuename == 'E':
            onehotdic[i] = [0, 1, 0, 0, 0, 0, 0]
        elif residuename == 'C':
            onehotdic[i] = [1, 0, 0, 0, 0, 0, 0]
        # residue is unknown:
        elif residuename == 'U':
            onehotdic[i] = [0, 0, 0, 0, 0, 0, 0]
    # print(chemicaldic)
    return onehotdic

def appendzero(windowsize, featureDic):
    seqlength = len(featureDic.keys())
    appendnum = int((windowsize + 1) / 2)
    for i in range(1, appendnum):
        featureDic[0 - i] = []
        featureDic[seqlength - 1 + i] = []
        # 前后补0；range范围为特征长度
        # deep-learning:30
        for a in range(30):
            featureDic[0 - i].append(0)
        for b in range(30):
            featureDic[seqlength - 1 + i].append(0)
    return featureDic


def combine(sequencelength, featuredic, windowsize):
    neighnum = int((windowsize - 1) / 2)
    combineDic = {}
    for i in range(0, sequencelength):
        combineDic[i] = []
        for a in range(i - neighnum, i + neighnum + 1):
            # combineDic[i].append(pssmdic[a])
            for each in featuredic[a]:
                # 每个氨基酸得到窗口大小的特征矩阵
                combineDic[i].append(each)
    featurelist = []
    for i in range(0, sequencelength):
        featurelist.append(combineDic[i])
    # print(featurelist)
    return featurelist

def featurecombine_for_deeplearning(uniport_id, PSSMpath, psipredoutpath, fastaline, windowsize=17):
    pssmdic = PSSMfeature(PSSMpath, uniport_id)
    psipreddic = psipredfeature(psipredoutpath, uniport_id)
    onehotdic = onehotfeature(fastaline, uniport_id)

    length = len(pssmdic.keys())
    featuredic = {}
    for i in range(length):
        featuredic[i] = []
        for each in pssmdic[i]:
            featuredic[i].append(each)
        for each in psipreddic[i]:
            featuredic[i].append(each)
        for each in onehotdic[i]:
            featuredic[i].append(each)
    appendedfeaturedic = appendzero(windowsize, featuredic)
    combinefeaturelist = combine(length, appendedfeaturedic, windowsize)
    return combinefeaturelist

def seperatefeature(combinefeaturelist, windowsize=17):
    pssmfeature = []
    psipredfeature = []
    chemicalfeature = []

    pssm_psipred = []
    pssm_onehot = []
    psipred_onehot = []

    for each in combinefeaturelist:
        pssmfeature_each = []
        psipredfeature_each = []
        chemicalfeature_each = []
        pssmpsi_each = []
        pssmonehot_each = []
        psionehot_each = []
        pssmflag = 0
        psipredflag = 20
        chemicalflag = 23
        for i in range(0, windowsize):
            for a in range(pssmflag, pssmflag + 20):
                pssmfeature_each.append(each[a])
            for b in range(psipredflag, psipredflag + 3):
                psipredfeature_each.append(each[b])
            for c in range(chemicalflag, chemicalflag + 7):
                chemicalfeature_each.append(each[c])
            pssmflag = pssmflag + 30
            psipredflag = psipredflag + 30
            chemicalflag = chemicalflag + 30
        pssmfeature.append(pssmfeature_each)
        psipredfeature.append(psipredfeature_each)
        chemicalfeature.append(chemicalfeature_each)

        pssmpsi_each.extend(pssmfeature_each)
        pssmpsi_each.extend(psipredfeature_each)
        pssm_psipred.append(pssmpsi_each)

        pssmonehot_each.extend(pssmfeature_each)
        pssmonehot_each.extend(chemicalfeature_each)
        pssm_onehot.append(pssmonehot_each)

        psionehot_each.extend(psipredfeature_each)
        psionehot_each.extend(chemicalfeature_each)
        psipred_onehot.append(psionehot_each)


    feature_combination = (pssmfeature, psipredfeature, chemicalfeature, pssm_psipred, pssm_onehot, psipred_onehot)
    return feature_combination


def labelize(labelline, uniport_id, bindtype):
    labellist = []
    length = len(labelline)
    if bindtype == 'DNA':
        for i in range(0, length):
            if labelline[i] == '.' or labelline[i] == 'L':
                labellist.append(0)
            if labelline[i] == 'A':
                labellist.append(1)
            if labelline[i] == 'B':
                labellist.append(2)
            if labelline[i] == 'S':
                labellist.append(3)
    if bindtype == 'mRNA':
        for i in range(0, length):
            if labelline[i] == '0':
                labellist.append(0)
            if labelline[i] == '1':
                labellist.append(4)
    if bindtype == 'tRNA':
        for i in range(0, length):
            if labelline[i] == '0':
                labellist.append(0)
            if labelline[i] == '1':
                labellist.append(5)
    if bindtype == 'rRNA':
        for i in range(0, length):
            if labelline[i] == '0':
                labellist.append(0)
            if labelline[i] == '1':
                labellist.append(6)
    if bindtype == 'nonbind':
        for i in range(0, length):
            labellist.append(0)
    return labellist

def readtxt(file, bindtype):
    protein = []
    protein_fasta = {}
    pritein_label = {}

    f = open(file, 'r')
    data = f.readlines()
    for line in range(0, len(data)):
        if data[line].startswith('>'):
            protein_name = data[line].lstrip('>').strip()
            protein.append(protein_name)
            protein_fasta[protein_name] = data[line+1].strip()
            # pritein_label[protein_name] = data[line+2].strip()
            labelline = data[line+2].strip()
            pritein_label[protein_name] = labelize(labelline, protein_name, bindtype)

    return protein, protein_fasta, pritein_label

def loadfeature_for_deeplearning(protein_list, fasta, PSSMpath, psipredpath, windowsize=17):
    list_length = len(protein_list)
    feature = []
    pssmfeature = []
    psifeature = []
    onehot = []
    pssm_psi = []
    pssm_onehot = []
    psi_onehot = []

    for i in range(list_length):
        protein = protein_list[i]
        eachfeature = featurecombine_for_deeplearning(protein, PSSMpath, psipredpath, fasta[protein],
                                                      windowsize=windowsize)
        each_featurecombination = seperatefeature(eachfeature, windowsize=windowsize)
        feature.extend(eachfeature)
        pssmfeature.extend(each_featurecombination[0])
        psifeature.extend(each_featurecombination[1])
        onehot.extend(each_featurecombination[2])
        pssm_psi.extend(each_featurecombination[3])
        pssm_onehot.extend(each_featurecombination[4])
        psi_onehot.extend(each_featurecombination[5])

    return feature, pssmfeature, psifeature, onehot, pssm_psi, pssm_onehot, psi_onehot


def savefile(filetosave, data):
    file = open(filetosave, 'wb')
    pickle.dump(data, file)
    file.close()

def get_subdict(keylist, dic):
    return dict([(key, dic[key]) for key in keylist])

def splitdata(class_id, class_dict):
    """
    :param class_id: key value[0,1,2,3,4,5,6]，string
    :param class_dict:  the index for each type of nucleic acid, dict
    :return:  the set of index after split original dataset
    """
    amount = len(class_dict[class_id])
    limit_count = int(round(amount / 6.0))
    output_index = []
    output_index.extend(class_dict[class_id])
    for key in class_dict.keys():
        if key != class_id:
            classindex_list = class_dict[key]
            class_num = len(classindex_list)
            random.seed(42)
            random.shuffle(classindex_list)
            if class_num > limit_count:
                output_index.extend([classindex_list[i] for i in range(limit_count)])
            else:
                output_index.extend([classindex_list[i] for i in range(class_num)])
    # print(output_index)
    return output_index

def get_classindex(label_list, class_id):
    """
     :param label_list: list
    :param class_id: int, [0,1,2,3,4,5,6]
    :return: A collection of index for each type of binding nucleic acid
    """
    class_index = []
    for i in range(len(label_list)):
        if label_list[i] == class_id:
            class_index.append(i)
    return class_index


def binary_label(label_list, class_id):
    output_label = []
    for i in range(len(label_list)):
        if label_list[i] == class_id:
            output_label.append(1)
        else:
            output_label.append(0)
    return output_label



if __name__ == '__main__':

    # the path you saved original .ss2 file generated from PSIPRED
    psipredpath = '~/feature/PSI'
    # the path you saved pssm.file file generated from PSI-BLAST
    PSSMpath= '~/feature/PSSM'
    # the path you saved original output file generated from HHblits
    ECOpath = '~/feature/ECO'

    train_ADNA = '~/data/train/ADNA_train.txt'
    train_BDNA = '~/data/train/BDNA_train.txt'
    train_ssDNA = '~/data/train/ssDNA_train.txt'
    train_mRNA = '~/data/train/mRNA_train.txt'
    train_tRNA = '~/data/train/tRNA_train.txt'
    train_rRNA = '~/data/train/rRNA_train.txt'
    train_nonbind = '~/data/train/nonbind_train.txt'

    print('start...')
    train_ADNA_protein, train_ADNA_fasta, train_ADNA_label = readtxt(train_ADNA, 'DNA')
    train_BDNA_protein, train_BDNA_fasta, train_BDNA_label = readtxt(train_BDNA, 'DNA')
    train_ssDNA_protein, train_ssDNA_fasta, train_ssDNA_label = readtxt(train_ssDNA, 'DNA')
    train_mRNA_protein, train_mRNA_fasta, train_mRNA_label = readtxt(train_mRNA, 'mRNA')
    train_tRNA_protein, train_tRNA_fasta, train_tRNA_label = readtxt(train_tRNA, 'tRNA')
    train_rRNA_protein, train_rRNA_fasta, train_rRNA_label = readtxt(train_rRNA, 'rRNA')
    train_nonbind_protein, train_nonbind_fasta, train_nonbind_label = readtxt(train_nonbind, 'nonbind')
    #所有训练蛋白质列表
    traindata_protein = train_ADNA_protein + train_BDNA_protein + train_ssDNA_protein + train_mRNA_protein + train_tRNA_protein + \
                        train_rRNA_protein + train_nonbind_protein

    traindata_fasta = {}
    traindata_fasta.update(train_ADNA_fasta)
    traindata_fasta.update(train_BDNA_fasta)
    traindata_fasta.update(train_ssDNA_fasta)
    traindata_fasta.update(train_mRNA_fasta)
    traindata_fasta.update(train_tRNA_fasta)
    traindata_fasta.update(train_rRNA_fasta)
    traindata_fasta.update(train_nonbind_fasta)

    traindata_label = {}
    traindata_label.update(train_ADNA_label)
    traindata_label.update(train_BDNA_label)
    traindata_label.update(train_ssDNA_label)
    traindata_label.update(train_mRNA_label)
    traindata_label.update(train_tRNA_label)
    traindata_label.update(train_rRNA_label)
    traindata_label.update(train_nonbind_label)

    traindata_labellist = []
    for key in traindata_label:
        traindata_labellist.extend(traindata_label[key])

    trainfeature, trainpssm, trainpsi, trainonehot, pssm_psi, pssm_onehot, psi_onehot = \
        loadfeature_for_deeplearning(traindata_protein, traindata_fasta, PSSMpath, psipredpath, windowsize=21)

    savefile('./feature/train/trainlabel_capsnet.pickle', traindata_labellist)
    savefile('./feature/train/pssm.pickle', trainpssm)
    savefile('./feature/train/pssm_psi.pickle', pssm_psi)
    savefile('./feature/train/pssm_onehot.pickle', pssm_onehot)
    savefile('./feature/train/trainfeature_capsnet.pickle', trainfeature)


    print("\nthe feature of training data has been finished!\n")
    
    count_list = {}
    class_id = ['0', '1', '2', '3', '4', '5', '6']
    for i in range(len(class_id)):
        count_list[class_id[i]] = get_classindex(traindata_labellist, eval(class_id[i]))
    
    print("Data Distribution：")
    print(count_list)
    
    # Features and labels for each type of binding nucleic acid
    pssm_eachclass = dict()
    pssmpsi_eachclass = dict()
    pssmonehot_eachclass = dict()
    traindata = dict()
    trainlabel = dict()
    
    for i in range(len(class_id)):
        if class_id[i] != '0':
            outputindex = splitdata(class_id[i], count_list)
            trainlabel[class_id[i]] = [traindata_labellist[index] for index in outputindex]
            traindata[class_id[i]] = [trainfeature[index] for index in outputindex]
            pssm_eachclass[class_id[i]] = [trainpssm[index] for index in outputindex]
            pssmpsi_eachclass[class_id[i]] = [pssm_psi[index] for index in outputindex]
            pssmonehot_eachclass[class_id[i]] = [pssm_onehot[index] for index in outputindex]
        else:
            output_index = []
            chosen_num = len(count_list['1']) + len(count_list['2']) + len(count_list['3']) + len(count_list['4']) + \
                         len(count_list['5']) + len(count_list['6'])
            class0_indexlist = count_list[class_id[i]]
            random.seed(42)
            random.shuffle(class0_indexlist)
            output_index.extend([class0_indexlist[index] for index in range(chosen_num)])
            for j in range(1, len(class_id)):
                output_index.extend(count_list[class_id[j]])
            trainlabel[class_id[i]] = [traindata_labellist[index] for index in output_index]
            traindata[class_id[i]] = [trainfeature[index] for index in output_index]
            pssm_eachclass[class_id[i]] = [trainpssm[index] for index in output_index]
            pssmpsi_eachclass[class_id[i]] = [pssm_psi[index] for index in output_index]
            pssmonehot_eachclass[class_id[i]] = [pssm_onehot[index] for index in output_index]
    
    
    savefile('./feature/train/pssm_eachclass.pickle', pssm_eachclass)
    savefile('./feature/train/trainlabel_capsnet_eachclass.pickle', trainlabel)
    savefile('./feature/train/pssmpsi_eachclass.pickle', pssmpsi_eachclass)
    savefile('./feature/train/pssmonehot_eachclass.pickle', pssmonehot_eachclass)
    savefile('./feature/train/trainfeature_capsnet_eachclass.pickle', traindata)

    # generate feature for test set or validation set
    test_ADNA = '~/data/test/ADNA_test.txt'
    test_BDNA = '~/data/test/BDNA_test.txt'
    test_ssDNA = '~/data/test/ssDNA_test.txt'
    test_mRNA = '~/data/test/mRNA_test.txt'
    test_tRNA = '~/data/test/tRNA_test.txt'
    test_rRNA = '~/data/test/rRNA_test.txt'
    test_nonbind = '~/data/test/nonbind_test.txt'

    print('test data preparing...')
    test_ADNA_protein, test_ADNA_fasta, test_ADNA_label = readtxt(test_ADNA, 'DNA')
    test_BDNA_protein, test_BDNA_fasta, test_BDNA_label = readtxt(test_BDNA, 'DNA')
    test_ssDNA_protein, test_ssDNA_fasta, test_ssDNA_label = readtxt(test_ssDNA, 'DNA')
    test_mRNA_protein, test_mRNA_fasta, test_mRNA_label = readtxt(test_mRNA, 'mRNA')
    test_tRNA_protein, test_tRNA_fasta, test_tRNA_label = readtxt(test_tRNA, 'tRNA')
    test_rRNA_protein, test_rRNA_fasta, test_rRNA_label = readtxt(test_rRNA, 'rRNA')
    test_nonbind_protein, test_nonbind_fasta, test_nonbind_label = readtxt(test_nonbind, 'nonbind')

    testdata_protein = test_ADNA_protein + test_BDNA_protein + test_ssDNA_protein + test_mRNA_protein + test_rRNA_protein + \
                       test_tRNA_protein + test_nonbind_protein

    testdata_fasta = {}
    testdata_fasta.update(test_ADNA_fasta)
    testdata_fasta.update(test_BDNA_fasta)
    testdata_fasta.update(test_ssDNA_fasta)
    testdata_fasta.update(test_mRNA_fasta)
    testdata_fasta.update(test_rRNA_fasta)
    testdata_fasta.update(test_tRNA_fasta)
    testdata_fasta.update(test_nonbind_fasta)

    testdata_label = {}
    testdata_label.update(test_ADNA_label)
    testdata_label.update(test_BDNA_label)
    testdata_label.update(test_ssDNA_label)
    testdata_label.update(test_mRNA_label)
    testdata_label.update(test_rRNA_label)
    testdata_label.update(test_tRNA_label)
    testdata_label.update(test_nonbind_label)

    print(testdata_fasta)
    print(testdata_label)
    testdata_labellist = []
    for key in testdata_label:
        testdata_labellist.extend(testdata_label[key])
    testdata_labelarray = np.array(testdata_labellist)
    testfeature, testpssm, testpsi, testonehot, test_pssmpsi, test_pssmonehot, test_psionehot = \
        loadfeature_for_deeplearning(testdata_protein, testdata_fasta, PSSMpath, psipredpath, windowsize=21)

    print(testfeature)
    print(np.array(testfeature).shape)
    print(testdata_labellist)

    savefile('./feature/test/testlabel_capsnet.pickle', testdata_labellist)
    savefile('./feature/test/testfeature_capsnet.pickle', testfeature)
  





