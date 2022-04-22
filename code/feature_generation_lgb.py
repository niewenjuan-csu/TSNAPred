import numpy as np
import os
import linecache
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pickle
import random

# Normalize numeric values to [0,1]
def normalize(value):
    a = 1 + math.exp(value)
    b = 1 / a
    return b

def loadRAAPdic(RAAPpath):
    filelines = linecache.getlines(RAAPpath)
    RAAPdic = {}
    newRAAPdic = {}
    for i in range(2, len(filelines)):
        line = filelines[i]
        aa = line.split()[0]
        RAAPdic[aa] = []
        newRAAPdic[aa] = []
        for j in range(1, 7):
            RAAPdic[aa].append(float(line.split()[j]))
    for n in range(6):
        col_array = []
        for key in RAAPdic.keys():
            value = RAAPdic[key][n]
            col_array.append(value)
        minvalue = np.min(col_array)
        maxvalue = np.max(col_array)
        i = 0
        for key in newRAAPdic.keys():
            if i >= 20:
                break
            else:
                newRAAPdic[key].append(float((col_array[i]-minvalue)/(maxvalue-minvalue)))
            i = i+1
    return newRAAPdic

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

def RSAfeature(RSASpath, uniport_id):
    rsaDic = {}
    filelines = linecache.getlines(RSASpath + '/' + 'asaq.' + uniport_id + '.fasta' + '/' + 'asaq.pred')
    length = len(filelines)
    for i in range(length-1):
        residuenum = int(filelines[i].split()[0]) - 1
        rsaDic[residuenum] = []
        rsaDic[residuenum].append(normalize(float(filelines[i].split()[2])))
        rsaDic[residuenum].append(normalize(float(filelines[i].split()[3])))
        # rsaDic[residuenum].append(normalize(float(filelines[i].split()[3])/float(filelines[i].split()[2])))
        flag = 1 if (float(filelines[i].split()[3])/float(filelines[i].split()[2])) >= 0.25 else 0
        rsaDic[residuenum].append(flag)
    # print(rsaDic)
    return rsaDic

def Disorderfeature(Disorderpath, uniport_id):
    DisorderDic = {}
    filelines = linecache.getlines(Disorderpath + '/' + uniport_id + '.txt')
    length = len(filelines)
    for i in range(7, length):
        residuenum = int(filelines[i].split()[0]) - 1
        DisorderDic[residuenum] = []
        DisorderDic[residuenum].append(float(filelines[i].split()[2]))
    # print(DisorderDic)
    return DisorderDic

def AAindex(uniport_id, fasta_line):
    """
    each residues: [hydrophobicity_1, hydrophobicity_2, polarity_1, polarity_2, positive charge, negative charge]
    """
    AAindexdic = {}
    length = len(fasta_line)

    for i in range(length):
        residuename = fasta_line[i]
        if residuename == 'A':
            # [normalize(0.02), normalize(0.25), normalize(8.1), normalize(0), 0, 0]
            AAindexdic[i] = [0.3932, 0.8072, 0.3950, 0.0, 0, 0]
        elif residuename == 'R':
            # [normalize(-0.42), normalize(-1.76), normalize(10.5), normalize(52), 1, 0]
            AAindexdic[i] = [0.2440, 0.0, 0.6913, 1.0, 1, 0]
        elif residuename == 'N':
            # [normalize(-0.77), normalize(-0.64), normalize(11.6), normalize(3.38), 0, 0]
            AAindexdic[i] = [0.1254, 0.4497, 0.8271, 0.0650, 0, 0]
        elif residuename == 'D':
            # [normalize(-0.14), normalize(-0.72), normalize(13), normalize(49.7), 0, 1]
            AAindexdic[i] = [0.0338, 0.4176, 1.0, 0.9557, 0, 1]
        elif residuename == 'C':
            # [normalize(0.77), normalize(0.04), normalize(5.5), normalize(1.48), 0, 0]
            AAindexdic[i] = [0.6474, 0.7228, 0.0740, 0.0284, 0, 0]
        elif residuename == 'Q':
            # [normalize(-1.1), normalize(-0.69), normalize(10.5), normalize(3.53), 0, 0]
            AAindexdic[i] = [0.0135, 0.4297, 0.6913, 0.0678, 0, 0]
        elif residuename == 'E':
            # [normalize(-1.14), normalize(-0.62), normalize(12.3), normalize(49.9), 0, 1]
            AAindexdic[i] = [0.0, 0.4578, 0.9135, 0.9596, 0, 1]
        elif residuename == 'G':
            # [normalize(-0.8), normalize(0.16), normalize(9), normalize(0), 0, 0]
            AAindexdic[i] = [0.1152, 0.7710, 0.5061, 0.0, 0, 0]
        elif residuename == 'H':
            # [normalize(0.26), normalize(-0.4), normalize(10.4), normalize(51.6), 1, 0]
            AAindexdic[i] = [0.4745, 0.5461, 0.6790, 0.9923, 1, 0]
        elif residuename == 'I':
            # [normalize(1.81), normalize(0.73), normalize(5.2), normalize(0.13), 0, 0]
            AAindexdic[i] = [1.0, 1.0, 0.0370, 0.0025, 0, 0]
        elif residuename == 'L':
            # [normalize(1.14), normalize(0.53), normalize(4.9), normalize(0.13), 0, 0]
            AAindexdic[i] = [0.7728, 0.9196, 0.0, 0.0025, 0, 0]
        elif residuename == 'K':
            # [normalize(-0.41), normalize(-1.1), normalize(11.3), normalize(49.5), 1, 0]
            AAindexdic[i] = [0.2474, 0.2650, 0.7901, 0.9519, 1, 0]
        elif residuename == 'M':
            # [normalize(1), normalize(0.26), normalize(5.7), normalize(1.43), 0, 0]
            AAindexdic[i] = [0.7254, 0.8112, 0.0987, 0.0275, 0, 0]
        elif residuename == 'F':
            # [normalize(1.35), normalize(0.61), normalize(5.2), normalize(0.35), 0, 0]
            AAindexdic[i] = [0.8440, 0.9518, 0.0370, 0.0067, 0, 0]
        elif residuename == 'P':
            # [normalize(-0.09), normalize(-0.07), normalize(8), normalize(1.58), 0, 0]
            AAindexdic[i] = [0.3559, 0.6787, 0.3872, 0.0303, 0, 0]
        elif residuename == 'S':
            # [normalize(-0.97), normalize(-0.26), normalize(9.2), normalize(1.67), 0, 0]
            AAindexdic[i] = [0.0576, 0.6024, 0.5308, 0.0321, 0, 0]
        elif residuename == 'T':
            # [normalize(-0.77), normalize(-0.18), normalize(8.6), normalize(1.66), 0, 0]
            AAindexdic[i] = [0.1254, 0.6345, 0.4567, 0.0319, 0, 0]
        elif residuename == 'W':
            # [normalize(1.71), normalize(0.37), normalize(5.4), normalize(2.1), 0, 0]
            AAindexdic[i] = [0.9661, 0.8554, 0.0617, 0.0403, 0, 0]
        elif residuename == 'Y':
            # [normalize(1.11), normalize(0.02), normalize(6.2), normalize(1.61), 0, 0]
            AAindexdic[i] = [0.7627, 0.7148, 0.1604, 0.0309, 0, 0]
        elif residuename == 'V':
            # [normalize(1.13), normalize(0.54), normalize(5.9), normalize(0.13), 0, 0]
            AAindexdic[i] = [0.7694, 0.9236, 0.1234, 0.0025, 0, 0]
    # print(AAindexdic)
    return AAindexdic


def ECOfeature(uniport_id, ECOpath):
    ECODic = {}
    ecofilelines = linecache.getlines(ECOpath + '/' + uniport_id + '.txt')
    length = len(ecofilelines)
    for i in range(length):
        residuePosition = int(ecofilelines[i].split()[0]) - 1
        ECODic[residuePosition] = []
        ECODic[residuePosition].append(float(ecofilelines[i].split()[2]))
    return ECODic

def RAAPfeature(uniport_id, fastaline, RAAPdic):
    RAAPfeaturedic = {}
    length = len(fastaline)
    for i in range(length):
        residuename = fastaline[i]
        RAAPfeaturedic[i] = RAAPdic[residuename]
    return RAAPfeaturedic

def appendzero(windowsize, featureDic):
    # print(featureDic)
    seqlength = len(featureDic.keys())
    appendnum = int((windowsize + 1) / 2)
    for i in range(1, appendnum):
        featureDic[0 - i] = []
        featureDic[seqlength - 1 + i] = []
        for a in range(20):
            featureDic[0 - i].append(0)
        for b in range(20):
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
                combineDic[i].append(each)
    featurelist = []
    for i in range(0, sequencelength):
        featurelist.append(combineDic[i])
    # print(featurelist)
    return featurelist

# for one protein
def featurecombine(uniport_id,psipredoutpath, RSApath, Disorderpath, ECOpath, RAAPdic, fastaline, windowsize=17):
    psipreddic = psipredfeature(psipredoutpath, uniport_id)
    rsadic = RSAfeature(RSApath, uniport_id)
    disorderdic = Disorderfeature(Disorderpath, uniport_id)
    aaindexdic = AAindex(uniport_id, fastaline)
    ecodic = ECOfeature(uniport_id, ECOpath)
    raapdic = RAAPfeature(uniport_id, fastaline, RAAPdic)

    length = len(psipreddic.keys())
    featuredic = {}
    for i in range(length):
        featuredic[i] = []
        for each in psipreddic[i]:
            featuredic[i].append(each)
        for each in rsadic[i]:
            featuredic[i].append(each)
        for each in ecodic[i]:
            featuredic[i].append(each)
        for each in disorderdic[i]:
            featuredic[i].append(each)
        for each in aaindexdic[i]:
            featuredic[i].append(each)
        for each in raapdic[i]:
            featuredic[i].append(each)
    print(featuredic)
    appendedfeaturedic = appendzero(windowsize, featuredic)
    combinefeaturelist = combine(length, appendedfeaturedic, windowsize)
    return combinefeaturelist

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

def dictolist(dic):
    newlist = []
    for key in dic:
        newlist.extend(dic[key])
    return newlist

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


def loadfeature(protein_list, fasta, psipredpath, RSApath, Disorderpath, ECOpath, RAAPdic, windowsize=17):
    list_length = len(protein_list)

    feature = []

    for i in range(list_length):
        protein = protein_list[i]
        print(protein)
        eachfeature_list = featurecombine(protein, psipredpath, RSApath, Disorderpath, ECOpath, RAAPdic,
                                          fasta[protein], windowsize=windowsize)
        feature.extend(eachfeature_list)

    print(feature)
    return feature


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
    random.seed(42)
    random.shuffle(output_index)
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
    # RAAP file can be downloaded from our project
    RAAPpath = '~/feature/RAAP.txt'
    RAAPdic = loadRAAPdic(RAAPpath)

    # the path you saved original .ss2 file generated from PSIPRED
    psipredpath = '~/feature/PSI'
    # the path you saved original rasaq.pred file generated from ASAquick
    RSApath = '~/feature/RSA'
    # the path you saved original output file generated from IUPred2A
    Disorderpath = '~/feature/Disorder'
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

    trainfeature = loadfeature(traindata_protein, traindata_fasta, psipredpath, RSApath,
                               Disorderpath, ECOpath, RAAPdic, windowsize=21)


    count_list = {}
    class_id = ['0', '1', '2', '3', '4', '5', '6']
    for i in range(len(class_id)):
        count_list[class_id[i]] = get_classindex(traindata_labellist, eval(class_id[i]))

    print("Data Distribution：")
    print(count_list)
    
    # generate the balance training subset for each type of binding nucleic acid
    traindata = {}
    trainlabel = {}
    
    for i in range(len(class_id)):
        if class_id[i] != '0':
            outputindex = splitdata(class_id[i], count_list)
            trainlabel[class_id[i]] = [traindata_labellist[index] for index in outputindex]
            traindata[class_id[i]] = [trainfeature[index] for index in outputindex]
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
            random.seed(42)
            random.shuffle(output_index)
            trainlabel[class_id[i]] = [traindata_labellist[index] for index in output_index]
            traindata[class_id[i]] = [trainfeature[index] for index in output_index]

    # the file path need to change where you want to save.
    savefile('./feature/train/trainfeature_lgb.pickle', trainfeature)
    savefile('./feature/train/trainlabel_lgb.pickle', traindata_labellist)
    savefile('./feature/train/trainfeature_lgb_eachclass.pickle', traindata)
    savefile('./feature/train/trainlabel_lgb_eachclass.pickle', trainlabel)

    
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

    testfeature = loadfeature(testdata_protein, testdata_fasta, psipredpath, RSApath,
                               Disorderpath, ECOpath, RAAPdic, windowsize=21)

    print(testfeature)
    print(testdata_labellist)

    # the file path need to change where you want to save.
    savefile('./feature/test/testfeature_lgb.pickle', testfeature)
    savefile('./feature/test/testlabel_lgb.pickle', testdata_labellist)
  






