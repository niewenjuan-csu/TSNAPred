import numpy as np
from keras import layers, models, optimizers, callbacks
from keras import backend as K
from keras.utils import to_categorical
from CapsLayer import CapsuleLayer, PrimaryCap, Length
import argparse
import pickle
from sklearn import metrics
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier



K.set_image_data_format('channels_last')

class roc_callback(callbacks.Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        self.auc = {'epoch': []}
        self.auc_val = {'epoch': []}
        self.loss = {'epoch': []}
        self.val_loss = {'epoch': []}
        self.accuracy = {'epoch': []}
        self.val_accuracy = {'epoch': []}
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = metrics.roc_auc_score(self.y, y_pred)

        y_pred_val = self.model.predict(self.x_val)
        roc_val = metrics.roc_auc_score(self.y_val, y_pred_val)
        
        val_targ = np.argmax(self.y_val, 1)
        val_predict = np.argmax(y_pred_val, 1)

        _val_f1 = metrics.f1_score(val_targ, val_predict)
        _val_recall = metrics.recall_score(val_targ, val_predict)
        _val_precision = metrics.precision_score(val_targ, val_predict)

        print('='*60)
        print(" — val_f1: %s — val_precision: %s — val_recall: %s" % (str(round(_val_f1, 4)),
                                                                      str(round(_val_precision, 4)),
                                                                      str(round(_val_recall, 4))))

        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val, 4))))
        print('\n')

        self.loss['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_accuracy['epoch'].append(logs.get('val_acc'))
        self.auc['epoch'].append(roc)
        self.auc_val['epoch'].append(roc_val)
        # self.f1_value['epoch'].append(_val_f1)
        # # target_name = ['non-binding', 'ADNA binding', 'BDNA binding', 'ssDNA binding', 'mRNA binding', 'tRNA binding', 'rRNA binding']
        # print(metrics.classification_report(val_targ, val_predict))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def loss_plot(self, loss_type):
        iters = range(len(self.loss[loss_type]))

        plt.figure()
        # auc
        plt.plot(iters, self.auc[loss_type], 'b', label='train auc')
        plt.plot(iters, self.auc_val[loss_type], 'r', label='valid auc')
        plt.xlabel(loss_type)
        plt.ylabel('auc')
        plt.legend(loc="upper right")
        plt.show()

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def CaspNet(input_shape, n_class, num_routing):

    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    # Batch Normalization Layer
    conv1 = layers.BatchNormalization()(conv1)
    # Dropout Layer
    conv1 = layers.Dropout(0.7)(conv1)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    primarycaps = layers.BatchNormalization()(primarycaps)
    primarycaps = layers.Dropout(0.2)(primarycaps)

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, num_routing=num_routing,
                             name='digitcaps')(primarycaps)
    digitcaps = layers.BatchNormalization()(digitcaps)
    digitcaps = layers.Dropout(0.1)(digitcaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)
    model = models.Model(inputs=x, outputs=out_caps)

    return model


def binary_label(labellist, class_id):
    output_label = []
    for i in range(len(labellist)):
        if labellist[i] == class_id:
            output_label.append(1)
        else:
            output_label.append(0)
    return output_label



if __name__ == '__main__':
    batch_size = 32
    num_classes = 2
    lr = 0.01

    traindata_pickle = open('./feature/train/trainfeature_capsnet_eachclass.pickle',
                            'rb')
    traindata = pickle.load(traindata_pickle)

    trainlabel_pickle = open('./feature/train/trainlabel_capsnet_eachclass.pickle',
                             'rb')
    trainlabel = pickle.load(trainlabel_pickle)

    ADNA_traindata = traindata['1']
    ADNA_trainlabel = binary_label(trainlabel['1'], 1)
    x_train, x_test, y_train, y_test = train_test_split(ADNA_traindata, ADNA_trainlabel, test_size=0.2, shuffle=True,
                                                        random_state=42)
    x_train = np.array(x_train).reshape(-1, 19, 30, 1)
    x_test = np.array(x_test).reshape(-1, 19, 30, 1)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)


    """"
    # grid search
    X = np.array(ADNA_traindata).reshape(-1, 19, 30, 1)
    Y = np.array(ADNA_trainlabel)
    Y = np_utils.to_categorical(Y, 2)
    model = KerasClassifier(build_fn=create_model, batch_size=16, epochs=100,  verbose=1)
    batch_size = [16, 32, 64, 128, 256]
    epochs = [50, 100, 150, 200]
    param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
    lr = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    param_grid = dict(lr=lr)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(X, Y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for means, stds, params in zip(means, stds, params):
        print("%f (%f) with %r" %(means, stds, params))
    """

    model = CaspNet(input_shape=(19, 30, 1), n_class=2, num_routing=3)
    model.summary()
    model.compile(loss=margin_loss, optimizer=optimizers.Adam(lr=lr), loss_weights=[1.], metrics=['accuracy'])
    checkpoint = callbacks.ModelCheckpoint(
        './model_21/ADNA/weights-{epoch:03d}-{val_acc:.4f}.h5',
        monitor='val_acc', save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (0.95 ** epoch))
    roccallback = roc_callback(training_data=(x_train, y_train), validation_data=(x_test, y_test))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=100, shuffle=True, validation_data=(x_test, y_test),
                   callbacks=[checkpoint, lr_decay, roccallback])
    roccallback.loss_plot('epoch')





