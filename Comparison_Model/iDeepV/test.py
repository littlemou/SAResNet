import os
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten  # , Merge
from keras.layers import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
from load_data import load_data

def set_cnn_embed(embedding_weights, nb_filter=16):
    # nb_filter = 64
    filter_length = 10
    dropout = 0.5
    model = Sequential()
    # pdb.set_trace()
    model.add(Embedding(input_dim=embedding_weights.shape[0], output_dim=embedding_weights.shape[1],
                        weights=[embedding_weights], input_length=404, trainable=True))
    print('after embed', model.output_shape)
    model.add(Convolution1D(nb_filter, filter_length, border_mode='valid', init='glorot_normal'))
    model.add(Activation(LeakyReLU(.3)))
    model.add(MaxPooling1D(pool_length=3))
    model.add(Dropout(dropout))

    return model


def get_cnn_network_graphprot(embedding_rna_weights, rna_len=501, nb_filter=16):
    model = set_cnn_embed(embedding_rna_weights, nb_filter=nb_filter)

    model.add(Flatten())
    model.add(Dense(nb_filter * 50, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(nb_filter * 10, activation='sigmoid'))
    model.add(Dropout(0.50))
    print(model.output_shape)

    return model


def run_network(model, training, testing, y, validation, val_y):
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)sgd)#'
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # pdb.set_trace()
    print('model training')
    # checkpointer = ModelCheckpoint(filepath="models/bestmodel.hdf5", verbose=0, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    model.fit(training, y, batch_size=100, nb_epoch=10, verbose=0, validation_data=(validation, val_y),
              callbacks=[earlystopper])

    predictions = model.predict_proba(testing)[:, 1]
    return predictions, model


def calculate_auc(net, train, train_y, test, true_y, validation, val_y):
    predict, model = run_network(net, train, test, train_y, validation, val_y)
    # pdb.set_trace()
    auc = roc_auc_score(true_y, predict)

    print("Test AUC: ", auc)

    return auc, predict


def train2():
    data_dir = r"D:\Python\PyCharm\workSpace\SAResNet_tensorflow2\DataSet\150_ChIP-seq_Datasets_addValid"
    seq_hid = 16
    seq_net = get_cnn_network_graphprot(rna_len=496, nb_filter=seq_hid)  # 根据pickle生成cnn-model


if __name__ == "__main__":
    trainData_path = r"D:\Python\PyCharm\workSpace\SAResNet_tensorflow2\DataSet\150_ChIP-seq_Datasets_addValid"
    num=0
    for file in os.listdir(trainData_path):
        num += 1
        # if(num<106):
        #     continue
        print(str(num) + "]当前file：" + file)
        now_trainData_Path = os.path.join(trainData_path, file)
        (train, train_label), (test, test_label), (valid, valid_label), embedding_matrix = load_data(now_trainData_Path)
        net = get_cnn_network_graphprot(embedding_matrix, rna_len=404)
        predict, model = run_network(net, train, train_label, test, test_label, valid, valid_label)
        auc = roc_auc_score(test_label, predict)