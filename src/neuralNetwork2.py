from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import csv
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD, Optimizer
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.ensemble import BaggingClassifier 
from sklearn.cross_validation import StratifiedKFold, KFold
path = '../data/raw/'
path_res = '../result/'
filename_train = ['train0.csv', 'train1.csv', 'train2.csv', 'train3.csv', 'train4.csv', 'train5.csv', 'train6.csv', 'train7.csv', 'train8.csv', 'train9.csv']
filename_test = ['test0.csv', 'test1.csv', 'test2.csv', 'test3.csv', 'test4.csv', 'test5.csv', 'test6.csv', 'test7.csv', 'test8.csv', 'test9.csv']
filename_labels = ['label0.csv', 'label1.csv', 'label2.csv', 'label3.csv', 'label4.csv', 'label5.csv', 'label6.csv', 'label7.csv', 'label8.csv', 'label9.csv']
file_num = 10

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def preprocess_labels(labels):
	lab = np.zeros(labels.shape[0])
	i = 0
	for l in labels:
		lab[i] = int(l[-1]) - 1
		i = i + 1
	return lab

def transfer_labels(lab):
	y = np.zeros((lab.shape[0], 9))
	i = 0
	for l in lab:
		yy = np.zeros(9)
		yy[l] = 1
		y[i] = yy
		i = i + 1
	print('y.shape[0]', y.shape[0])
	print('y.shape[1]', y.shape[1])
	return y

def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))


#The number of classes: 9 classes
nb_classes = 9
print(nb_classes, 'classes')


sample = pd.read_csv(path+'sampleSubmission.csv')

score_test = np.zeros(file_num)
res_total_file = path_res + 'res_nn2.txt'

for idx in range(0, 10):
	print('-----------------------------')
	print('Run the', idx + 1, 'th dataset...')
	pred_id_file = path_res + 'pred_nn2_dataset' + str(idx) + '.csv' 
	# load training dataset
	X, labels = load_data(path + filename_train[idx], train=True)
	X, scaler = preprocess_data(X)
	# The number of features: 93 dims
	dims = X.shape[1]
	print(dims, 'dims')
	lab = preprocess_labels(labels)
	y = transfer_labels(lab)
	# load testing dataset
	X_test, ids = load_data(path + filename_test[idx], train=False)
	X_test, _ = preprocess_data(X_test, scaler)
	# load the transfered labels for the testing dataset
	df = pd.read_csv(path + filename_labels[idx], header=None)
	y_labels = df.values.copy()
	y_test = transfer_labels(y_labels)
	# setting cross validation
	nfold = 5
	score = np.zeros(nfold)
	RND = np.random.randint(0, 70000, nfold)
	pred = np.zeros((X_test.shape[0],9))
	i = 0
	skf = StratifiedKFold(labels, nfold, random_state=1337)
	for tr, te in skf:
		X_train, X_valid, y_train, y_valid = X[tr], X[te], y[tr], y[te]
		predTr = np.zeros((X_valid.shape[0],9))
		n_bag = 1
		for j in range(n_bag):
			print('nfold: ',i,'/',nfold, ' n_bag: ',j,' /',n_bag)
			print("Building model...")
			model = Sequential()
			# Dense(512) is a fully-connected layer with 512 hidden units and the input data has 93 dims in the first layer
			model.add(Dense(512, input_shape=(dims,)))
			model.add(PReLU())
			model.add(BatchNormalization())
			model.add(Dropout(0.5))
			model.add(Dense(512))
			model.add(PReLU())
			model.add(BatchNormalization())
			model.add(Dropout(0.5))
			model.add(Dense(nb_classes))
			model.add(Activation('softmax'))
			ADAM=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
			sgd=SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
			model.compile(loss='categorical_crossentropy', optimizer="adam")
			print("Training model...")
			earlystopping=EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1)
			checkpointer = ModelCheckpoint(filepath = path+"tmp/weights.hdf5", verbose = 0, save_best_only = True)
			model.fit(X_train, y_train, nb_epoch = 30, batch_size = 128, verbose = 2, show_accuracy = True,
			validation_data=(X_valid,y_valid), callbacks=[earlystopping,checkpointer])
			model.load_weights(path+"tmp/weights.hdf5")
			print("Generating submission...")
			pred += model.predict_proba(X_test)
			predTr += model.predict_proba(X_valid)
		predTr /= n_bag
		#submissionTr.iloc[te] = predTr
		score[i] = log_loss(y_valid,predTr,eps=1e-15, normalize=True)
		print('score ', score[i])
		i += 1

	print("ave: "+ str(np.average(score)) + ", stddev: " + str(np.std(score)))
	pred /= (nfold * n_bag)
	score_test[idx] = log_loss(y_test, pred, eps = 1e-15, normalize = True)
	print('score_test[', idx, ']: ', score_test[idx])
	#write to file
	with open(res_total_file, 'a') as f:
		f.write('The ' + str(idx) + "th datasets\n")
		f.write('layer:' + str(2) + ', neurons: ' + str(512) + ', epoch: ' + str(30) + ', batch_size: ' + str(128) + '\n')
		f.write("ave: "+ str(np.average(score)) + ", stddev: " + str(np.std(score)) + '\n')
		f.write('score_test[' + str(idx) + ']: ' + str(score_test[idx]) + '\n')

	my_df = pd.DataFrame(pred)
	my_df.to_csv(pred_id_file, index = False, header = True)


for idx in range(0, 10):
	print('score_test[', idx, ']: ', score_test[idx])

#make_submission(pred, ids, encoder, fname=path+'kerasNN2.csv')
#print(log_loss(labels,submissionTr.values,eps=1e-15, normalize=True))
#submissionTr.to_csv(path+"kerasNN2_retrain.csv",index_label='id')
	

# nfold 3, bagging  5: 0.4800704 + 0.005194
# nfold 3, bagging 10: 0.4764856 + 0.0060724
# nfold 5, bagging  5: 0.470483 + 0.011645
# nfold 5, bagging 10: 0.468049 + 0.0118616
# nfold 8, bagging 10: 0.469461 + 0.0100765
# tsne, nfold 5, bagging 5: 0.474645 + 0.0109076

