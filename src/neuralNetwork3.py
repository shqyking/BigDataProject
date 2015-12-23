import numpy as np
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, Optimizer
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.ensemble import BaggingClassifier 
from sklearn.cross_validation import StratifiedKFold, KFold
path = '../data/raw/'
path_res = '../data/prediction/'
filename_train = ['train0.csv', 'train1.csv', 'train2.csv', 'train3.csv', 'train4.csv', 'train5.csv', 'train6.csv', 'train7.csv', 'train8.csv', 'train9.csv']
filename_test = ['test0.csv', 'test1.csv', 'test2.csv', 'test3.csv', 'test4.csv', 'test5.csv', 'test6.csv', 'test7.csv', 'test8.csv', 'test9.csv']
filename_labels = ['label0.csv', 'label1.csv', 'label2.csv', 'label3.csv', 'label4.csv', 'label5.csv', 'label6.csv', 'label7.csv', 'label8.csv', 'label9.csv']
file_num = 10
np.random.seed(1111)

def load_dataset(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids

def preprocess_dataset(X, scaler=None):
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
	return y

#The number of classes: 9 classes
nb_classes = 9

score_test = np.zeros(file_num)
res_total_file = path_res + 'res_nn3.txt'

for idx in range(0, 10):
	print'Run the ' + str(idx + 1) + 'th dataset...'
	pred_id_file = path_res + 'pred_nn3_dataset' + str(idx) + '.csv' 
	# load training dataset
	X, labels = load_dataset(path + filename_train[idx], train=True)
	X, scaler = preprocess_dataset(X)
	# The number of features: 93 dims
	dims = X.shape[1]
	lab = preprocess_labels(labels)
	y = transfer_labels(lab)
	# load testing dataset
	X_test, ids = load_dataset(path + filename_test[idx], train=False)
	X_test, _ = preprocess_dataset(X_test, scaler)
	# load the transfered labels for the testing dataset
	df = pd.read_csv(path + filename_labels[idx], header=None)
	y_labels = df.values.copy()
	y_test = transfer_labels(y_labels)
	# setting cross validation
	nfold = 5
	score = np.zeros(nfold)
	RND = np.random.randint(0, 70000, nfold)
	pred_test = np.zeros((X_test.shape[0],9))
	i = 0
	randomDatasets = StratifiedKFold(labels, nfold, random_state = 1111)
	for train, test in randomDatasets:
		X_train, X_valid, y_train, y_valid = X[train], X[test], y[train], y[test]
		pred_train = np.zeros((X_valid.shape[0],9))
		n_bag = 1
		for j in range(n_bag):
			print 'nfold: ' + str(i) + '/' + str(nfold) + ' n_bag: ' + str(j) + ' /' + str(n_bag)
			model = Sequential()
			# Dense(512) is a fully-connected layer with 512 hidden units and the input data has 93 dims in the first layer
			model.add(Dense(512, input_shape = (dims,)))
			model.add(PReLU())
			model.add(BatchNormalization())
			model.add(Dropout(0.5))
			model.add(Dense(512, input_shape = (dims,)))
			model.add(PReLU())
			model.add(BatchNormalization())
			model.add(Dropout(0.5))
			model.add(Dense(512))
			model.add(PReLU())
			model.add(BatchNormalization())
			model.add(Dropout(0.5))
			model.add(Dense(nb_classes))
			model.add(Activation('softmax'))
			ADAM = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-6)
			model.compile(loss = 'categorical_crossentropy', optimizer = "adam")
			print 'Training model...'
			earlystopping=EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1)
			checkpointer = ModelCheckpoint(filepath = path+"tmp/weights.hdf5", verbose = 0, save_best_only = True)
			model.fit(X_train, y_train, nb_epoch = 30, batch_size = 128, verbose = 2, show_accuracy = True,
			validation_data=(X_valid,y_valid), callbacks=[earlystopping,checkpointer])
			model.load_weights(path + "tmp/weights.hdf5")
			pred_test += model.predict_proba(X_test)
			pred_train += model.predict_proba(X_valid)
		pred_train /= n_bag
		score[i] = log_loss(y_valid,pred_train,eps=1e-12, normalize=True)
		print 'score ' + str(score[i])
		i += 1

	print 'ave: '+ str(np.average(score)) + ', stddev: ' + str(np.std(score))
	pred_test /= (nfold * n_bag)
	score_test[idx] = log_loss(y_test, pred_test, eps = 1e-12, normalize = True)
	print 'score_test[' + str(idx) + ']: ' + str(score_test[idx])
	#write to file
	with open(res_total_file, 'a') as f:
		f.write('The ' + str(idx) + "th datasets\n")
		f.write('layer:' + str(2) + ', neurons: ' + str(512) + ', epoch: ' + str(30) + ', batch_size: ' + str(128) + '\n')
		f.write("ave: "+ str(np.average(score)) + ", stddev: " + str(np.std(score)) + '\n')
		f.write('score_test[' + str(idx) + ']: ' + str(score_test[idx]) + '\n')

	my_df = pd.DataFrame(pred_test)
	my_df.to_csv(pred_id_file, index = False, header = True)


for idx in range(0, 10):
	print('score_test[', idx, ']: ', score_test[idx])

