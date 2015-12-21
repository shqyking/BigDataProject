import csv
import random
import os
import numpy as np
import pandas as pd
import gc
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit

'''
Generate test set IDs. It only write down the row ID of testing data
sample call: generate_test_id('data/raw/train.csv', 'data/generated', 0.1, 10)
'''
def generate_test_id(input_file, output_folder, test_percentage, num_copies):
	#row_count is the number of rows in filename. However, in csv file, the first row is column information. Ingore the first row
	row_count = count_num_of_lines(input_file)
	for i in range(num_copies):
		test_id_list = random.sample(xrange(1, row_count), int((row_count - 1)* test_percentage))
		output_file = os.path.join(output_folder, 'testID' + str(i) + '.txt')
		with open(output_file, 'w') as f:
			for test_id in test_id_list:
				f.write(str(test_id) + '\n')

'''
Get number of rows of input_file
'''
def count_num_of_lines(input_file):
	with open(input_file, 'r') as f:
		row_count = sum(1 for row in f)
	return row_count

'''
Generate training and testing data from testID files
It produce 3 kind of files, train.csv, test.csv and label.csv
sample call: generate_train_test_data('data/raw', 'data/generated')
'''
def generate_train_test_data(raw_folder, test_id_folder):
	for i in range(10):
		test_id_file = os.path.join(test_id_folder, 'testID' + str(i) + '.txt')
		raw_train_file = os.path.join(raw_folder, 'train.csv')
		with open(test_id_file, 'r') as f:
			test_ids = f.read().splitlines()
		test_ids = set(test_ids)
		train_splited = []
		test_splited = []
		test_labels = []
		with open(raw_train_file) as f:
			contents = f.read().splitlines()
			header = contents[0].split(',')
			for j in range(1, len(contents)):
				line_content = contents[j].split(',')
				if line_content[0] not in test_ids:
					train_splited.append(line_content)
				else:
					test_splited.append(line_content[:-1])
					test_labels.append([int(line_content[-1][-1]) - 1])
		train_splited_file = os.path.join(raw_folder, 'train' + str(i) + '.csv')
		test_splited_file = os.path.join(raw_folder, 'test' + str(i) + '.csv')
		train_labels_file = os.path.join(raw_folder, 'label' + str(i) + '.csv')
		with open(train_splited_file, 'w') as f:
			a = csv.writer(f)
			f.write(','.join(header) + '\n')
			a.writerows(train_splited)
		with open(test_splited_file, 'w') as f:
			a = csv.writer(f)
			f.write(','.join(header[:-1]) + '\n')
			a.writerows(test_splited)
		with open(train_labels_file, 'w') as f:
			a = csv.writer(f, delimiter=',')
			a.writerows(test_labels)

def load_data(train_file, test_file):
    train = pd.read_csv(train_file)
    train_labels = [int(v[-1])-1 for v in train.target.values]
    train = train.drop('id', axis=1)
    train = train.drop('target', axis=1)

    test = pd.read_csv(test_file)
    test = test.drop('id', axis=1)

    return np.array(train, dtype=float), np.array(train_labels), np.array(test, dtype=float)

def save_prediction(output_file, predictions):
    submission = pd.DataFrame(predictions)
    submission.to_csv(output_file, index=False)

def get_cv(y, n_folds=5):
    return StratifiedKFold(y, n_folds=n_folds, random_state=23)

def calculate_logloss(prediction_file, label_file):
	y_true = []
	with open(label_file, 'r') as f:
		for line in f:
			y_true.append(int(line))
	y_predict = []
	with open(prediction_file, 'r') as f:
		tmp = f.readline()
		for line in f:
			y_predict.append([float(x) for x in line.split(',')])
	return log_loss(y_true, y_predict)


def make_blender_cv(classifier, x, y, calibrate=False):
    skf = StratifiedKFold(y, n_folds=5, random_state=23)
    scores, predictions = [], None
    for train_index, test_index in skf:
        if calibrate:
            # Make training and calibration
            calibrated_classifier = CalibratedClassifierCV(classifier, method='isotonic', cv=get_cv(y[train_index]))
            fitted_classifier = calibrated_classifier.fit(x[train_index, :], y[train_index])
        else:
            fitted_classifier = classifier.fit(x[train_index, :], y[train_index])
        preds = fitted_classifier.predict_proba(x[test_index, :])

        # Free memory
        calibrated_classifier, fitted_classifier = None, None
        gc.collect()

        scores.append(log_loss(y[test_index], preds))
        predictions = np.append(predictions, preds, axis=0) if predictions is not None else preds
    return scores, predictions

def get_xgboost_score():
	scores = []
	for i in range(10):
		pred_file = os.path.join('data', 'prediction', 'xgboost' + str(i) + '.csv')
		label_file = os.path.join('data', 'raw', 'label' + str(i) + '.csv')
		scores.append(calculate_logloss(pred_file, label_file))
	print "Average score of xgboost is " + str(np.mean(scores))
	print "Standard deviation of xgboost is " + str(np.std(scores))

def get_neural_network_score():
	scores = []
	for i in range(10):
		pred_file = os.path.join('data', 'prediction', 'pred_nn2_dataset' + str(i) + '.csv')
		label_file = os.path.join('data', 'raw', 'label' + str(i) + '.csv')
		scores.append(calculate_logloss(pred_file, label_file))
	print "Average score of neural network is " + str(np.mean(scores))
	print "Standard deviation of neural network is " + str(np.std(scores))

def get_random_forest_score():
	scores = []
	for i in range(10):
		pred_file = os.path.join('data', 'prediction', 'calibratedRandomForest' + str(i) + '.csv')
		label_file = os.path.join('data', 'raw', 'label' + str(i) + '.csv')
		scores.append(calculate_logloss(pred_file, label_file))
	print "Average score of calibrated random forest is " + str(np.mean(scores))
	print "Standard deviation of calibrated random forest is " + str(np.std(scores))

def get_commbination_score():
	scores = []
	for i in range(10):
		pred_file = os.path.join('data', 'prediction', 'com' + str(i) + '.csv')
		label_file = os.path.join('data', 'raw', 'label' + str(i) + '.csv')
		scores.append(calculate_logloss(pred_file, label_file))
	print "Average score of combination is " + str(np.mean(scores))
	print "Standard deviation of combination is " + str(np.std(scores))


# get_xgboost_score()
# get_neural_network_score()
# get_random_forest_score()
# get_commbination_score()













