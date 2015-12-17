import pandas as pd
from numpy import genfromtxt, savetxt
from sklearn import ensemble
from sklearn.metrics import log_loss

def read_id_from_file(input_file):
	id_set = set()
	with open(input_file, 'r') as f:
		for number in f:
			id_set.add(number.split()[0])
	return id_set

def read_test_ids(num_copies, test):
	for i in range(num_copies):
		test.append(read_id_from_file("data/generated/testID" + str(i) + ".txt"))

def train_classifier(dataset, test_ids):
	train = []
	test = []
	train_target = []
	test_target = []
	
	for record in dataset:
		if str(record[0]) in test_ids:
			test.append(record[1:-1])
			test_target.append(record[-1])
		else:
			train.append(record[1:-1])
			train_target.append(record[-1])
	
	rf = ensemble.RandomForestClassifier(n_estimators = 1000, max_features = 20, n_jobs = -1)
	rf.fit(train, train_target)

	return train, train_target, test, test_target, rf
	
if __name__ == "__main__":
	dataset = genfromtxt(open('data/raw/train.csv','r'), delimiter=',', dtype=None)[1:]	

	test_ids = []
	read_test_ids(10, test_ids)
	
	predict_score = [0.0] * 10
	avg = 0

	for i in range(10):
		train, train_target, test, test_target, rf = train_classifier(dataset, test_ids[i])
		predicted = rf.predict_proba(test)
		
		predict_score[i] = log_loss(test_target, predicted, eps=1e-15, normalize=True)
		savetxt('data/prediction/randomForest' + str(i) + '.csv', predicted, delimiter=',', fmt='%f', header="0,1,2,3,4,5,6,7,8", comments='')
		avg += predict_score[i]

	print predict_score
	avg = avg / float(10)
	print avg

