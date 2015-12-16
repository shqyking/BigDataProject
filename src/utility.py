import csv
import random
import os

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




















