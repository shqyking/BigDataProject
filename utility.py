import csv
import random

'''
Splits the input_file to testing and training set. It only write down the row ID of testing data
An sample call: split_to_train_test('data/raw/train.csv', 'data/generated', 0.1, 10)
'''
def split_to_train_test(input_file, output_folder, test_percentage, num_copies):
	#row_count is the number of rows in filename. However, in csv file, the first row is column information. Ingore the first row
	row_count = count_num_of_lines(input_file)
	for i in range(num_copies):
		test_id_list = random.sample(xrange(1, row_count), int((row_count - 1)* test_percentage))
		output_file = output_folder + "/test" + str(i) + ".txt"
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
Read ID from file and return a set
'''
def read_id_from_file(input_file):
	id_set = set()
	with open(input_file, 'r') as f:
		for number in f:
			id_set.add(number.split()[0])
	return id_set