import csv
import numpy as np

ORI_TRAIN_DIR = "../data_set/train.csv"
ORI_TEST_DIR = "../data_set/test.csv"
OUT_DIR = "../data_set/processed_data.csv"
#these data columns are what we interested in
PRE_DEFINED_SET = set([1, 2, 3, 8, 10, 13, 14, 15, 16, 
17, 19, 21, 23, 24, 26, 30, 33, 34, 36, 39, 40, 42, 43,
44, 45, 47, 48, 49, 73])

def load_data_set(data_dir):
	data_element = []
	with open(data_dir, "rb") as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			data_element.append(row)
	return data_element

def dataTable_preProcess(data_table):
	'''
	do data preprocess by remove those columns in which NA type contained
	we predefined severl data columns that we're interested
	the columns index we're interested here are:
	{0, 1, 2, 3, 8, 10, 13, 14, 15, 16, 17, 19, 21, 23, 24, 26, 30, 
	33, 34, 36, 39, 40, 42, 43, 44, 45, 47, 48, 49}

	Input:
	data_table: a two-dimensional python list

	Output:
	None
	'''
	#record data columsn which contains "NA"
	col_idx_set = set()
	processed_dataset = []
	for data_row in data_table:
		for col_val in data_row:
			if col_val == "NA":
				col_idx_set.add(data_row.index("NA"))
	#del element with value "NA" for each row
	#we just choose the data column we want here
	for data_row in data_table:
		tmp_processed = []
		for idx in col_idx_set:
			del data_row[idx]
		for idx in range(len(data_row)):
			if idx in PRE_DEFINED_SET:
				tmp_processed.append(data_row[idx])
		processed_dataset.append(tmp_processed)

	with open(OUT_DIR, "wb") as csv_out:
		csv_writer = csv.writer(csv_out)
		for data_row in processed_dataset:
			csv_writer.writerow(data_row)

if __name__ == "__main__":
	original_train_set = load_data_set(ORI_TRAIN_DIR)
	dataTable_preProcess(original_train_set)
#	original_test_set = load_data_set(ORI_TEST_DIR)
#	full_data_table = np.concatenate((original_train_set, original_test_set), axis=0)
#	np.savetxt("../data_set/full_data_table.csv", processed_dataset, delimiter=",")
#	for item in original_train_set:
#		print(len(item))