#code: utf-8 -*-
#author: hwang
#created on: May 5th, 2017

import csv
import numpy as np

ORI_TRAIN_DIR = "../data_set/train.csv"
ORI_TEST_DIR = "../data_set/test.csv"
OUT_DIR = "../data_set/processed_data.csv"
EXT_DIR = "../data_set/data_to_analyze.csv"
#these data columns are what we interested in
PRE_DEFINED_SET = set([1, 2, 3, 8, 10, 13, 14, 15, 16, 17, 19, 21, 23, 25, 26,
28, 29, 30, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 47, 48, 50, 51, 52, 57, 
58, 61])

def load_data_set(data_dir):
	data_element = []
	with open(data_dir, "rb") as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			data_element.append(row)
	return data_element

def write_to_csv(out_dir, data_table):
	with open(out_dir, "wb") as csv_out:
		csv_writer = csv.writer(csv_out)
		for data_row in data_table:
			csv_writer.writerow(data_row)

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
	extracted_dataset = []
	for data_row in data_table:
		for col_val_id in range(len(data_row)):
			col_val = data_row[col_val_id]
			if col_val == "NA":
				col_idx_set.add(int(col_val_id))
	#del element with value "NA" for each row
	#we just choose the data column we want here
	for data_row in data_table:
		tmp_processed = []
		for ele_idx in range(len(data_row)):
			if ele_idx not in col_idx_set:
				tmp_processed.append(data_row[ele_idx])
		processed_dataset.append(tmp_processed)
	#write processed data to dir
	write_to_csv(out_dir=OUT_DIR, data_table=processed_dataset)
	for row in processed_dataset:
		tmp_processed = []
		for ele_idx in range(len(row)):
			if ele_idx in PRE_DEFINED_SET:
				tmp_processed.append(row[ele_idx])
		extracted_dataset.append(tmp_processed)
	#write interested data to dir
	write_to_csv(out_dir=EXT_DIR, data_table=extracted_dataset)

if __name__ == "__main__":
	original_train_set = load_data_set(ORI_TRAIN_DIR)
	dataTable_preProcess(original_train_set)