import matplotlib.pyplot as plt
import numpy as np
import csv
DATA_DIR = "../data_set/processed_data.csv"

def load_csv_data(data_dir):
	'''
	we need to convert several features into numerical values
	with correspond to predefined column indexes
	'''
	data_table = []
	with open(data_dir, "rb") as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			data_element.append(row)
	return data_table

def extract_feature_value(data_table):
	pass

if __name__ == "__main__":
	data_table = load_csv_data(DATA_DIR)

