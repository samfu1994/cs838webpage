#code: utf-8 -*-
#author: hwang
#created on: May 6th, 2017
#modified on: May 7th, 2017

from boxplot_script import *
from sklearn import preprocessing
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

DEFINE_SET_ML = set([2, 7, 8, 14, 15, 16, 18, 19, 20, 25, 29, 30, 31, 32, 35])

def extract_features_labels(data_dir):
	'''
	we need to convert several features into numerical values
	with correspond to predefined column indexes

	Input
	data_dir

	Output
	data_table: 2 dimensional python list
	'''
	#for debug session
	row_counter = 0
	data_table = []
	year_label = []
	header_flag = True
	with open(data_dir, "rb") as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			tmp_row = []
			if header_flag:
				header_flag = False
				continue
			else:
				for ele_idx in range(len(row)):
					if ele_idx in DEFINE_SET_ML:
					#	settings for debug
					#	print(ele_idx, row_counter, row[ele_idx])
					#	print("=======================================")
						tmp_row.append(int(row[ele_idx]))
			data_table.append(tmp_row)
			if int(row[9]) >= 1975:
				year_label.append(1)
			else:
				year_label.append(0)
			row_counter += 1
	return data_table, year_label

if __name__ == "__main__":
	data_table, label = extract_features_labels(DATA_DIR)
	scaler_data = preprocessing.StandardScaler().fit(data_table)
	scaled_data = scaler_data.transform(data_table)
	train_set, test_set, train_label, test_label = train_test_split(scaled_data, label, 
																	  test_size=0.3, 
																	  random_state=42)
	clf = svm.SVC()
	clf.fit(train_set, train_label)
	predict_result = clf.predict(test_set)
	accuracy_counter = 0
	for idx in range(len(predict_result)):
		print(predict_result[idx], test_label[idx])
		if int(test_label[idx]) == predict_result[idx]:
			accuracy_counter += 1
	accuracy = float(accuracy_counter) / len(test_label)
	print(accuracy)
	'''
	#debug settings
	pos_counter = 0
	neg_counter = 0
	for l in labels:
		if l == "0":
			neg_counter+=1
		else:
			pos_counter+=1
	print(pos_counter, neg_counter)
	'''


