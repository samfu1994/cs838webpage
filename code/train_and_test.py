#author: hwang
#version: 2.0,0
import csv
import numpy as np
import random
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
import itertools
import sys
import argparse
import pydotplus
from IPython.display import Image


TRAIN_RATIO = 0.8
TEST_RATIO = 0.2

def k_fold(k, dataset, label):
	#initialize 
	classifier_list = []
	sub_data_sets = []
	sub_labels = []
	CV_accuracy = []
	index_interval = len(dataset) / k
	iter_idx = 0
	#split data set and labels to k sub_sets
	for i in range(k):
		classifier_list.append(svm.SVC())
		sub_data_sets.append(dataset[iter_idx:iter_idx+index_interval])
		sub_labels.append(label[iter_idx:iter_idx+index_interval])
		iter_idx += index_interval
	#begin k-fold validation
	for i in range(k):
		accuracy_counter = 0
		tmp_train_set = []
		tmp_train_label = []
		#choose i-th sub_sets for CV set
		#choose other sub_sets as training set
		if i == 0:
			tmp_train_set.extend(sub_data_sets[i+1:])
			tmp_train_label.extend(sub_labels[i+1:])
			tmp_train_set = list(itertools.chain.from_iterable(tmp_train_set))
			tmp_train_label = list(itertools.chain.from_iterable(tmp_train_label))
			
		else:
			tmp_train_set.extend(sub_data_sets[0:i])
			tmp_train_set.extend(sub_data_sets[i+1:])
			tmp_train_label.extend(sub_labels[0:i])
			tmp_train_label.extend(sub_labels[i+1:])
			tmp_train_set = list(itertools.chain.from_iterable(tmp_train_set))
			tmp_train_label = list(itertools.chain.from_iterable(tmp_train_label))

		tmp_cv_set = sub_data_sets[i]
		tmp_cv_label = sub_labels[i]

		tmp_train_set = np.array(tmp_train_set)
		tmp_train_label = np.array(tmp_train_label)
		tmp_cv_set = np.array(tmp_cv_set)
		tmp_cv_label = np.array(tmp_cv_label)

		classifier_list[i].fit(tmp_train_set, tmp_train_label)
		tmp_prediction = classifier_list[i].predict(tmp_cv_set)
		#calc CV accuracy
		for idx in range(len(tmp_prediction)):
			if tmp_cv_label[idx] == tmp_prediction[idx]:
				accuracy_counter += 1
		CV_accuracy.append(float(accuracy_counter) / len(test_label))
	max_accuracy = max(CV_accuracy)
	max_idx = CV_accuracy.index(max_accuracy)
	best_cv_classifier = classifier_list[max_idx]
	return best_cv_classifier

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--modelname', help='The machine learning model name.', required=True)
	args = vars(parser.parse_args())
	model_name = args["modelname"]

	train_data = []
	test_data = []
	#load and parse the csv file
	with open('train_data.csv', 'r') as data_file:
		spamreader = csv.reader(data_file)
		for row in spamreader:
			train_data.append(row[1:])
	train_data = train_data[1:]
	for instance in train_data:
		instance[3] = int(instance[3])

	with open('test_data.csv', 'r') as data_file:
		spamreader = csv.reader(data_file)
		for row in spamreader:
			test_data.append(row[1:])
	test_data = test_data[1:]
	for instance in test_data:
		instance[3] = int(instance[3])

	#start train with SVM
	if model_name == 'SVM':
		clf = svm.SVC()
	if model_name == 'decision_tree':
		clf = tree.DecisionTreeClassifier()
	if model_name == 'random_forest':
		clf = RandomForestClassifier()
	if model_name == 'logistic_regression':
		clf = LogisticRegression()
	if model_name == 'linear_regression' or model_name == 'neural_net':
		if model_name == 'linear_regression':
			clf = LinearRegression()
		elif model_name == 'neural_net':
			clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
		for instance in train_data:
			for feature_idx in range(len(train_data[0])):
				instance[feature_idx] = int(instance[feature_idx])

	#calc ratio between traning instances and test instances
	#split raw data into training set and test set
	train_set = []
	train_label = []
	test_set = []
	test_label = []
	for i in range(len(train_data)):
		train_row = train_data[i]
		train_set.append(train_row[0:-1])
		train_label.append(train_row[-1])
	for i in range(len(test_data)):
		test_row = test_data[i]
		test_set.append(test_row[0:-1])
		test_label.append(test_row[-1])

	train_set = np.array(train_set)
	train_label = np.array(train_label)
	test_set = np.array(test_set)
	test_label = np.array(test_label)

	clf.fit(train_set, train_label)

	if model_name == "decision_tree":
	#visualize the tree
		university_feature_names = [
							"has university",
							"has state name",
							"has state word",
							"length",
							"has dash",
							"all_capital",
							"has num"
							]
		university_target_name = ["True", "False"]
		dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=university_feature_names,  
                         class_names=university_target_name,  
                         filled=True, rounded=True,  
                         special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_jpg("tree.jpg")

	#make prediction
	if model_name == 'logistic_regression' or model_name == 'linear_regression' or model_name == 'neural_net':
		test_set = test_set.astype(np.float)
	predict_result = clf.predict(test_set)

	if model_name == 'linear_regression' or model_name == 'neural_net':
		accuracy_counter = 0
		tp = 0
		fn = 0
		fp = 0
		for i in range(len(predict_result)):
			predict = predict_result[i]
			if predict >= 0.5:
				predict_result[i] = 1
			else:
				predict_result[i] = 0

		for idx in range(len(predict_result)):
			if int(test_label[idx]) == predict_result[idx]:
				accuracy_counter += 1
			if predict_result[idx] == 1 and int(test_label[idx]) == 1:
				tp += 1
			if predict_result[idx] == 0 and int(test_label[idx]) == 1:
				fn += 1
			if predict_result[idx] == 1 and int(test_label[idx]) == 0:
				fp += 1
	 	accuracy = float(accuracy_counter) / len(test_label)
	 	precision = float(tp) / (tp + fp)
	 	recall = float(tp) / (tp + fn) 
	 	F1_score = 2 * (precision*recall) / float(precision + recall)
		print("Test set accuracy: %s\tPrecision: %s\tRecall: %s\tF_1: %s" % (str(accuracy), str(precision), str(recall),
			str(F1_score)))		

	#calc test set accuracy/ precision/ recall
	#naive classifier only with training set
	else:
		accuracy_counter = 0
		tp = 0
		fn = 0
		fp = 0
		for idx in range(len(predict_result)):
			if test_label[idx] == predict_result[idx]:
				accuracy_counter += 1
			if predict_result[idx] == '1' and test_label[idx] == '1':
				tp += 1
			if predict_result[idx] == '0' and test_label[idx] == '1':
				fn += 1
			if predict_result[idx] == '1' and test_label[idx] == '0':
				fp += 1
	 	accuracy = float(accuracy_counter) / len(test_label)
	 	precision = float(tp) / (tp + fp)
	 	recall = float(tp) / (tp + fn)
	 	F1_score = 2 * (precision*recall) / float(precision + recall)
		print("Test set accuracy: %s\tPrecision: %s\tRecall: %s\tF_1: %s" % (str(accuracy), str(precision), str(recall),
			str(F1_score)))	
