#code: utf-8 -*-
#author: hwang
#created on: May 6th, 2017
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVR
from boxplot_script import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

#DEFINE_SET_ML = set([2, 7, 8, 9, 14, 15, 16, 18, 19, 20, 25, 29, 30, 31, 32, 33, 34, 35])
DEFINE_SET_ML = set([2, 7, 8, 14, 15, 16, 18, 19, 20, 25, 29, 30, 31, 32, 35])

def extract_numerical_features(data_dir):
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
	header_names = []
	header_flag = True
	with open(data_dir, "rb") as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			tmp_row = []
			if header_flag:
				header_flag = False
				for ele_idx in range(len(row)):
					if ele_idx in DEFINE_SET_ML:
						header_names.append(row[ele_idx])
				continue
			else:
				for ele_idx in range(len(row)):
					if ele_idx in DEFINE_SET_ML:
					#	settings for debug
					#	print(ele_idx, row_counter, row[ele_idx])
					#	print("=======================================")
						tmp_row.append(int(row[ele_idx]))
			data_table.append(tmp_row)
			row_counter += 1
	return data_table, header_names

def split_data_table(data_table):
	'''
	split data table into data tuples and labels

	Input:
	data_table: 2D python list 

	Output:
	data_set: 2D python list
	label: 1D python list
	'''
	return [item[0:-1] for item in data_table], [item[-1] for item in data_table]

def ridge_regression(train_data, train_label, rf=0.5):
	reg_l2 = linear_model.Ridge(alpha = rf)
	reg_l1 = linear_model.Lasso(alpha = 0.1)
	reg_l2.fit(train_data, train_label)
	reg_l1.fit(train_data, train_label)
	return reg_l1, reg_l2 

def SVR_regression(train_data, train_label):
	clf = SVR(C=1.0, epsilon=0.2)
	clf.fit(train_data, train_label)
	return clf

def get_accuracy_score(predictor, test_data, test_label):
	#do test, and get MSE value from test set and test labels
	prediction = predictor.predict(test_data)
	MSE_socre = mean_squared_error(test_label, prediction)
	MAE_score = mean_absolute_error(test_label, prediction)
	median_score = median_absolute_error(test_label, prediction)
	r2_socre = r2_score(test_label, prediction, multioutput='uniform_average')
	return MSE_socre, MAE_score, median_score, r2_socre

def plot_coefficients(classifier, feature_names, top_features=14):
	coef = classifier.coef_.ravel()
	top_positive_coefficients = np.argsort(coef)[-top_features:]
	top_negative_coefficients = np.argsort(coef)[:top_features]
	#top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
	# create plot
	plt.figure(figsize=(14, 8))
	colors = ['red' if c < 0 else 'blue' for c in coef[top_positive_coefficients]]
	#plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
	plt.bar(np.arange(top_features), coef[top_positive_coefficients], color=colors)
	feature_names = np.array(feature_names)
	#plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
	plt.xticks(np.arange(1, 1 + top_features), feature_names[top_positive_coefficients], rotation=60, ha='right')
	plt.show()

if __name__ == "__main__":
	data_table, feature_names = extract_numerical_features(DATA_DIR)
	data_set, label =  split_data_table(data_table)
	scaler_label = preprocessing.StandardScaler().fit(label)
	scaled_label = scaler_label.transform(label)
	scaler_data = preprocessing.StandardScaler().fit(data_set)
	scaled_data = scaler_data.transform(data_set)
	train_data, test_data, train_label, test_label = train_test_split(scaled_data, scaled_label, 
																	  test_size=0.3, 
																	  random_state=42)
	ridge_model, lasso_model = ridge_regression(train_data, train_label)
	SVR_model = SVR_regression(train_data, train_label)
	MSE_socre, MAE_score, median_score, r2_socre = get_accuracy_score(ridge_model, train_data, train_label)
	plot_coefficients(lasso_model, feature_names)
	print(MSE_socre, MAE_score, median_score, r2_socre)