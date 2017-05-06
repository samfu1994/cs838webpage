#code: utf-8 -*-
#author: hwang
#created on: May 6th, 2017

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import csv

DATA_DIR = "../data_set/data_to_analyze.csv"
#attributes whose type should be numerical
PRE_DEFINE_SET = set([2, 7, 8, 9, 14, 15, 16, 18, 19, 20, 25, 29, 30])

def load_csv_data(data_dir):
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
	header_flag = True
	with open(data_dir, "rb") as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			if header_flag:
				header_flag = False
				continue
			else:
				for ele_idx in range(len(row)):
					if ele_idx in PRE_DEFINE_SET:
					#	settings for debug
					#	print(ele_idx, row_counter, row[ele_idx])
					#	print("=======================================")
						row[ele_idx] = int(row[ele_idx])
			data_table.append(row)
			row_counter += 1
	return data_table

def extract_feature_value(data_table):
	'''
	for each atrribute with discrete value space, 
	we extract their possible values

	Input:
	data_table: 2 dimensional python list

	Output:
	feature_dict: with key-value pairs storeed
	key: index of column
	values: possible value of attribute
	'''
	feature_dict = {}
	iter_num = 0
	for row in data_table:
		for col_idx in range(len(row)):
			if col_idx not in PRE_DEFINE_SET:
				if iter_num == 0:
					feature_dict[col_idx] = {row[col_idx]}
				else:
					feature_dict[col_idx].add(row[col_idx])
		iter_num+=1
	return feature_dict

def extract_data_for_box_plot(feature_dict, data_table, col_idx=None):
	'''
	based on each possible value of a certain attribute we gather house
	prince for it

	Input:
	feature_dict: dict returned by extract_feature_value
	data_table: data set
	col_idx: designed by user

	Output:
	plot_dict: {key(attribute vals): [val (price)]}
	'''
	plot_dict = {}
	col_idx_list = feature_dict.keys()
	feature_val_list = list(feature_dict[col_idx])
	for feature_val in feature_val_list:
		list_per_featVal = []
		for row in data_table:
			#gather pricing val for each possible feature val
			if row[col_idx] == feature_val:
				list_per_featVal.append(row[-1])
		plot_dict[feature_val] = list_per_featVal
	return plot_dict

def do_boxplot(plot_dict):
	data = []
	N = 500
	#extract data from the plot dict
	for item in plot_dict.items():
		data.append(np.array(item[1]))
	#extract keys from data table
	randomDists = plot_dict.keys()
	numDists = len(randomDists)
	#start doing the boxplot
	fig, ax1 = plt.subplots(figsize=(10, 6))
	fig.canvas.set_window_title('A Boxplot Example')
	plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

	bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
	plt.setp(bp['boxes'], color='black')
	plt.setp(bp['whiskers'], color='black')
	plt.setp(bp['fliers'], color='red', marker='+')

	# Add a horizontal grid to the plot, but make it very light in color
	# so we can use it for reading data values but not be distracting
	ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
	               alpha=0.5)

	# Hide these grid behind plot objects
	ax1.set_axisbelow(True)
	ax1.set_title('Comparison of IID Bootstrap Resampling Across Five Distributions')
	ax1.set_xlabel('Distribution')
	ax1.set_ylabel('Value')

	boxColors = ['darkkhaki', 'royalblue']
	numBoxes = numDists
	medians = list(range(numBoxes))
	for i in range(numBoxes):
	    box = bp['boxes'][i]
	    boxX = []
	    boxY = []
	    for j in range(5):
	        boxX.append(box.get_xdata()[j])
	        boxY.append(box.get_ydata()[j])
	    boxCoords = list(zip(boxX, boxY))
	    # Alternate between Dark Khaki and Royal Blue
	    k = i % 2
	    boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
	    ax1.add_patch(boxPolygon)
	    # Now draw the median lines back over what we just filled in
	    med = bp['medians'][i]
	    medianX = []
	    medianY = []
	    for j in range(2):
	        medianX.append(med.get_xdata()[j])
	        medianY.append(med.get_ydata()[j])
	        plt.plot(medianX, medianY, 'k')
	        medians[i] = medianY[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
	plt.plot([np.average(med.get_xdata())], [np.average(data[i])],
	         color='w', marker='*', markeredgecolor='k')

	# Set the axes ranges and axes labels
	ax1.set_xlim(0.5, numBoxes + 0.5)
	top = 450000
	bottom = 50000
	ax1.set_ylim(bottom, top)
	#xtickNames = plt.setp(ax1, xticklabels=np.repeat(randomDists, 2))
	xtickNames = plt.setp(ax1, xticklabels=np.array(randomDists))
	plt.setp(xtickNames, rotation=45, fontsize=8)

	# Due to the Y-axis scale being different across samples, it can be
	# hard to compare differences in medians across the samples. Add upper
	# X-axis tick labels with the sample medians to aid in comparison
	# (just use two decimal places of precision)
	pos = np.arange(numBoxes) + 1
	upperLabels = [str(np.round(s, 2)) for s in medians]
	weights = ['bold', 'semibold']
	for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
	    k = tick % 2
	    ax1.text(pos[tick], top - (top*0.05), upperLabels[tick],
	             horizontalalignment='center', size='x-small', weight=weights[k],
	             color=boxColors[k])

	# Finally, add a basic legend
	
	plt.figtext(0.80, 0.08, str(N) + ' Random Numbers',
	            backgroundcolor=boxColors[0], color='black', weight='roman',
	            size='x-small')
	plt.figtext(0.80, 0.045, 'IID Bootstrap Resample',
	            backgroundcolor=boxColors[1],
	            color='white', weight='roman', size='x-small')
	plt.figtext(0.80, 0.015, '*', color='white', backgroundcolor='silver',
	            weight='roman', size='medium')
	plt.figtext(0.815, 0.013, ' Average Value', color='black', weight='roman',
	            size='x-small')
	
	plt.show()
	
if __name__ == "__main__":
	data_table = load_csv_data(DATA_DIR)
	feature_dict =  extract_feature_value(data_table)
	plot_dict = extract_data_for_box_plot(feature_dict, data_table, col_idx=0)
	do_boxplot(plot_dict)