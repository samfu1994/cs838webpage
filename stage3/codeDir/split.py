import csv
import math
import numpy as np
train_set = []
test_set = []
# with open('final_metadata.csv', 'rb') as csvfile:
# 	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
# 	pos = []
# 	neg = []
# 	for row in spamreader:
# 		print row
# 		if row[-1] == '0':
# 			neg.append(row)
# 		else:
# 			pos.append(row)
# 	thres = 0.7
# 	neg = neg[:200]
# 	mat = []
# 	for row in pos:
# 		mat.append(row)
# 	for row in neg:
# 		mat.append(row)
# 	np.random.shuffle(mat)
# 	with open('f_metadata.csv', 'wb') as c1:
# 		writer = csv.writer(c1,delimiter = ',', quotechar='|')
# 		for row in mat:
# 			writer.writerow(row)

with open('f_metadata.csv', 'rb') as c1:
	spamreader = csv.reader(c1, delimiter=',', quotechar='|')
	pos = []
	neg = []
	for row in spamreader:
		if row[-1] == '0':
			neg.append(row)
		else:
			pos.append(row)
	thres = 0.7
	l_pos = int(math.floor(len(pos) * thres))
	l_neg = int(math.floor(len(neg) * thres))
	for i in range(l_pos):
		train_set.append(pos[i])
	for i in range(l_neg):
		train_set.append(neg[i])
	for i in range(l_pos, len(pos)):
		test_set.append(pos[i])	
	for i in range(l_neg, len(neg)):
		test_set.append(neg[i])	
	np.random.shuffle(train_set)
	np.random.shuffle(test_set)
	with open('I_set.csv', 'wb') as c2:
		writer = csv.writer(c2, delimiter=',', quotechar='|')
		for row in train_set:
			writer.writerow(row)
	with open('J_set.csv', 'wb') as c3:
		writer = csv.writer(c3, delimiter=',', quotechar='|')
		for row in test_set:
			writer.writerow(row)
