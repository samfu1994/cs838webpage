import re
import csv
from os import listdir
from os.path import isfile, join

mypath = "/Users/fuhao/Desktop/textFile/"
files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[-4:] == ".txt"]
data = []
word = ""

fieldsName = ['word', 'has_university', 'has_state_name', 'has_state_word']

with open("data.csv", 'w') as csvFile:
	csvWriter = csv.DictWriter(csvFile, fieldnames=fieldsName)
	csvWriter.writeheader()
	for f in files:
		with open(mypath + f) as file:
			lines = file.readlines()
			for line in lines: #each line
				data = re.findall("<[pn].*?>", line)
				l = len(data)
				if l != 0:
					for i in range(l):#each instance
						has_university = 0
						has_state_name = 0
						has_state_word = 0
						cur_list = data[i].split()
						cur_list = cur_list[1:-1]
						for i in range(len(cur_list)):
							cur_list[i] = cur_list[i].strip().lower()
						if ("university" in cur_list) or ("college" in cur_list) or ("institute" in cur_list):
							has_university = 1
						if "state" in cur_list:
							has_state_word = 1
						word = ""
						for ele in cur_list:
							word += ele
						row = {'word':word, 'has_university' : has_university, 'has_state_name' : has_state_name, 'has_state_word' : has_state_word}
						csvWriter.writerow(row)
