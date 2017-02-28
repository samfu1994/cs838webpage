import re
import csv
from os import listdir
from os.path import isfile, join

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
def main():
	mypath = "/Users/fuhao/Desktop/textFile/"
	files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[-4:] == ".txt"]
	data = []
	word = ""
	state_set = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",\
		"Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",\
		"Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "Hampshire", "Jersey", "Mexico", "York", "Carolina", "Dakota",\
		"Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode",  "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "Wisconsin", "Wyoming"
	]
	for i in range(len(state_set)):
		state_set[i] = state_set[i].strip().lower()

	fieldsName = ['word', 'has_university', 'has_state_name', 'has_state_word', 'length', 'has_dash', 'all_capital', 'has_num', 'label']

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
							label = 0
							has_university = 0
							has_state_name = 0
							has_state_word = 0
							length = 0
							has_dash = 0
							all_capital = 1
							has_num = 0
							cur_list = data[i].split()
							tmp = cur_list[0]
							tmp = tmp.strip()
							if tmp == "<p1" or tmp == "<p2":
								label = 1	
							origin_list = cur_list[1:-1]
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
								length += len(ele)
								if ele.find("-") != -1:
									has_dash = 1
								if hasNumbers(ele):
									has_num = 1
								if ele in state_set:
									has_state_name = 1

							if len(origin_list) == 1:
								for i in range(len(origin_list[0])):
									if origin_list[0][i] > 'Z' or origin_list[0][i] < 'A':
										all_capital = 0
										break
							else:
								all_capital = 0

							row = {'word':word, 'has_university' : has_university, 'has_state_name' : has_state_name, 'has_state_word' : has_state_word,\
								 'length' : length, 'has_dash' : has_dash, 'all_capital' : all_capital, 'has_num' : has_num, 'label' : label}
							csvWriter.writerow(row)
if __name__ == "__main__":
	main()
