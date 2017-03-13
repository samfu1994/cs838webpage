import random
import re
from os import listdir
from os.path import isfile, join



def extra_negative_instances(inFile, num):
  negArray = []
  file = open(inFile)
  text = str(file.read())
  p1 = '<n (.*?) n>'
  p2 = '<p1 (.*?) p1>'
  p3 = '<p2 (.*?) p2>'
  text = re.sub(p1+'|'+p2+'|'+p3, '', text)
  text = re.sub('[0-9,.;:?!\[\]\r\n\t()\"]', ' ', text)
  array = text.split()
  for i in range(num / 2):
    index = random.randrange(len(array))
    negArray.append(array[index])

  for i in range(num - num / 2):
    index = random.randrange(len(array))
    st = array[index]
    if index > 0:
      st = array[index - 1] + ' ' + st
    negArray.append(st)
    return negArray
    
def main():
  mypath = "/Users/fuhao/Development/cs838webpage/textFile/"
  files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[-4:] == ".txt"]
  train_files = files[:200]
  test_files = files[200:]
  data = []
  word = ""
  countp = 0
  countn = 0
  for f in files:
    c1, c2 = extra_negative_instances(mypath + f, 0)
    countp += c1
    countn += c2
  print countp
  print countn

if __name__ == "__main__":
  main()