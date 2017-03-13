import random
import re


def extra_negative_instances(inFile, num):
  negArray = []
  file = open(inFile)
  text = str(file.read())
  p1 = '(<n>(.*?)<n>)'
  p2 = '(<p1>(.*?)<p1>)'
  p3 = '(<p2>(.*?)<p2>)'
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
  print negArray