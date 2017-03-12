import matplotlib.pyplot as plt
import numpy as np


precision = np.array([0.9648, 0.9649, 0.9739, 0.9546, 0.9647, 0.9649])
recall = np.array([0.9357, 0.9406, 0.9455, 0.9369, 0.9307, 0.9405])
F1 = np.array([0.9500, 0.9526, 0.9595, 0.9457, 0.9474, 0.9525])

index = np.arange(6)
bar_width = 0.2
opacity = 0.75
fig, ax = plt.subplots()

rects1 = plt.bar(index, precision, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Precision')
rects2 = plt.bar(index+bar_width, recall, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Recall')
rects3 = plt.bar(index+2*bar_width, F1, bar_width,
                 alpha=opacity,
                 color='b',
                 label='F1')

plt.xlabel('ML Models')
plt.ylabel('Evaluations')
plt.title('Comparison between different ML models')
plt.xticks(index + bar_width / 2, (
	'SVM', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'Linear Regression', 'Neural Network'))
plt.legend()

plt.tight_layout()
plt.show()


'''
plt.figure()
plt.hist(x, len(x), histtype='bar')
plt.title("Comparison between different ML models")
plt.xlabel("ML models")
plt.ylabel("Evaluations")
plt.show()
'''
