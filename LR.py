import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from math import sqrt
from numpy.random import seed
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import scipy.sparse as sp
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression





dataframe = pandas.read_csv("NoSmote.csv", header = 0)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X_orig = dataset[:,0:8].astype(float)
y_orig = dataset[:,8:11].astype(float)


search_list = [1.5]
coef_list = [10]


for i in search_list:
	for l in coef_list:
		classifier = ClassifierChain(
		classifier = LogisticRegression(C = 1, class_weight = None, dual = False, fit_intercept = True, multi_class = 'auto',
    	n_jobs = None, penalty = 'l2', random_state = None, verbose = 0, max_iter = l, solver = 'liblinear'))


		i = 0
		j = 0
		for i in range(0, 47):
			X_copy = X_orig[(i):(i+1)]  #Slice the ith element from the numpy array
			y_copy = y_orig[(i):(i+1)]
			X_model = X_orig
			y_model = y_orig
			X_model = np.delete(X_model, i, axis = 0)  #Create a new array to train the model with slicing out the ith item for LOOCV
			y_model = np.delete(y_model, i, axis = 0)
			classifier.fit(X_model, y_model)
			prediction = classifier.predict(X_copy)
			equal = prediction.toarray()
			print(equal, y_copy)
			if np.array_equal(equal, y_copy):
				j = j + 1
				#print(y_copy, equal)
			if np.not_equal:
				#print(y_copy, equal)
				pass
		print(j/48)
			#prediction = classifier.predict(X_test)
			#print(prediction.toarray())





#classifier.fit(X_train, y_train)
#predictions = classifier.predict(X_test)
#ans_formatted = predictions.toarray()
#print(ans_formatted)
#print(y_test)

#j = 0
#for i in ans_formatted:
#	if np.array_equal(y_test[j], i):
#		print("yes")
#	j = j + 1
#predictions = classifier.predict(X_test)




