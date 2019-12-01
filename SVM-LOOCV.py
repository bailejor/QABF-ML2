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



dataframe = pandas.read_csv("NoSmote.csv", header = 0)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X_orig = dataset[:,0:8].astype(float)
y_orig = dataset[:,8:11].astype(float)


#X_train = sp.csr_matrix(X_train)

param_tuner = [1]
second_parameter = [0.08]
for i in param_tuner:
	for l in second_parameter:
		classifier = BinaryRelevance(
		classifier = SVC(C= l, cache_size=200, class_weight=None, coef0=0.0,
    	decision_function_shape='ovr', degree=1, gamma=100, kernel='linear',
    	max_iter=-1, probability=False, random_state=6, shrinking=True,
    	tol=0.001, verbose=False),
		require_dense = [False, True])

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
			if np.array_equal(y_copy, equal):
				j = j + 1
				#print(y_copy, equal)
			if np.not_equal:
				#print(y_copy, equal)
				pass
		print(j/48)






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




