import numpy as np
import pandas
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from math import sqrt
from numpy.random import seed
from keras.regularizers import l1
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from skmultilearn.problem_transform import ClassifierChain




dataframe = pandas.read_csv("NoSmote.csv", header = 0)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X_orig = dataset[:,0:8].astype(float)
y_orig = dataset[:,8:11].astype(float)

search_list = [1]
depth_list = [7]

for i in search_list:
	for l in depth_list:
		model = DecisionTreeClassifier(criterion= "gini", splitter="best", max_depth=25, min_samples_split=2, 
			min_samples_leaf=i, min_weight_fraction_leaf=0.0, max_features= l, random_state=6, max_leaf_nodes=None, 
			min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)


		loo = LeaveOneOut()
		j = 0
		test_fold_predictions = []
		for train_index, test_index in loo.split(X_orig):
			X_train, X_test = X_orig[train_index], X_orig[test_index]
			y_train, y_test = y_orig[train_index], y_orig[test_index]
			model.fit(X_train, y_train)
			prediction = (model.predict(X_test))

			print(prediction, y_test)
			#test_fold_predictions.append(model.predict(X_test))
			#print(test_fold_predictions)	

			if np.array_equal(prediction, y_test):
				j = j + 1
				#print(y_copy, prediction)
			if np.not_equal:
				#print(y_copy, prediction)
				pass
		print(j/48)




