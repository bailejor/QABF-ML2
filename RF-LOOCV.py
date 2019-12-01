import numpy as np
import pandas
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from math import sqrt
from numpy.random import seed
from keras.regularizers import l1
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE



dataframe = pandas.read_csv("NoSmote.csv", header = 0)




dataset = dataframe.values
# split into input (X) and output (Y) variables
X_orig = dataset[:,0:8].astype(float)
y_orig = dataset[:,8:11].astype(float)



search_list = [1]
depth_list = [1]

for i in search_list:
	for l in depth_list:
		model = RandomForestClassifier(bootstrap = True, n_estimators = 100, criterion = 'entropy', random_state = 6,
			max_depth = 5, max_features = 6, max_leaf_nodes = None, 
			min_impurity_decrease = 0.0, min_impurity_split = None, min_samples_split = 2 ,
			min_samples_leaf = 1, min_weight_fraction_leaf = 0.0, n_jobs = None, oob_score = False,
			verbose = 0, warm_start = False)


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



