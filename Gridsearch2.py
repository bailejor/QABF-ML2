from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas 
from sklearn.tree import DecisionTreeClassifier


dataframe = pandas.read_csv("NoSmote.csv", header = 0)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X_orig = dataset[:,0:8].astype(float)
y_orig = dataset[:,8:11].astype(float)


parameters = [
    {
        'classifier': [MultinomialNB()],
        'classifier__alpha': [0.3, 0.5, 0.7, 1.0, 2.0, 3.0],
        'classifier__fit_prior': [True, False],
    },

    {
        'classifier': [DecisionTreeClassifier()],
        'classifier__max_depth' : [3, 5, 10, 15, 20, 25, 30],
        'classifier__criterion' : ['entropy', 'gini'],
        'classifier__min_samples_leaf' : [3, 5, 10, 15, 20, 25],
        'classifier__max_features' : [3, 4, 6, 8],

    },
        {
        'classifier': [LogisticRegression()],
        'classifier__C': [1.0, 1.5, 2.0, 2.5],
        'classifier__max_iter': [10, 25, 50, 100, 200, 300],
        'classifier__solver': ['liblinear'],
    },

]



clf = GridSearchCV(ClassifierChain(), parameters, scoring='accuracy', cv = 48)
clf.fit(X_orig, y_orig)

print (clf.best_params_, clf.best_score_)

clf2 = GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy', cv = 48)
clf2.fit(X_orig, y_orig)

print (clf2.best_params_, clf2.best_score_)
