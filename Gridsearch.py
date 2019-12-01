from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas 


dataframe = pandas.read_csv("NoSmote.csv", header = 0)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X_orig = dataset[:,0:8].astype(float)
y_orig = dataset[:,8:11].astype(float)


parameters = [
    #{
        #'classifier': [MultinomialNB()],
        #'classifier__alpha': [0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0],
        #'classifier__fit_prior': [True, False],
    #},
    {
        'classifier': [SVC()],
        'classifier__kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 20, 25, 30],
        'classifier__shrinking': [True, False],
        'classifier__random_state': [6],
        'classifier__gamma': [0.1, 1, 10, 100],
        'classifier__degree': [0, 1, 2, 3, 4, 5, 6],

    },
        #{
        #'classifier': [RandomForestClassifier()],
        #'classifier__criterion': ['gini', 'entropy'],
        #'classifier__n_estimators': [20, 50, 100],
        #'classifier__max_depth': [5, 10],
        #'classifier__random_state': [6],
        #'classifier__min_samples_leaf': [1, 2, 4],
        #'classifier__max_features': [2, 3, 4, 6, 8],


    #},

]



clf = GridSearchCV(ClassifierChain(), parameters, scoring='accuracy', cv = 10)
clf.fit(X_orig, y_orig)

print (clf.best_params_, clf.best_score_)

clf2 = GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy', cv = 10)
clf2.fit(X_orig, y_orig)

print (clf2.best_params_, clf2.best_score_)