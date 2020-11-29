from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class Classifiers(object):

    dict_classifiers = dict()
    dict_classifiers['SVM'] = [SVC(),
                               {'C': [1000, 100, 10, 1, 0.1, 0.001, 0.0001, 0.00001],
                                'kernel': ['rbf', 'linear', 'poly'],
                                'probability': [True, False]}
                               ]
    dict_classifiers['LogisticRegression'] = [LogisticRegression(),
                                              {'C': [1000, 100, 10, 1, 0.1, 0.001, 0.0001, 0.00001],
                                               'solver': ['saga', 'sag', 'lbfgs', 'liblinear', 'newton-cg']}
                                              ]
    dict_classifiers['MultinomialNB'] = [MultinomialNB(),
                                         {'alpha': [1000, 500, 100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005,
                                                    0.001, 0.0005, 0.0001]}]
    dict_classifiers['RandomForest'] = [RandomForestRegressor(random_state=42),
                                        {'bootstrap': [True, False],
                                         'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                                         'max_features': ['auto', 'sqrt'],
                                         'min_samples_leaf': [1, 2, 4],
                                         'min_samples_split': [2, 5, 10],
                                         'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
                                        ]
    dict_classifiers['DecisionTree'] = [DecisionTreeClassifier(),
                                        {'criterion': ['entropy', 'gini'],
                                         'max_depth': [2, 3, 5, 10, 20, 70],
                                         'min_samples_split': [2, 3, 5, 10],
                                         'min_samples_leaf': [1, 4, 5, 8]}
                                        ]
    dict_classifiers['MLPClassifier'] = [MLPClassifier(hidden_layer_sizes=(100, 30)),
                                         {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                                          'activation': ['tanh', 'relu'],
                                          'solver': ['sgd', 'adam'],
                                          'alpha': [0.0001, 0.05],
                                          'learning_rate': ['constant', 'adaptive']}
                                         ]