import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class Classifiers(object):

    cores = multiprocessing.cpu_count() - 1
    dict_classifiers = dict()
    dict_classifiers['SVM'] = SVC(kernel='linear', C=0.5, probability=True)
    dict_classifiers['LogisticRegression'] = LogisticRegression(C=10, solver='lbfgs', multi_class='multinomial',
                                                                max_iter=1000, n_jobs=cores)
    dict_classifiers['RandomForest'] = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0,
                                                              n_jobs=cores)
    '''
    dict_classifiers['DecisionTree'] = DecisionTreeClassifier(max_depth=70, min_samples_leaf=1, min_samples_split=3)
    dict_classifiers['KNeighborsClassifier'] = KNeighborsClassifier(algorithm='auto', n_jobs=cores)    
    dict_classifiers['MLPClassifier'] = MLPClassifier(hidden_layer_sizes=(8, 6, 1), max_iter=300, activation='tanh',
                                                      solver='adam', random_state=123)
    '''