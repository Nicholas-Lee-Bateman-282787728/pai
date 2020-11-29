from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# RandomForest
array_rf = np.array([[68, 45, 20],
                     [9, 23, 14],
                     [8, 17, 51]])
# LogisticRegression
array_lr = np.array([[56, 36, 17],
                     [14, 32, 16],
                     [15, 17, 52]])
# SVM
array_svm = np.array([[60, 38, 14],
                      [19, 36, 21],
                      [6, 11, 50]])


class_names = ['pos', 'neu', 'neg']

fig, ax = plot_confusion_matrix(conf_mat=array_svm,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True,
                                class_names=class_names)
plt.show()


