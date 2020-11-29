import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from logic.feature_extraction import FeatureExtraction
from logic.text_analysis import TextAnalysis
from root import DIR_WMPSVAD
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np



class WPE(object):
    def __init__(self, model_file: str = 'W+S+FP+OP+AP@1010_model_es.sav', lang='es'):
        setting = {'sep': ';', 'url': True, 'mention': True, 'emoji': False,
                   'hashtag': True, 'lemmatizer': False, 'stopwords': True}
        file_test = 'Valence_test_oc_' + lang + '.csv'
        filename_model = DIR_WMPSVAD + model_file
        self.clf = pickle.load(open(filename_model, 'rb'))
        ta = TextAnalysis(lang=lang)
        self.features = FeatureExtraction(lang=lang, text_analysis=ta)
        self.test_data = ta.import_dataset(file=file_test, **setting)

    def predict(self, model_type: str = '11111', binary_vad: str = '1010'):
        class_names = [-1, 0, 1]
        list_text = [row[0] for row in self.test_data.values.tolist()]
        y_test = np.array([row[1] for row in self.test_data.values.tolist()])
        y_predict = []
        for text in list_text:
            features = self.features.get_features(messages=[text], model_type=model_type, binary_vad=binary_vad)[0]
            yyy = self.clf(features)[0]
            print(yyy)

        y_predict = np.array(y_predict)
        confusion = confusion_matrix(y_test, y_predict, labels=class_names)
        fig, ax = plot_confusion_matrix(conf_mat=confusion,
                                        colorbar=True,
                                        show_absolute=False,
                                        show_normed=True,
                                        class_names=class_names)
        plt.show()
        df = pd.DataFrame({'text': list_text, 'y_predict': y_predict, 'y_test': y_test})
        print(df.to_string())


if __name__ == "__main__":

    wpe = WPE(lang='es')
    result = wpe.predict()