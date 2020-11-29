import sys
import time
import warnings
from tqdm import tqdm
from math import log
from sklearn.metrics import log_loss
import numpy as np
from numpy import asarray
import pandas as pd
from scipy.stats import kruskal
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from collections import Counter
from logic.feature_extraction import FeatureExtraction
from logic.text_analysis import TextAnalysis
from logic.utils import Utils
warnings.filterwarnings("ignore")


class MachineLearning(object):

    def __init__(self, lang='es', text_analysis=None):
        try:
            print('Load Machine Learning')
            if text_analysis is None:
                self.ta = TextAnalysis(lang=lang)
            else:
                self.ta = text_analysis
            self.features = FeatureExtraction(lang=lang, text_analysis=self.ta)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error MachineLearning: {0}'.format(e))

    # calculate cross entropy
    @staticmethod
    def cross_entropy(p, q, ets=1e-15):
        return -sum([p[i] * log(q[i] + ets) for i in range(len(p))])

    @staticmethod
    def model_name(model_type, binary_vad):

        dict_type = {'W': int(model_type[0]), 'S': int(model_type[1]),
                     'FP': int(model_type[2]), 'OP': int(model_type[3]),
                     'AP': int(model_type[4])}

        model_name = '+'.join([k for k, v in dict_type.items() if v == 1 and len(model_type) > 0])

        result = {'model_name': model_name + '@' + binary_vad,
                  'word': int(model_type[0]),
                  'syllable': int(model_type[1]),
                  'freq_phoneme': int(model_type[2]),
                  'one_phoneme': int(model_type[3]),
                  'all_phoneme': int(model_type[4]),
                  'valence': int(binary_vad[0]),
                  'arousal': int(binary_vad[1]),
                  'dominance': int(binary_vad[2]),
                  'sum_vad': int(binary_vad[3])}
        return result

    def replica(self, dict_model, classifier_name, classifier, fold, rep, target,
                x, y, x_test, y_test, sample_train, sample_test):
        try:
            dict_model.update({'classifier': classifier_name,
                               'sample_train': sample_train,
                               'sample_test': sample_test})
            k_fold = StratifiedShuffleSplit(n_splits=fold, test_size=0.30, random_state=42)
            start_time = time.time()

            accuracies_scores = []
            recalls_scores = []
            precisions_scores = []
            f1_scores = []
            ll_score = []
            ce_score = []
            stat_kw_score = []
            p_kw_score = []
            clf = classifier
            for train_index, test_index in k_fold.split(x, y):
                data_train = x[train_index]
                target_train = y[train_index]

                data_test = x[test_index]
                target_test = y[test_index]

                clf.fit(data_train, target_train)
                predict = classifier.predict(data_test)
                # Accuracy
                accuracy = accuracy_score(target_test, predict)
                accuracies_scores.append(accuracy)
                # Recall
                recall = recall_score(target_test, predict, average='macro')
                recalls_scores.append(recall)
                # Precision
                precision = precision_score(target_test, predict, average='weighted')
                precisions_scores.append(precision)
                # F1
                f1 = f1_score(target_test, predict, average='weighted')
                f1_scores.append(f1)

                # Log Loss function
                # prepare classification data
                probability = classifier.predict_proba(data_test)
                y_true = asarray(target_test)
                y_pred = np.nan_to_num(asarray(probability))

                # calculate the average log loss
                ll = log_loss(y_true, y_pred)
                ll_score.append(ll)

                # cross-entropy for predicted probability distribution
                ents = np.nan_to_num([self.cross_entropy(target, d) for d in probability])
                ce = abs(np.mean(ents))
                ce_score.append(ce)

                # Kruskal-Wallis H Test
                stat_kw, p_kw = kruskal(y_true, predict)
                stat_kw_score.append(stat_kw)
                p_kw_score.append(p_kw)

            average_recall = round(np.mean(recalls_scores) * 100, 2)
            dict_model['recall'] = average_recall

            average_precision = round(np.mean(precisions_scores) * 100, 2)
            dict_model['precision'] = average_precision

            average_f1 = round(np.mean(f1_scores) * 100, 2)
            dict_model['f1'] = average_f1

            average_accuracy = round(np.mean(accuracies_scores) * 100, 2)
            dict_model['accuracy'] = average_accuracy

            # calculate the average cross entropy
            mean_ll = round(float(np.mean(ll_score)), 2)
            dict_model['log_loss'] = mean_ll

            mean_ce = round(float(np.mean(ce_score)), 2)
            dict_model['cross_entropy'] = mean_ce

            mean_p_kw = round(float(np.mean(p_kw_score)), 2)
            dict_model['kruskal_wallis'] = mean_p_kw
            y_predict = []
            for features in x_test:
                features = features.reshape(1, -1)
                value = clf.predict(features)[0]
                y_predict.append(value)

            classification = classification_report(y_test, y_predict)
            dict_model['classification'] = classification
            confusion = confusion_matrix(y_predict, y_test)
            dict_model['confusion'] = confusion

            dict_model['predict_model'] = clf

            # Calculated Time processing
            t_sec = round(time.time() - start_time)
            (t_min, t_sec) = divmod(t_sec, 60)
            (t_hour, t_min) = divmod(t_min, 60)
            time_processing = '{} hour:{} min:{} sec'.format(t_hour, t_min, t_sec)
            dict_model['time_processing'] = time_processing

            # print result
            print('{0} | Begin {1} - Replica #{2} | {0}'.format("#" * 12, classifier_name, rep))
            output_result = {'F1-score': average_f1, 'Accuracy': average_accuracy,
                             'Recall': average_recall, 'Precision': average_precision,
                             'Log Loss': mean_ll, 'Cross Entropy': mean_ce,
                             'Kruskal - Wallis': mean_p_kw, 'Time Processing': time_processing,
                             'Classification Report': classification, 'Confusion Matrix\n': confusion}
            for item, val in output_result.items():
                print('{0}: {1}'.format(item, val))
            print('{0} | End {1} - Replica #{2} | {0}'.format("#" * 12, classifier_name, rep))
            return dict_model
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error replica: {0}'.format(e))
            return None

    def train(self, model_type='11111', classifier_name=None, classifier=None, train_data=None, test_data=None,
              binary_vad='0000', over_sampler=True, iteration=10, fold=10, target=None):
        try:
            dict_model = self.model_name(model_type=model_type, binary_vad=binary_vad)
            if train_data is not None:
                print("#" * 15 + '| Start Model: ' + dict_model['model_name'] + ' |' + "#" * 15)
                print('Training {0} ....'.format(classifier_name))

                train_data = pd.DataFrame(train_data)
                x = train_data['message'].tolist()
                y = train_data['valence'].to_numpy()

                test_data = pd.DataFrame(test_data)
                x_test = test_data['message'].tolist()
                y_test = test_data['valence'].to_numpy()

                print('***Get training features')
                x = self.features.get_features(x, model_type=model_type, binary_vad=binary_vad)
                x = preprocessing.normalize(x)

                print('***Get testing features')
                x_test = self.features.get_features(x_test, model_type=model_type, binary_vad=binary_vad)
                x_test = preprocessing.normalize(x_test)

                # Calculated Over Sample
                print('**Sample train:', sorted(Counter(y).items()))
                print('**Sample test:', sorted(Counter(y_test).items()))
                sample_train = 'Sample train:' + str(sorted(Counter(y).items())) + '\n'
                sample_test = 'Sample test:' + str(sorted(Counter(y_test).items())) + '\n'

                if over_sampler:
                    ros_train = RandomOverSampler(random_state=1000)
                    x, y = ros_train.fit_resample(x, y)
                    print('**RandomOverSampler train:', sorted(Counter(y).items()))
                    sample_train += 'RandomOverSampler train:' + str(sorted(Counter(y).items()))
                    # test
                    ros_test = RandomOverSampler(random_state=1000)
                    x_test, y_test = ros_test.fit_resample(x_test, y_test)
                    print('**RandomOverSampler test:', sorted(Counter(y_test).items()))
                    sample_test += 'RandomOverSampler test:' + str(sorted(Counter(y_test).items()))
                else:
                    ros_train = RandomUnderSampler(random_state=1000)
                    x, y = ros_train.fit_resample(x, y)
                    print('**RandomUnderSampler train:', sorted(Counter(y).items()))
                    sample_train += 'RandomUnderSampler train:' + str(sorted(Counter(y).items()))
                    # test
                    ros_test = RandomOverSampler(random_state=1000)
                    x_test, y_test = ros_test.fit_resample(x_test, y_test)
                    print('**RandomUnderSampler test:', sorted(Counter(y_test).items()))
                    sample_test += 'RandomUnderSampler test:' + str(sorted(Counter(y_test).items()))

                result = []
                for i in range(1, iteration+1):
                    rep = int(i)
                    out_put = {'replica': rep}
                    data_dict = self.replica(dict_model=dict_model, classifier_name=classifier_name,
                                             classifier=classifier, fold=fold, target=target, x=x, y=y,
                                             x_test=x_test, y_test=y_test, sample_train=sample_train,
                                             sample_test=sample_test, rep=rep)
                    out_put.update(data_dict)
                    result.append(out_put)
                print("#" * 15 + '| End Model: ' + dict_model['model_name'] + ' |' + "#" * 15)
                return result
            else:
                print('ERROR without dataset')
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error train: {0}'.format(e))
            return None

