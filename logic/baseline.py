import csv
import datetime
import pickle
import sys
from root import DIR_OUTPUT, DIR_WMPSVAD
from logic.machine_learning import MachineLearning
from logic.text_analysis import TextAnalysis
from logic.classifiers import Classifiers
from logic.utils import Utils

# permutation 2^4 = 15
# 0001, 0010, 0011, 0100, 0101, 0110, 0111, 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111
list_vad = ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111',
            '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111']

fieldnames = ('model_name', 'word', 'syllable', 'freq_phoneme', 'one_phoneme', 'all_phoneme', 'valence', 'arousal',
              'dominance', 'sum_vad', 'classifier', 'replica', 'f1', 'accuracy', 'recall', 'precision', 'log_loss',
              'cross_entropy', 'kruskal_wallis', 'classification', 'confusion', 'best_estimator', 'sample_train',
              'sample_test', 'time_processing', 'predict_model')


class Baseline(object):

    @staticmethod
    def main(lang: str = 'es', report_name: str = 'Default', model_type: str = '11111',
             over_sampler: bool = True, target=None):
        try:
            target = [1, 0] if target is None else target
            # syllable_binary=10 Syllable phonetic sum all phonemes
            # syllable_binary=11 Syllable phonetic sum first phoneme
            # syllable_binary=00 Phoneme sum all phonemes
            # syllable_binary=01 Phoneme sum first phonemes

            file_train = 'Valence_train_oc_' + lang + '.csv'
            file_test = 'Valence_test_oc_' + lang + '.csv'

            method = Classifiers.dict_classifiers
            date_file = datetime.datetime.now().strftime("%Y-%m-%d")
            file_path_csv = DIR_OUTPUT + "{0}_{1}_{2}_{3}.csv".format(report_name, model_type, lang, date_file)
            ta = TextAnalysis(lang=lang)
            ml = MachineLearning(lang=lang, text_analysis=ta)

            setting = {'sep': ';', 'url': True, 'mention': True, 'emoji': False,
                       'hashtag': True, 'lemmatizer': False, 'stopwords': True}

            train_data = ta.import_dataset(file=file_train, **setting)
            train_data = train_data.loc[train_data['valence'].isin(target)]

            test_data = ta.import_dataset(file=file_test, **setting)
            test_data = test_data.loc[train_data['valence'].isin(target)]

            best_model = None
            best_f1 = 0.0
            # headers = dict((n, n) for n in fieldnames)
            with open(file_path_csv, 'w') as out_csv:
                writer = csv.DictWriter(out_csv, fieldnames=fieldnames, delimiter=';', lineterminator='\n')
                writer.writeheader()
                for binary_vad in list_vad:
                    for k, v in method.items():
                        result = ml.train(model_type=model_type, train_data=train_data, test_data=test_data,
                                          classifier_name=k, classifier=v, binary_vad=binary_vad,
                                          over_sampler=over_sampler, target=target)
                        writer.writerows(result)
                        out_csv.flush()
                        print('Models {0}, Classifier {1} and 10 replicas save successful!'.format(model_type, k))
                        for item in result:
                            f1 = item['f1']
                            if f1 > best_f1:
                                best_f1 = f1
                                best_model = item['model_name']
                                # save model
                                print(best_model)
                                file_model = DIR_WMPSVAD + best_model + '_model_' + lang + '.sav'
                                outfile = open(file_model, 'wb')
                                classifier = item['predict_model']
                                pickle.dump(classifier, outfile)
                                outfile.close()
                                print('Model exported in {0}'.format(file_model))
            out_csv.close()
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error baseline: {0}'.format(e))