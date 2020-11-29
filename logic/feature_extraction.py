import sys
import epitran
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from logic.text_analysis import TextAnalysis
from logic.utils import Utils
from root import DIR_INPUT, DIR_MODELS


class FeatureExtraction(BaseEstimator, TransformerMixin):

    def __init__(self, lang='es', text_analysis=None):
        try:
            if text_analysis is None:
                self.ta = TextAnalysis(lang=lang)
            else:
                self.ta = text_analysis
            file_lexicon = DIR_INPUT + 'NRC-VAD-Lexicon.txt'
            file_word_embedding_en = DIR_MODELS + 'word_embedding_en.model'
            file_word_embedding_es = DIR_MODELS + 'word_embedding_es.model'
            file_syllable_embedding_en = DIR_MODELS + 'syllable_embedding_en.model'
            file_syllable_embedding_es = DIR_MODELS + 'syllable_embedding_es.model'
            file_phoneme_embedding_en = DIR_MODELS + 'phoneme_embedding_en.model'
            file_phoneme_embedding_es = DIR_MODELS + 'phoneme_embedding_es.model'
            print('Loading Lexicons and Embedding.....')
            if lang == 'es':
                epi = epitran.Epitran('spa-Latn')
                lexicon = self.ta.import_lexicon_vad(file_lexicon, lang=lang)
                word_embedding = Word2Vec.load(file_word_embedding_es)
                syllable_embedding = Word2Vec.load(file_syllable_embedding_es)
                phoneme_embedding = Word2Vec.load(file_phoneme_embedding_es)
            else:
                epi = epitran.Epitran('eng-Latn')
                lexicon = self.ta.import_lexicon_vad(file_lexicon, lang=lang)
                word_embedding = Word2Vec.load(file_word_embedding_en)
                syllable_embedding = Word2Vec.load(file_syllable_embedding_en)
                phoneme_embedding = Word2Vec.load(file_phoneme_embedding_en)

            self.epi = epi
            self.lexicon = lexicon
            self.word_embedding = word_embedding
            self.syllable_embedding = syllable_embedding
            self.phoneme_embedding = phoneme_embedding
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error FeatureExtraction: {0}'.format(e))

    def fit(self, x, y=None):
        return self

    def transform(self, list_messages):
        try:
            result = self.get_features(list_messages)
            return result
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error transform: {0}'.format(e))

    def get_features(self, messages, model_type='11111', binary_vad='0000'):
        try:
            # W: Word, S:Syllable, F: Frequency Phoneme, S: One/All Phoneme
            # '1111', '1110', '1101', '1100', '1011', '1010', '1001', '1000',
            # '0111', '0110', '0101', '0100', '0011', '0010'
            word_features = self.get_feature_word(messages)
            syllable_features = self.get_feature_syllable(messages)
            phoneme_frequency = self.get_frequency_phoneme(messages)
            one_syllable = self.get_feature_phoneme(messages)
            all_syllable = self.get_feature_phoneme(messages, syllable=True)
            vad_features = self.get_feature_vad(messages, binary=binary_vad)
            result = np.zeros((len(messages), 0), dtype="float32")
            if int(model_type[0]) == 1:
                result = np.append(result, word_features, axis=1)
            elif int(model_type[1]) == 1:
                result = np.append(result, syllable_features, axis=1)
            elif int(model_type[2]) == 1:
                result = np.append(result, phoneme_frequency, axis=1)
            elif int(model_type[3]) == 1:
                result = np.append(result, one_syllable, axis=1)
            elif int(model_type[4]) == 1:
                result = np.append(result, all_syllable, axis=1)

            result = np.append(result, vad_features, axis=1)
            return result
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_features: {0}'.format(e))
            return None

    def get_feature_vad(self, messages, binary='0000'):
        try:
            counter = 0
            num_features = 4
            msg_feature_vec = np.zeros((len(messages), num_features), dtype="float32")
            for msg in tqdm(messages):
                dict_vad = self.get_vad(msg)
                v = dict_vad['valence']
                a = dict_vad['arousal']
                d = dict_vad['dominance']
                vad = dict_vad['vad']
                row = []
                if binary == '0001':
                    row = [0.0, 0.0, 0.0, vad]
                elif binary == '0010':
                    row = [0.0, 0.0, d, 0.0]
                elif binary == '0011':
                    row = [0.0, 0.0, d, vad]
                elif binary == '0100':
                    row = [0.0, a, 0.0, 0.0]
                elif binary == '0101':
                    row = [0.0, a, 0.0, vad]
                elif binary == '0110':
                    row = [0.0, a, d, 0.0]
                elif binary == '0111':
                    row = [0.0, a, d, vad]
                elif binary == '1000':
                    row = [v, 0.0, 0.0, 0.0]
                elif binary == '1001':
                    row = [v, 0.0, 0.0, vad]
                elif binary == '1010':
                    row = [v, 0.0, d, 0.0]
                elif binary == '1011':
                    row = [v, 0.0, d, vad]
                elif binary == '1100':
                    row = [v, a, 0.0, 0.0]
                elif binary == '1101':
                    row = [v, a, 0.0, vad]
                elif binary == '1110':
                    row = [v, a, d, 0.0]
                elif binary == '1111':
                    row = [v, a, d, vad]
                elif binary == '0000':
                    row = [0.0, 0.0, 0.0, 0.0]
                msg_feature_vec[counter] = row
                counter = counter + 1
            return msg_feature_vec
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_feature_vad: {0}'.format(e))
            return None

    def get_feature_word(self, messages):
        try:
            counter = 0
            model = self.word_embedding
            num_features = model.vector_size
            index2word_set = set(model.wv.index2word)
            msg_feature_vec = np.zeros((len(messages), num_features), dtype="float32")
            for msg in tqdm(messages):
                num_words = 1
                feature_vec = []
                list_words = [token['text'] for token in self.ta.tagger(msg)]
                for word in list_words:
                    if word in index2word_set:
                        vec = model.wv[word]
                        feature_vec.append(vec)
                    else:
                        feature_vec.append(np.zeros(num_features, dtype="float32"))
                    num_words += 1
                feature_vec = np.array(feature_vec, dtype="float32")
                feature_vec = np.sum(feature_vec, axis=0)
                feature_vec = np.divide(feature_vec, num_words)
                msg_feature_vec[counter] = feature_vec
                counter = counter + 1
            return msg_feature_vec
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_feature_word: {0}'.format(e))
            return None

    def get_feature_syllable(self, messages, syllable_binary='11'):
        try:
            counter = 0
            model = self.syllable_embedding
            num_features = model.vector_size
            index2phoneme_set = set(model.wv.index2word)
            msg_feature_vec = np.zeros((len(messages), num_features), dtype="float32")
            for msg in tqdm(messages):
                num_phonemes = 1
                feature_vec = []
                # print('Msg: {0}'.format(msg))
                list_syllable = [token['syllables'] for token in self.ta.tagger(msg) if token['syllables'] is not None]
                for syllable in list_syllable:
                    for s in syllable:
                        syllable_phonetic = self.epi.transliterate(s, normpunc=True)
                        if syllable_phonetic in index2phoneme_set:
                            vec = model.wv[syllable_phonetic]
                            feature_vec.append(vec)
                            num_phonemes += 1
                feature_vec = np.array(feature_vec, dtype="float32")
                feature_vec = np.sum(feature_vec, axis=0)
                feature_vec = np.divide(feature_vec, num_phonemes)
                msg_feature_vec[counter] = feature_vec
                counter += 1
            return msg_feature_vec
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_feature_syllable: {0}'.format(e))
            return None

    def get_frequency_phoneme(self, messages):
        try:
            counter = 0
            model = self.phoneme_embedding
            index2phoneme = list(model.wv.index2word)
            num_features = len(index2phoneme)
            msg_feature_vec = np.zeros((len(messages), num_features), dtype="float32")
            for msg in tqdm(messages):
                # print('Msg: {0}'.format(msg))
                feature_vec = np.zeros(num_features, dtype="float32")
                list_syllable = [token['syllables'] for token in self.ta.tagger(msg) if token['syllables'] is not None]
                for syllable in list_syllable:
                    for s in syllable:
                        syllable_phonetic = self.epi.transliterate(s, normpunc=True)
                        if syllable_phonetic in index2phoneme:
                            index = index2phoneme.index(syllable_phonetic)
                            value = feature_vec[index]
                            feature_vec[index] = value + 1
                msg_feature_vec[counter] = feature_vec
                counter += 1
            return msg_feature_vec
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_frequency_phoneme: {0}'.format(e))
            return None

    def get_feature_phoneme(self, messages, syllable=False):
        try:
            counter = 0
            model = self.phoneme_embedding
            num_features = model.vector_size
            index2phoneme_set = set(model.wv.index2word)
            msg_feature_vec = np.zeros((len(messages), num_features), dtype="float32")
            for msg in tqdm(messages):
                size = 1
                feature_vec = []
                list_syllable = [token['syllables'] for token in self.ta.tagger(msg) if token['syllables'] is not None]
                if syllable:
                    try:
                        first_syllable = str(list_syllable[0][0])
                        first_syllable = first_syllable[0] \
                            if (first_syllable is not None) and (len(first_syllable) > 0) else ''
                        syllable_phonetic = self.epi.transliterate(first_syllable)
                        if syllable_phonetic in index2phoneme_set:
                            vec = model.wv[syllable_phonetic]
                            feature_vec.append(vec)
                        else:
                            feature_vec.append(np.zeros(num_features, dtype="float32"))
                    except Exception as e_epi:
                        print('Error transliterate: {0}'.format(e_epi))
                        pass
                else:
                    list_phoneme = self.epi.trans_list(msg)
                    size = len(list_phoneme)
                    for phoneme in list_phoneme:
                        if phoneme in index2phoneme_set:
                            vec = model.wv[phoneme]
                            feature_vec.append(vec)
                        else:
                            feature_vec.append(np.zeros(num_features, dtype="float32"))
                # print('Vector: {0}'.format(feature_vec))
                feature_vec = np.array(feature_vec, dtype="float32")
                feature_vec = np.sum(feature_vec, axis=0)
                feature_vec = np.divide(feature_vec, size)
                msg_feature_vec[counter] = feature_vec
                counter += 1
            return msg_feature_vec
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_feature_phoneme: {0}'.format(e))
            return None

    def get_vad(self, message):
        result = None
        try:
            valence = 0.0
            arousal = 0.0
            dominance = 0.0
            vad = 0.0
            num_word = 1
            lexicon = self.lexicon
            list_word = [token['text'] for token in self.ta.tagger(message)]
            for word in list_word:
                if word in lexicon:
                    values = lexicon[word]
                    valence += values[0]
                    arousal += values[1]
                    dominance += values[2]
                    vad += values[3]
                else:
                    valence = 0.0
                    arousal = 0.0
                    dominance = 0.0
                    vad = 0.0
                num_word += 1
            valence = round((valence / num_word), 4)
            arousal = round((arousal / num_word), 4)
            dominance = round((dominance / num_word), 4)
            vad = round((vad / num_word), 4)
            result = {'valence': valence, 'arousal': arousal, 'dominance': dominance, 'vad': vad}
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_vad: {0}'.format(e))
        return result