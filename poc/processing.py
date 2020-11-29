import re
import sys
from tqdm import tqdm_notebook as tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class TextProcessing(object):

    def __init__(self):
        name = 'text_analysis'

    @staticmethod
    def stopwords(text):
        result = ''
        try:
            list_stopwords = set(stopwords.words("spanish")) + set(stopwords.words("english"))
            word_tokens = word_tokenize(text)
            word_tokens = [w for w in word_tokens if w not in list_stopwords]
            result = ' '.join(word_tokens)
        except Exception as e:
            print(e)
        return result

    @staticmethod
    def delete_special_patterns(text):
        result = ''
        try:
            text = re.sub(r'\©|\×|\⇔|\_|\»|\«|\~|\#|\$|\€|\Â|\�|\¬', '', text)# Elimina caracteres especilaes
            text = re.sub(r'\,|\;|\:|\!|\¡|\’|\‘|\”|\“|\"|\'|\`', '', text)# Elimina puntuaciones
            text = re.sub(r'\}|\{|\[|\]|\(|\)|\<|\>|\?|\¿|\°|\|', '', text)  # Elimina parentesis
            text = re.sub(r'\/|\-|\+|\*|\=|\^|\%|\&|\$', '', text)  # Elimina operadores
            #text = re.sub(r'\s+\w\s+|\w\d+|\d+\w|\d+|\d+\\|\/|\-|\s\d+|\w{22}', ' ', text)  # Elimina número y númeron con letra
            result = text.lower()
        except Exception as e:
            print(e)
        return result

    @staticmethod
    def clean_text(text, **kwargs):
        result = ''
        try:
            url = kwargs.get('url') if type(kwargs.get('url')) is bool else False
            mention = kwargs.get('mention') if type(kwargs.get('mention')) is bool else False
            emoji = kwargs.get('emoji') if type(kwargs.get('emoji')) is bool else False
            hashtag = kwargs.get('hashtag') if type(kwargs.get('hashtag')) is bool else False
            stopwords = kwargs.get('stopwords') if type(kwargs.get('stopwords')) is bool else False

            text_out = str(text).lower()
            text_out = re.sub("[\U0001f000-\U000e007f]", ' ', text_out) if emoji else text_out
            text_out = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+'
                              r'|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
                              ' ', text_out) if url else text_out
            text_out = re.sub("@([A-Za-z0-9_]{1,40})", ' ', text_out) if mention else text_out
            text_out = re.sub("#([A-Za-z0-9_]{1,40})", ' ', text_out) if hashtag else text_out
            text_out = TextProcessing.delete_special_patterns(text_out)
            text_out = TextProcessing.stopwords(text_out) if stopwords else text_out
            text_out = re.sub(r'\s+', ' ', text_out).strip()
            text_out = text_out.rstrip()
            result = text_out if text_out != ' ' else None
        except Exception as e:
            print(e)
        return result