import csv
import os

from numpy import mean

from logic.text_analysis import TextAnalysis
from root import DIR_INPUT, DIR_EMBEDDING

lang = 'es'
sep = os.sep

file_lexicon = DIR_INPUT + 'NRC-VAD-Lexicon.txt'
file_words = DIR_EMBEDDING + 'frequency' + sep + 'frequency_word_embedding_es.csv'
ta = TextAnalysis(lang=lang)
lexicon = ta.import_lexicon_vad(file_lexicon, lang=lang)
words_intensity = {}
avg = 0.0
with open(file_words, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    avg = mean([float(row['freq']) for row in reader])
f.close()

with open(file_words, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        word = str(row['token']).strip()
        if word in lexicon:
            tmp_list = [round((float(row['freq'])/avg), 3)] + lexicon[word]
            words_intensity[word] = tmp_list
f.close()
for k, v in words_intensity.items():
    print('{0},{1}'.format(k, v))



