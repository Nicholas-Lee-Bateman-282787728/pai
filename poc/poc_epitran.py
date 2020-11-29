import epitran
text_es = "20 niños están contagiados de Covid en Cartagena."
text_en = "This will go down in history as one of science and medical research's greatest achievements."
epi_en = epitran.Epitran('eng-Latn')
epi_es = epitran.Epitran('spa-Latn')

result_en = epi_en.transliterate(text_en, normpunc=True)
result_es = epi_en.transliterate(text_es, normpunc=True)

print(result_en)

print(result_es)