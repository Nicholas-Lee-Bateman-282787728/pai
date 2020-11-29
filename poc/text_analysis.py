from logic.text_analysis import TextAnalysis


serviceText_es = TextAnalysis('es')
serviceText_en = TextAnalysis('en')
text_es = 'El día empezó en el banco con un perrito uerfano jugando alegre y corriendo  con los aspersores...  la verificasion va ser un buen día 7 de Junio.'
text_en = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book"

print('Español')
print(serviceText_es.tagger(text_es))

print('Ingles')
print(serviceText_en.tagger(text_en))