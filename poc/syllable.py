import spacy
from spacy_syllables import SpacySyllables

nlp = spacy.load("en_core_web_sm")
syllables = SpacySyllables(nlp)
nlp.add_pipe(syllables, after="tagger")

assert nlp.pipe_names == ["tagger", "syllables", "parser", "ner"]

doc = nlp("terribly long")
data = [(token.text, token._.syllables, token._.syllables_count) for token in doc]
print(data)