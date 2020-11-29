
from logic.embedding import Embedding

em = Embedding(lang='en')
em.words_embedding(model_name='word_embedding')
em.plot('word_embedding_en')
em.plot_clusters('word_embedding_en')