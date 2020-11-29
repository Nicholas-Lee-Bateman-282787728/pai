from logic.embedding import Embedding

em = Embedding(lang='en')
'''
em.part_embedding(model_name='phoneme_embedding', syllable=False)
em.plot('phoneme_embedding_en')
em.plot_clusters('phoneme_embedding_en')
'''
em.get_similarity(model_name='phoneme_embedding')