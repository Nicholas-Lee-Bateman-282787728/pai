from logic.embedding import Embedding

em = Embedding(lang='es')
'''
em.part_embedding(model_name='phoneme_embedding', syllable=False)
em.plot('phoneme_embedding_es')
em.plot_clusters('phoneme_embedding_es')
'''
em.get_similarity(model_name='phoneme_embedding')