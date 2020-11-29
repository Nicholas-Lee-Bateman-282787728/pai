from logic.embedding import Embedding

em = Embedding(lang='en')
'''
em.part_embedding(model_name='syllable_embedding')
em.plot('syllable_embedding_en')
em.plot_clusters('syllable_embedding_en')
'''
em.get_similarity(model_name='syllable_embedding')