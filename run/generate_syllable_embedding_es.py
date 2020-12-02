from logic.embedding import Embedding

em = Embedding(lang='es')
em.part_embedding(model_name='syllable_embedding')
em.plot('syllable_embedding_es')
em.plot_clusters('syllable_embedding_es')
em.get_similarity(model_name='syllable_embedding')