from source.util.data_helpers import readFileIntoList
import numpy as np

def getEmbeddingMatrix(embedding_file, tokenizer, MAX_FEATURES):

    embeddings_index = readFileIntoList(embedding_file)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    word_index_main = tokenizer.word_index

    # find embeddings for original dataset
    nb_words = min(MAX_FEATURES, len(word_index_main))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index_main.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def getEmbeddingVectorForWord(word, embeddings_index, default_embeding_vector):

    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        return embedding_vector
    else:
        return default_embeding_vector


def getEmbeddingIndexFromFile(embedding_file):
    return  readFileIntoList(embedding_file)

def getDefaultEmbeddingVector(embeddings_index):
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    return np.random.normal(emb_mean, emb_std, embed_size)

def getEmbeddingMatrixByAverage(embedding_file, tokenizer, MAX_FEATURES):

    embeddings_index = readFileIntoList(embedding_file)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    word_index_main = tokenizer.word_index

    # find embeddings for original dataset
    nb_words = min(MAX_FEATURES, len(word_index_main))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index_main.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        sum_wv = [sum(i) for i in zip(*embedding_vector)]
        avg_wv = [x / len(wv) for x in sum_wv]
    return embedding_matrix