from sklearn.decomposition import LatentDirichletAllocation as LDA
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer

number_words=20

def lda(data, NUMBER_OF_CLUSTERS):
    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(data)

    lda = LDA(n_components=NUMBER_OF_CLUSTERS)
    lda.fit(count_data)

    all_topics_words=[]
    words=count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(lda.components_):
        each_topic_words=([words[i] for i in topic.argsort()[:-number_words - 1:-1]])
        # each_topic_words=([words[i] for i  in topic.argsort()[::-1]])
        all_topics_words.append(each_topic_words)

    return all_topics_words




