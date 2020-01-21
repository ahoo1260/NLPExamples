import pandas as pd

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import configparser

config=configparser.ConfigParser()
config.read('./config.txt')

NUMBER_OF_CLUSTERS = config.getint('source-config', 'NUMBER_OF_CLUSTERS')


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plotClusters(data):
    #--------------------------------------plotting-----------------------------------------------------------------------
    cmap = get_cmap(NUMBER_OF_CLUSTERS)
    fig, ax = plt.subplots()

    topic_index=0
    for embedding_matrix in data:
        pca = PCA(n_components=2)
        result_to_show = pca.fit_transform(embedding_matrix)
        ax.plot(result_to_show[:, 0], result_to_show[:, 1], 'o', color=cmap(topic_index))
        topic_index=topic_index+1
    plt.show()


def plotTwoLists(data1, data2, title):
    s1=pd.Series(data1)
    s2=pd.Series(data2)

    d = pd.DataFrame({'Real': s1, 'Prediction': s2})
    d.apply(pd.value_counts).plot(kind='bar',title=title)
    plt.xlabel("Sentiment")
    plt.ylabel('Number of Samples')
    plt.show()





