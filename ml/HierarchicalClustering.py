
from sklearn.cluster import AgglomerativeClustering

def getHierarchicalClusters(words_vectors,NUMBER_OF_CLUSTERS):
    cluster = AgglomerativeClustering(n_clusters=NUMBER_OF_CLUSTERS, affinity='euclidean', linkage='ward')
    clusters= cluster.fit_predict(words_vectors)
    return clusters