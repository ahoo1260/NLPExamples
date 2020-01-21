print(__doc__)


from sklearn.cluster import KMeans


def clusterEmbededData(embeddedMatrix):
    kmeans=KMeans(n_clusters=5, random_state=0).fit(embeddedMatrix)
    y_kmeans=kmeans.fit_predict(embeddedMatrix)

    return y_kmeans




