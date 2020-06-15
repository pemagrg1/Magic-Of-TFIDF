"""
    clustering the ham or spam data with cluster as 2
"""

import os
import collections
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
retval = os.getcwd()
os.chdir( "/".join(retval.split("/")[:-1]))
retval = os.getcwd()

class TextClustering:
    def __init__(self, sents, nclusters,visualization=False):
        """

        :param sents: corpus
        :param nclusters: number of clusters
        :param visualization: True is you want to visualize else mark it as False
        """
        self.sents = sents
        self.nclusters = nclusters
        self.visualization = visualization

    def kmeans_clustering(self):
        """

        :return:  cluster dict after applying kmeans_clustering algorithm
        """
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.sents)
        kmeans = KMeans(n_clusters=self.nclusters)
        kmeans.fit(tfidf_matrix)

        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
        return dict(clusters)

    def affinity_clustering(self):
        """

        :return: cluster dict after applying affinity_clustering algorithm
        """
        tfidf_vectorizer = TfidfVectorizer()
        tf_idf_matrix = tfidf_vectorizer.fit_transform(self.sents)
        similarity_matrix = (tf_idf_matrix * tf_idf_matrix.T).A
        affinity_propagation = AffinityPropagation(affinity="precomputed",
                                                   damping=0.5)
        affinity_propagation.fit(similarity_matrix)

        clusters = collections.defaultdict(list)
        for i, label in enumerate(affinity_propagation.labels_):
            clusters[label].append(i)
        return dict(clusters)

    def Agglomerative_clustering(self):
        """

        :return: cluster dict after applying Agglomerative_clustering algorithm
        """
        tfidf_vectorizer = TfidfVectorizer()
        tf_idf_matrix = tfidf_vectorizer.fit_transform(self.sents)
        similarity_matrix = (tf_idf_matrix * tf_idf_matrix.T).A
        agglomerativeclustering = AgglomerativeClustering()
        agglomerativeclustering.fit(similarity_matrix)

        clusters = collections.defaultdict(list)
        for i, label in enumerate(agglomerativeclustering.labels_):
            clusters[label].append(i)
        return dict(clusters)


def main():
    data = pd.read_csv("data/hamOrspam.csv", encoding="latin-1")
    nclusters= 2
    sent_clus = TextClustering(sents=data["text"], nclusters=nclusters, visualization=True)
    clusters = sent_clus.kmeans_clustering()
    """
    Other alog you can try:
            clusters = sent_clus.kmeans_clustering()
            clusters = sent_clus.Agglomerative_clustering()
            clusters = sent_clus.affinity_clustering()
    """
    df = pd.DataFrame(columns=['sentence','label'])
    for cluster in range(len(clusters.keys())):
        print("cluster ",cluster,":")
        for i,sentence in enumerate(clusters[cluster]):
            print("\tsentence ",i,": ",data["text"][sentence])
            df = df.append({'sentence': data["text"][sentence], 'label': cluster},
                           ignore_index=True)
    print(df.head())


if __name__ == '__main__':
    main()