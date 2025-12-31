import pandas as pd
from sklearn.cluster import OPTICS

class OpticsClustering:
    """
    Applies OPTICS clustering to identify groups of similar assets.
    Ref: Section 3.2 Sarmento & Horta (2020)
    """
    
    def __init__(self, min_samples=5, metric='euclidean'):
        self.model = OPTICS(min_samples=min_samples, metric=metric)
        
    def fit_predict(self, embeddings, asset_names):
        """
        Fits OPTICS and returns a Series mapping Asset -> Cluster Label.
        Label -1 indicates noise (outliers).
        """
        labels = self.model.fit_predict(embeddings)
        
        results = pd.Series(labels, index=asset_names, name="Cluster_ID")
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"OPTICS found {n_clusters} clusters and {n_noise} noise points.")
        return results