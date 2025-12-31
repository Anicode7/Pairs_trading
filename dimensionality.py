import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DimensionalityReducer:
    """
    Applies PCA to reduce asset dimensionality based on return series.
    Ref: Section 3.1 Sarmento & Horta (2020)
    """
    
    def __init__(self, variance_threshold=0.90):
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance_threshold)
        
    def fit_transform(self, price_data: pd.DataFrame):
        """
        Converts prices to returns, standardizes, and applies PCA.
        Returns the PCA embeddings (Assets x Components).
        """
        # Calculate Returns
        returns = price_data.pct_change().dropna()
        
        # Standardize (Critical for PCA)
        # Transpose so rows=Assets, cols=Time features
        X_std = self.scaler.fit_transform(returns.T)
        
        # Apply PCA
        embeddings = self.pca.fit_transform(X_std)
        
        print(f"PCA retained {self.pca.n_components_} components explaining {self.variance_threshold*100}% variance.")
        return embeddings