import pandas as pd
import os
from src.data_loader import DataLoader
from src.dimensionality import DimensionalityReducer
from src.clustering import OpticsClustering
from src.pair_selection import PairSelector

def main():
    print("--- Starting Pairs Selection Pipeline ---")
    
    # 1. Load Data
    loader = DataLoader(start_date="2020-01-01", end_date="2024-01-01")
    tickers = loader.get_snp500_tickers()
    # Limit for testing speed (remove [:100] to run full list)
    print(f"Fetched {len(tickers)} tickers. downloading data...")
    data = loader.fetch_data(tickers)
    
    # 2. PCA
    reducer = DimensionalityReducer(variance_threshold=0.90)
    embeddings = reducer.fit_transform(data)
    
    # 3. Clustering
    # min_samples set to 3 to be lenient for smaller datasets
    clusterer = OpticsClustering(min_samples=3)
    clusters = clusterer.fit_predict(embeddings, data.columns)
    
    # 4. Pair Selection
    selector = PairSelector(
        coint_threshold=0.05,
        hurst_threshold=0.5,
        min_half_life=5,    # Minimum 5 days for robust mean reversion
        max_half_life=252,
        min_crossings=8
    )
    
    results = selector.validate_pairs(data, clusters)
    
    # 5. Save Output
    output_dir = "data/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    if not results.empty:
        output_path = os.path.join(output_dir, "validated_pairs.csv")
        results.to_csv(output_path, index=False)
        print(f"\nSuccess! Found {len(results)} pairs.")
        print(f"Results saved to {output_path}")
        print(results[['asset_1', 'asset_2', 'cluster_id', 'p_value', 'half_life']].head())
    else:
        print("\nNo pairs passed the strict statistical validation.")

if __name__ == "__main__":
    main()