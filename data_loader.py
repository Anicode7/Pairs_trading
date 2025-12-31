import pandas as pd
import numpy as np
import os
import requests
import io

class DataLoader:
    """
    Handles data ingestion from yfinance for S&P 500 assets.
    """
    
    def __init__(self, start_date="2020-01-01", end_date="2023-01-01", cache_dir="data/inputs"):
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_snp500_tickers(self):
        """Scrapes S&P 500 tickers from Wikipedia.

        Uses a browser-like User-Agent to avoid 403 responses and wraps the
        HTML in `StringIO` when calling `pd.read_html` to silence a FutureWarning.
        Returns an empty list on failure and prints a clear message.
        """
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            resp.raise_for_status()
            # pandas.read_html will accept a file-like object for literal HTML
            table = pd.read_html(io.StringIO(resp.text))[0]
            tickers = table['Symbol'].tolist()
            # Clean ticker symbols (e.g., BRK.B -> BRK-B for yfinance)
            tickers = [t.replace('.', '-') for t in tickers]
            return tickers
        except Exception as e:
            print(f"Error fetching tickers from {url}: {e}")
            return []

    def fetch_data(self, tickers, missing_threshold=0.1, row_nonnull_thresh=0.95):
        """
        Downloads adjusted close prices for the provided tickers.

        Cleaning strategy:
        - Drop tickers with > `missing_threshold` fraction of missing values.
        - If that removes all tickers, attempt to fill small gaps via forward/backward fill
          and then drop tickers with >50% missing values.
        - Finally, drop rows (dates) that do not have at least `row_nonnull_thresh` fraction
          of non-missing ticker prices.

        Raises RuntimeError with a helpful message if no usable data remains.
        """
        file_path = os.path.join(self.cache_dir, "snp500_prices.csv")

        raw = None
        if os.path.exists(file_path):
            print(f"Loading data from cache: {file_path}")
            raw = pd.read_csv(file_path, index_col=0, parse_dates=True)
            # If cache looks invalid (no non-empty columns), re-download
            if raw.dropna(axis=1, how='all').shape[1] == 0:
                print("Cached data appears invalid (no non-empty columns). Re-downloading.")
                raw = None

        if raw is None:
            print(f"Downloading data for {len(tickers)} assets...")
            try:
                import yfinance as yf
            except ImportError:
                raise RuntimeError("yfinance is required to download market data. Install with `pip install yfinance`")

            # Download in chunks to avoid partial/malformed results on very large batches
            chunk_size = 100
            frames = []
            failed = []
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i : i + chunk_size]
                try:
                    chunk_raw = yf.download(chunk, start=self.start_date, end=self.end_date)
                except Exception as e:
                    print(f"Chunk download failed for {chunk[:5]}...: {e}")
                    failed.extend(chunk)
                    continue

                # Normalize chunk_raw similar to single-call behavior
                if isinstance(chunk_raw, pd.DataFrame) and 'Adj Close' in chunk_raw.columns:
                    chunk_df = chunk_raw['Adj Close']
                elif isinstance(chunk_raw, pd.DataFrame) and isinstance(chunk_raw.columns, pd.MultiIndex):
                    cols0 = chunk_raw.columns.get_level_values(0)
                    cols1 = chunk_raw.columns.get_level_values(1)
                    if 'Adj Close' in cols0:
                        chunk_df = chunk_raw.xs('Adj Close', axis=1, level=0, drop_level=True)
                    elif 'Adj Close' in cols1:
                        chunk_df = chunk_raw.xs('Adj Close', axis=1, level=1, drop_level=True)
                    else:
                        chunk_df = chunk_raw.copy()
                        chunk_df.columns = ['_'.join(map(str, c)).strip() for c in chunk_df.columns.values]
                else:
                    chunk_df = pd.DataFrame()

                if chunk_df.shape[1] > 0:
                    frames.append(chunk_df)
                else:
                    failed.extend(chunk)

            if frames:
                raw = pd.concat(frames, axis=1)
            else:
                raw = pd.DataFrame()

            if failed:
                # de-duplicate while preserving order
                seen = set()
                failed_unique = []
                for t in failed:
                    if t not in seen:
                        seen.add(t)
                        failed_unique.append(t)
                print(f"Failed downloads: {failed_unique}")

            # ensure DataFrame
            if not isinstance(raw, pd.DataFrame) or raw.shape[1] == 0:
                raise RuntimeError("Downloaded market data is empty or all chunked downloads failed. Check network/tickers.")

            # DEBUG: show shape & sample columns to help diagnose format issues
            print(f"Downloaded raw shape: {raw.shape}")
            try:
                sample_cols = list(raw.columns[:10])
            except Exception:
                sample_cols = None
            print(f"Sample columns (first 10): {sample_cols}")
            raw.to_csv(file_path)

        # If raw is empty or has no columns, fail early with helpful message
        if raw.empty or raw.shape[1] == 0:
            raise RuntimeError("Downloaded market data is empty. Ensure tickers are valid and yfinance/network are available.")

        # First pass: drop tickers with > missing_threshold missing values
        missing_frac = raw.isnull().mean()
        drop_cols = missing_frac[missing_frac > missing_threshold].index
        if len(drop_cols) > 0:
            print(f"Dropping {len(drop_cols)} tickers with >{missing_threshold*100:.0f}% missing values")
        data = raw.drop(columns=drop_cols)

        # If we dropped all tickers, attempt a less strict strategy: fill small gaps then drop >50%
        if data.shape[1] == 0:
            print("All tickers were dropped with the strict missingness threshold. Attempting to fill small gaps and relax threshold.")
            filled = raw.ffill().bfill()
            missing_frac2 = filled.isnull().mean()
            drop_cols2 = missing_frac2[missing_frac2 > 0.5].index
            if len(drop_cols2) > 0:
                print(f"After filling, dropping {len(drop_cols2)} tickers with >50% missing values")
            data = filled.drop(columns=drop_cols2)

        # Ensure we still have columns
        if data.shape[1] == 0:
            raise RuntimeError("No usable tickers after applying missing-data rules. Consider lowering thresholds or using a cached dataset.")

        # Drop rows (dates) that have too many missing values across tickers
        thresh = max(1, int(row_nonnull_thresh * data.shape[1]))
        data = data.dropna(axis=0, thresh=thresh)
        # Also remove any columns that are entirely NaN after row dropping
        data = data.dropna(axis=1, how='all')

        if data.empty or data.shape[1] == 0:
            raise RuntimeError("No usable data after cleaning. Try adjusting thresholds or check data source.")

        print(f"Data shape after cleaning: {data.shape}")
        return data