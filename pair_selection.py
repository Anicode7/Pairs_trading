import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from itertools import combinations

class PairSelector:
    """
    Validates pairs within clusters using statistical tests.
    Ref: Section 3.3 Sarmento & Horta (2020)
    """
    
    def __init__(self, 
                 coint_threshold=0.05, 
                 hurst_threshold=0.5, 
                 min_half_life=1, 
                 max_half_life=252,
                 min_crossings=12):
        self.coint_threshold = coint_threshold
        self.hurst_threshold = hurst_threshold
        self.min_hl = min_half_life
        self.max_hl = max_half_life
        self.min_crossings = min_crossings

    def validate_pairs(self, price_data, clusters):
        """
        Iterates over clusters and validates pairs.
        """
        validated_pairs = []
        unique_labels = set(clusters.unique())
        
        if -1 in unique_labels:
            unique_labels.remove(-1) # Ignore noise
            
        print(f"Scanning {len(unique_labels)} clusters for candidates...")
        
        for label in unique_labels:
            cluster_assets = clusters[clusters == label].index.tolist()
            if len(cluster_assets) < 2:
                continue
                
            for a1, a2 in combinations(cluster_assets, 2):
                pair_stats = self._check_pair(price_data[a1], price_data[a2])
                
                if pair_stats:
                    pair_stats.update({
                        'asset_1': a1, 
                        'asset_2': a2, 
                        'cluster_id': label
                    })
                    validated_pairs.append(pair_stats)
                    
        return pd.DataFrame(validated_pairs)

    def _check_pair(self, y_series, x_series):
        # Log prices for cointegration
        log_y = np.log(y_series)
        log_x = np.log(x_series)
        
        # 1. Cointegration (Bidirectional)
        # We test Y~X and X~Y, pick the one with lower t-stat (stronger rejection of null)
        t1, p1, _ = coint(log_y, log_x)
        t2, p2, _ = coint(log_x, log_y)
        
        if p1 < p2:
            p_val, dependent, independent = p1, log_y, log_x
            is_y_dependent = True
        else:
            p_val, dependent, independent = p2, log_x, log_y
            is_y_dependent = False
            
        if p_val > self.coint_threshold:
            return None
            
        # Calculate Spread
        model = sm.OLS(dependent, sm.add_constant(independent))
        res = model.fit()
        hedge_ratio = res.params.iloc[1]
        spread = dependent - hedge_ratio * independent
        
        # 2. Hurst Exponent
        hurst = self._calculate_hurst(spread)
        if hurst >= self.hurst_threshold:
            return None
            
        # 3. Half Life
        half_life = self._calculate_half_life(spread)
        if not (self.min_hl <= half_life <= self.max_hl):
            return None
            
        # 4. Mean Crossings
        crossings = self._count_crossings(spread)
        if crossings < self.min_crossings:
            return None
            
        return {
            'p_value': p_val,
            'hedge_ratio': hedge_ratio,
            'dependent_is_asset_1': is_y_dependent, # True if asset_1 is Y
            'hurst': hurst,
            'half_life': half_life,
            'crossings': crossings
        }

    def _calculate_hurst(self, ts):
        """Returns the Hurst Exponent of the time series."""
        lags = range(2, 20)
        tau = [np.sqrt(np.std(ts.diff(lag))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0 

    def _calculate_half_life(self, spread):
        """Calculates Ornstein-Uhlenbeck half-life."""
        spread_lag = spread.shift(1)
        spread_ret = spread - spread_lag
        spread_ret = spread_ret.dropna()
        spread_lag = spread_lag.dropna()
        
        model = sm.OLS(spread_ret, sm.add_constant(spread_lag))
        res = model.fit()
        lam = res.params.iloc[1]
        
        if lam >= 0: return np.inf
        return -np.log(2) / lam

    def _count_crossings(self, spread):
        """Counts mean crossings per year (approx)."""
        mu = spread.mean()
        centered = spread - mu
        # Number of sign changes
        return len(np.where(np.diff(np.sign(centered)))[0])