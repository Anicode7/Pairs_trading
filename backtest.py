import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.data_loader import DataLoader

class Strategy:
    """
    A simple vectorized backtester for pairs trading.
    Strategy: Bollinger Bands / Z-Score Mean Reversion.
    """
    
    def __init__(self, entry_z=2.0, exit_z=0.5, window=20):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.window = window

    def run(self, pair_row, price_data):
        """
        Backtests a single pair.
        pair_row: A row from validated_pairs.csv (asset_1, asset_2, hedge_ratio)
        price_data: DataFrame of all asset prices
        """
        y_sym = pair_row['asset_1']
        x_sym = pair_row['asset_2']
        beta = pair_row['hedge_ratio']

        # 1. Construct Spread (Log-prices for Cointegration consistency)
        if y_sym not in price_data or x_sym not in price_data:
            return None
        
        y = np.log(price_data[y_sym])
        x = np.log(price_data[x_sym])
        spread = y - (beta * x)

        # 2. Calculate Z-Score
        # In a real model, use a rolling window to avoid lookahead bias
        roll_mean = spread.rolling(window=self.window).mean()
        roll_std = spread.rolling(window=self.window).std()
        z_score = (spread - roll_mean) / roll_std

        # 3. Generate Signals
        signals = pd.Series(index=spread.index, data=0)
        
        # Long Entry: Z < -Entry (Spread is too low, buy Y sell X)
        signals[z_score < -self.entry_z] = 1 
        
        # Short Entry: Z > +Entry (Spread is too high, sell Y buy X)
        signals[z_score > self.entry_z] = -1 
        
        # Exit: |Z| < Exit (Spread reverted to mean)
        signals[abs(z_score) < self.exit_z] = 0
        
        # Forward fill positions (hold until exit signal)
        # Logic: 
        # - If signal is 1 or -1, we take that position.
        # - If signal is 0, we close.
        # - If signal is NaN (no breach), we keep previous position.
        # We need a loop or a smart ffill logic. 
        # A simple "Signal State" approach:
        positions = pd.Series(index=spread.index, data=np.nan)
        positions[z_score < -self.entry_z] = 1   # Long
        positions[z_score > self.entry_z] = -1   # Short
        positions[abs(z_score) < self.exit_z] = 0 # Flat
        
        # Forward fill to hold positions
        positions = positions.ffill().fillna(0)

        # 4. Calculate PnL
        # Spread PnL = Position * (Spread_change)
        # Note: This is an approximation. Real PnL depends on Y and X dollar values.
        # Approx: Ret ≈ Position * (dY - beta*dX)
        # More precise: We are trading the spread unit directly.
        spread_ret = spread.diff()
        pnl = positions.shift(1) * spread_ret
        
        return pnl.cumsum()

def main():
    print("--- Running Backtest Model ---")
    
    # 1. Load the Pairs you selected in Step 1
    pairs_path = "data/outputs/validated_pairs.csv"
    if not os.path.exists(pairs_path):
        print("Error: No validated pairs found. Run main.py first!")
        return
        
    pairs_df = pd.read_csv(pairs_path)
    print(f"Loaded {len(pairs_df)} pairs for backtesting.")

    # 2. Load Data (Using your existing DataLoader)
    loader = DataLoader(start_date="2020-01-01", end_date="2024-01-01")
    # Get all unique tickers from the pairs
    tickers = list(set(pairs_df['asset_1']).union(set(pairs_df['asset_2'])))
    
    print(f"Fetching data for {len(tickers)} assets in the pairs...")
    price_data = loader.fetch_data(tickers)

    # 3. Run Strategy
    strategy = Strategy(entry_z=2.0, exit_z=0.5, window=30)
    
    results = pd.DataFrame()
    
    for idx, row in pairs_df.iterrows():
        pair_name = f"{row['asset_1']}-{row['asset_2']}"
        print(f"Backtesting {pair_name}...", end=" ")
        
        equity_curve = strategy.run(row, price_data)
        
        if equity_curve is not None:
            results[pair_name] = equity_curve
            final_return = equity_curve.iloc[-1]
            print(f"Return: {final_return:.4f}")
        else:
            print("Skipped (Data missing)")

    # 4. Aggregate Results
    if not results.empty:
        # Equal weight portfolio of all pairs
        portfolio_curve = results.mean(axis=1)
        
        print("\n--- Portfolio Summary ---")
        print(f"Total Cumulative Return: {portfolio_curve.iloc[-1]*100:.2f}%")
        
        # Simple Sharpe Ratio (assuming daily data)
        daily_ret = portfolio_curve.diff()
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
        print(f"Sharpe Ratio: {sharpe:.2f}")

        # Plot
        try:
            portfolio_curve.plot(title="Strategy: Portfolio Equity Curve")
            plt.ylabel("Cumulative Log-Return")
            plt.show()
            print("Plot generated.")
        except:
            print("Could not generate plot (check matplotlib backend).")
            
        # Save output
        results.to_csv("data/outputs/backtest_results.csv")
    else:
        print("No trades generated.")

if __name__ == "__main__":
    main()