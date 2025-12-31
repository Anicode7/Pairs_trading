# Pairs Selection Pipeline (Sarmento & Horta, 2020)

This project implements the pair selection framework from the paper *"Enhancing a Pairs Trading strategy with the application of Machine Learning"*. It uses Unsupervised Learning (PCA + OPTICS) to reduce the search space and strictly verifies statistical equilibrium (Cointegration, Hurst Exponent, Half-Life).

## Structure
- `src/data_loader.py`: Fetches S&P 500 data from Yahoo Finance.
- `src/dimensionality.py`: Performs PCA on asset returns.
- `src/clustering.py`: Clusters assets using OPTICS to find candidate groups.
- `src/pair_selection.py`: Validates pairs using Engle-Granger, Hurst, and OU process checks.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt