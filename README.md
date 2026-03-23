Hello！
This is the final project for MATH629 of group AAPL

# LOB Price Movement Prediction with PCA and Market Factors

## Overview

This project studies short-term cryptocurrency price prediction using high-frequency limit order book (LOB) data. It combines deep learning models with dimensionality reduction and market state features to improve predictive performance and trading profitability.

The core idea is to integrate:

* Microstructure signals from the limit order book
* PCA-based feature compression
* Market state variables such as volatility and trend

We evaluate whether these components can enhance both prediction accuracy and economic performance in trading.

---

## Motivation and Financial Background

High-frequency cryptocurrency markets exhibit rapid order flow dynamics, fragmented liquidity, and strong short-term predictability driven by market microstructure effects.

The limit order book (LOB) provides detailed information about:

* Supply–demand imbalance
* Liquidity pressure
* Trader behavior

Empirical evidence suggests that:

* Order book depth
* Bid-ask imbalance
* Liquidity dynamics

can help forecast short-horizon price movements.

However:

* LOB data are high-dimensional and noisy
* Price formation also depends on broader market conditions

Therefore, incorporating:

* PCA-reduced LOB features
* Market state variables (volatility, trend)

may improve both predictive accuracy and financial insight.

---

## Research Questions

This project aims to answer the following:

1. Can deep learning models (CNN, LSTM, Transformer) extract predictive signals from LOB data?

2. Does PCA-based feature compression improve stability and accuracy by reducing noise and multicollinearity?

3. Do additional market state variables (volatility, trend) enhance prediction performance?

4. Can predicted signals generate profitable trading strategies after transaction costs?

---

## Data Description

### LOB Data

We use high-frequency cryptocurrency limit order book data containing:

* Multiple price levels on bid and ask sides
* Order flow information (market, limit, cancel)

At each timestamp, the dataset includes:

* Bid/ask distances from midprice
* Order volumes across multiple levels
* Market, limit, and cancellation activity

These data capture real-time liquidity and order flow dynamics.

### Alternative Dataset

If necessary, the FI-2010 dataset may be used as a benchmark. It provides:

* Normalized LOB data
* Predefined mid-price movement labels
* Standardized evaluation setup

---

### Market State Features

We augment LOB features with additional variables:

* Past returns (multiple horizons)
* Rolling realized volatility
* Momentum indicators
* Bid-ask spread
* Depth imbalance

These features capture:

* Market regimes
* Liquidity conditions
* Trend strength

---

## Methodology

### Problem Formulation

We formulate the task as a multi-class classification problem:

At each time step, predict whether the future mid-price will:

* Move up
* Move down
* Stay neutral

over a short prediction horizon.

---

### Feature Engineering

#### 1. Raw LOB Tensor

Construct input tensors from:

* Bid/ask levels
* Price distances
* Volume and order flow

This preserves spatial structure and order imbalance.

#### 2. PCA Feature Compression

Apply Principal Component Analysis (PCA) to LOB features to:

* Reduce dimensionality
* Remove noise
* Extract latent factors (e.g., liquidity pressure)

#### 3. Market Features

Add:

* Rolling volatility
* Returns
* Trend indicators
* Spread and imbalance

Final input combines:

* Raw LOB tensor
* PCA factors
* Market state variables

---

## Model Architecture

We compare three deep learning models:

### 1. CNN

* Captures spatial structure across price levels
* Learns local liquidity and imbalance patterns

### 2. LSTM

* Models temporal dependencies
* Captures order flow dynamics over time

### 3. Transformer

* Uses attention mechanisms
* Captures long-range dependencies and nonlinear interactions

All models are trained using cross-entropy loss for classification.

---

## Evaluation Framework

### Statistical Metrics

* Accuracy
* Precision
* Recall
* F1-score

These evaluate classification performance across:

* Up
* Down
* Neutral

---

### Economic Evaluation

We implement a trading strategy:

* Predict up → take long position
* Predict down → take short position
* Predict neutral → no position

Performance metrics:

* Cumulative returns
* Sharpe ratio
* Maximum drawdown
* Turnover

Transaction costs are included for realism.

---

## Expected Contributions

This project contributes:

1. Application of deep learning to high-frequency LOB data
2. PCA-based feature compression for noise reduction
3. Integration of market state variables for robustness

Overall, it bridges:

* Market microstructure modeling
* Regime-aware market analysis

while evaluating both predictive and economic performance.

---

## Project Deliverables

* Implementation of CNN, LSTM, Transformer models
* PCA-based feature analysis
* Model comparison (with vs without market features)
* Trading strategy backtesting
* Final report and presentation

