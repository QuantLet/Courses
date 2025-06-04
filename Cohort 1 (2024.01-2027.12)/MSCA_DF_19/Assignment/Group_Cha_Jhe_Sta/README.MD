# Reinforcement Learning for crypto portfolio allocation 

## Main Notebook Table Of Contents :
1. Data collection and EDA
    1. Data collection
        * One time API request
        * Loop API request
    2. EDA
        * Missing Values
        * Static Statistics Table
        * Dynamic Statistics Table
        * Time-series Price Plot
        * Time-series Cumulative Returns Plot
        * Correlation Matrix
        * Rolling Correlation Matrix
        * Boxplot
2. Modeling
    1. Pytorch Lightning formulation
        * Dataset creation
        * Model definition
        

    2. Synthetic data
        * Noisy Trends
        * Noisy Periodic
        * Noisy Seasonal Trends
    3. Model Training and Evaluation
        * Model Training
        * Benchmark and Backtest Preparation
        * Model Evaluation
        * Test TS training
        * Test Initial Assumptions

## Structure :

- The [final notebook](./20250527_Final_Notebook.ipynb) contains the code to generate all figures in the final presentation 

- Dataset (as parquet) and model checkpoints are provided to facilitate faster runtime of the main notebook

- additional models can be run using CLI :

````
python Run_RL.py --training buffer --c 0.0025 --logger 0 --model NBEATS --loss sharpe
````


