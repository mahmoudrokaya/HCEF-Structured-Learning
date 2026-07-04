
### Experiment 5: Lookback Window Sensitivity Analysis

The fifth experiment evaluated the sensitivity of the full Temporal HCE-F LSTM model to the length of historical temporal context. Four sliding-window datasets were evaluated using 3-day, 7-day, 14-day, and 30-day lookback periods. Each dataset used the same target definition: prediction of the current day's mean temperature from previous multivariate daily climate observations.

For each lookback window, the flattened daily-window representation was reshaped into a sequence tensor preserving chronological order. The same full Temporal HCE-F architecture, consisting of daily input projection, LSTM temporal encoding, residual-gated representation, contrastive projection, and two-head ensemble regression, was trained under the same chronological train-validation-test split and optimization settings. Forecasting performance was compared using MAE, RMSE, SMAPE, MAPE, and R2. This experiment tested whether shorter or longer temporal histories provided better predictive information and assessed the robustness of the proposed model across different temporal memory lengths.
