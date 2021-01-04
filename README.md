# XGBoost-For-BTC
xgboost model for bitcoin trend prediction


This model can be used as pre-trained model,  it trained in 1.000.000 1-min BTC/USDT OHLC data

Our model features:

Previous  Open High Low Close for last 5 min, 15 min 30 min, 1h and 4h data.
Previous Log change for Open High Low Close  values  for  last 5 min, 15 min 30 min, 1h and 4h data.


hyper parameters for Our Model:

min_child_weight=10
gamma=1
subsample=1
colsample_bytree=0.8
max_depth=3


Our test Accuracy Score : 65%
