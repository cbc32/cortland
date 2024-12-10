# Charlie Bauer
# 9-20-23
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from statsmodels.tsa.api import SimpleExpSmoothing, Holt # classical statistics

from prophet import Prophet # classical statistics

from orbit.models import DLT, ETS, LGT # bayesian statistics

# from neuralprophet import NeuralProphet # deep learning

# from darts.models.forecasting.baselines import NaiveDrift # linear regression
# from darts.models.forecasting.auto_arima import AutoARIMA
# from darts.models.forecasting.theta import FourTheta
# from darts.models.forecasting.tbats import TBATS

# from kats.consts import TimeSeriesData
# from kats.models.holtwinters import HoltWintersParams, HoltWintersModel
# from kats.models.stlf import STLFParams, STLFModel

import warnings
warnings.filterwarnings("ignore")

def mean_ly_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    ly = train[train['ds'] > test['ds'].iloc[0] - DateOffset(years=1)]
    pred = pd.DataFrame(test['ds'])
    pred['y'] = ly['y'].mean()
    return pred

def values_ly_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    ly = train[train['ds'] > test['ds'].iloc[0] - DateOffset(years=1)]
    roll_ds = []
    roll_y = []
    for n in range(1, test['ds'].iloc[-1].year - test['ds'].iloc[0].year + 2):
        roll_ds += list((ly['ds'] + DateOffset(years=n)).values)
        roll_y += list(ly['y'].values)
    pred = pd.DataFrame(data={'ds': roll_ds, 'y': roll_y})
    return pred[:len(test)]

def forecast(train: pd.DataFrame, test: pd.DataFrame, season_len: int, n_lags=3):
    if train.columns[0] != 'date' or len(train.columns) < 2:
        return "train and test must be dataframes with columns ['date', numerical1, ..., numericaln]"
    
    n_pred = len(test)
    preds = dict() # predictions
    results = pd.DataFrame() # RMSEs
    for col in train.columns[1:]:
        train_col = train[['date', col]]
        train_col.columns = ['ds', 'y']
        test_col = test[['date', col]].reset_index(drop=True)
        test_col.columns = ['ds', 'y']

        train_series = train_col.set_index('ds')

        # train_darts = TimeSeries.from_dataframe(train_col, time_col='ds', value_cols='y')
        # train_kats = TimeSeriesData(time=train_col['ds'], value=train_col['y'])
        
        min_col = train_col['y'].min()
        max_col = train_col['y'].max()
        train_mm = train_col.copy()
        train_mm['minmax'] = (train_mm['y'] - min_col) / (max_col - min_col)

        models = {
            "Naive: Mean LY": "Custom",
            "Naive: Values LY": "Custom",
            "Linear": LinearRegression(),
            # "Exp. Smoothing": SimpleExpSmoothing(train_series, initialization_method='estimated'),
            "Holt-Winters": Holt(train_series,
                initialization_method='estimated'),
            "Holt-Winters Damped": Holt(train_series,
                damped_trend=True,
                initialization_method='estimated'),  
            # "Linear Trend": NaiveDrift(),
            # "Darts Theta 2": FourTheta(theta=2,
            #     seasonality_period=season_len),
            # "TBATS Damped": TBATS(use_box_cox=False,
            #       use_trend=True,
            #       use_damped_trend=True,
            #       seasonal_periods=[season_len]),
            # "AutoARIMA": AutoARIMA(m=season_len,
            #     max_p=4,
            #     max_q=4),
            # "STL Forecast": STLFModel(data=train_kats, params=STLFParams(
            #     method='linear',
            #     m=season_len)),
            # "HoltWinters Damped": HoltWintersModel(data=train_kats, params=HoltWintersParams(
            #     trend='add',
            #     damped=True,
            #     seasonal='add',
            #     seasonal_periods=season_len)),
            "Prophet": Prophet(daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=True),
            # "Prophet Mult.": Prophet(daily_seasonality=False,
            #     weekly_seasonality=False,
            #     yearly_seasonality=True,
            #     seasonality_mode='multiplicative'),
            # "NeuralProphet": NeuralProphet(daily_seasonality=False,
            #     weekly_seasonality=False,
            #     yearly_seasonality=True,
            #     collect_metrics=False),
            # "NeuralProphet Mult.": NeuralProphet(daily_seasonality=False,
            #     weekly_seasonality=False,
            #     yearly_seasonality=True,
            #     collect_metrics=False,
            #     seasonality_mode='multiplicative'),
            # "NeuralProphet w/AR": NeuralProphet(daily_seasonality=False, 
            #     weekly_seasonality=False,
            #     yearly_seasonality=True,
            #     collect_metrics=False,
            #     n_lags=n_lags,
            #     num_hidden_layers=5,
            #     d_hidden=5),
            # "ETS": ETS(date_col='ds',
            #     response_col='minmax',
            #     seasonality=season_len),
            "DLT": DLT(date_col='ds',
                response_col='minmax',
                seasonality=season_len),
            # "DLT No Minmax": DLT(date_col='ds',
            #     response_col='y',
            #     seasonality=season_len),
            "LGT": LGT(date_col='ds',
                response_col='minmax',
                seasonality=season_len)
        }
        
        results_row = dict()
        results_row["Mean"] = train[col].mean()
        results_row["St Dev"] = train[col].std()
        
        pred_df = pd.DataFrame(test['date'])
        
        for name, m in models.items():
            if name == "Naive: Mean LY":
                pred = mean_ly_predict(train_col, test_col)['y']
            elif name == "Naive: Values LY":
                pred = values_ly_predict(train_col, test_col)['y']
            elif name == "Linear":
                m.fit(np.array(train.index).reshape(-1,1), np.array(train[col].values).reshape(-1,1))
                pred = m.predict(np.array(test.index).reshape(-1,1))
            elif name in ["Exp. Smoothing", "Holt-Winters", "Holt-Winters Damped"]:
                fit = m.fit()
                pred = fit.forecast(n_pred).reset_index(drop=True)
            elif name in ["Linear Trend", "Darts Theta 2", "TBATS Damped", "AutoARIMA"]:
                m.fit(train_darts)
                pred = m.predict(n=n_pred).pd_dataframe()
            elif name in ["STL Forecast", "HoltWinters Damped"]:
                m.fit()
                pred = m.predict(steps=n_pred)['fcst']
            elif name in ["Prophet", "Prophet Mult."]:
                m.fit(train_col)
                pred = m.predict(test_col)['yhat']
            elif name in ["NeuralProphet", "NeuralProphet Mult."]:
                m.fit(train_col)
                pred = m.predict(test_col)['yhat1']
            elif name in ["NeuralProphet w/AR"]:
                m.fit(train_col)
                pred = m.predict(pd.concat([train_col.iloc[-n_lags:], test_col]))['yhat1'].dropna()
            elif name in ["ETS", "DLT", "LGT"]:
                if min_col == 0:
                    continue
                m.fit(train_mm)
                pred = m.predict(df=test_col)['prediction']
                pred = (pred * (max_col - min_col)) + min_col
            elif name in ["DLT No Minmax"]:
                m.fit(train_col)
                pred = m.predict(df=test_col)['prediction']
                
            results_row[name] = np.sqrt(mean_squared_error(test_col['y'], pred))
            
            pred_df[name] = pred
        
        if results.empty:
            results = pd.DataFrame(columns=results_row.keys())
        results.loc[col] = results_row
        
        preds[col] = pred_df

    if len(train.columns) == 2:
        return preds[train.columns[1]], results
    return preds, results


def forward_forecast(train: pd.DataFrame, horizon: int):
    if len(train.columns) != 2:
        return "train must be a dataframe with column dtypes [datetime, float]"
    
    train.columns = ['ds', 'y']
    
    forward = pd.date_range(train.set_index('ds').first('1D').index[0], periods=horizon, freq='1M')

    m = Prophet(daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=True)
    m.fit(train)
    return m.predict(forward)