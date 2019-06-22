import pandas as pd


def predict_with_nan(x: pd.DataFrame, func_predict):
    slicing = pd.np.isnan(x).sum(axis=1) == 0
    y_hat = pd.np.ones(x.shape[0])
    y_hat[:] = pd.np.nan
    y_hat[slicing] = func_predict(x[slicing])
    return y_hat


def get_1lag(dfx, value_col, id_cols, time_col, ascending=True):
    return (
        dfx
        .sort_values(time_col, ascending=ascending)
        .groupby(id_cols)
        [value_col]
        .shift(1)
        .loc[dfx.index]
    )