import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy import sort, abs, array
from teacher_va.dataframe import TeacherDataFrame, StudentDataFrame
from teacher_va.method.lib import get_variance_student, get_variance_teacher_using_adjancent

tqdm.pandas()

# NAME SPACE
_singal_year = 'singal_year'
_signal_precision = 'signal_precision'
_charge_type_list = ['some', 'one']

get_kth_diag_is_one_matrix = \
    lambda rows, kth: np.diag(np.ones(shape=rows - kth), k=kth) + np.diag(np.ones(shape=rows - kth), k=-kth)


encode_lag_column_name = lambda lag: 'lag{0}'.format(lag)
decode_lag_column_name = lambda name: int(name.replace('lag', ''))


def signal_year_info(
        dfx: pd.DataFrame,
        signal_class_col, n_class_col,
        variance_resid=None, variance_theta=None, variance_epsilon=None, sigma_a0=None,
        charge_type='some'):
    if charge_type == 'some':
        dfx['signal_precision_cls'] = 1 / (variance_theta + variance_epsilon / dfx[n_class_col])
        signal_precision = sigma_a0 + (dfx['signal_precision_cls'] ** -1).sum()
    elif charge_type == 'one':
        signal_precision = variance_resid - variance_epsilon + variance_epsilon / dfx[n_class_col].values[0]
    else:
        raise ValueError
    return pd.Series({
        _singal_year: dfx[signal_class_col].mean(),
        _signal_precision: signal_precision
    })


def get_lag(dfx: pd.DataFrame, group_key, target, max_lag):
    grouped = dfx.groupby(group_key)
    for i in range(1, max_lag):
        dfx[encode_lag_column_name(i)] = grouped[target].shift(i)
    return dfx


def get_cov_base(length, time_list, sigma_as_dict):
    cov_base = np.zeros(shape=(length, length))
    for i, t1 in enumerate(time_list):
        for j, t2 in enumerate(time_list):
            if (i != j) & (i >= j):
                cov_base[i, j] = sigma_as_dict[abs(t1 - t2)]
                cov_base[j, i] = sigma_as_dict[abs(t1 - t2)]
                # cov_base += get_kth_diag_is_one_matrix(rows=length, kth=abs(i - j)) * sigma_as_dict[abs(t1 - t2)]  # trash
    return cov_base


def calc_mu_jt(dfx, cov_except_diag, gamma_full):
    """
    dfxの構造に依存している関数でしんどい
    """
    # is_exist_jt, a_minus_jt, precision_minus_jt = \
    #     wei.loc[teacher_id, ['is_exist_jt', 'a_minus_jt', 'precision_minus_jt']]
    is_exist_jt, a_minus_jt, precision_minus_jt = dfx[['is_exist_jt', 'a_minus_jt', 'precision_minus_jt']]
    if is_exist_jt.sum() == 0:
        return np.nan
    # import pdb;pdb.set_trace()
    cov_jt = cov_except_diag
    cov_jt[pd.np.diag_indices_from(cov_jt)] = precision_minus_jt
    sig_inv = np.linalg.inv(cov_jt[is_exist_jt, :][:, is_exist_jt])
    gamma = gamma_full[is_exist_jt]
    a_minus = a_minus_jt[is_exist_jt]
    return (sig_inv @ gamma).T @ a_minus


class CFREstimator:
    def __init__(self):
        self.teacher_effect = None

    def fit(
            self, sdf: StudentDataFrame,
            charge_type = 'one',
            fixed_effect_cols=None, factor_cols=None,
            is_custom_predict=False, custom_resid=None):
        # STEP1: 最終的に欲しいもの：Custom Residuals
        if is_custom_predict is False:
            from teacher_va.regression.teacher_value_added import TeacherFixedEffect
            tfe = TeacherFixedEffect(
                fixed_effect_cols=fixed_effect_cols,
                factor_cols=factor_cols
            )
            tfe.fit(
                sdf, outcome_col=sdf.outcome_col, covariate_cols=sdf.covariate_cols,
                teacher_id_col=sdf.teacher_id_col
            )
            sdf[sdf.resid_col] = tfe.residuals_without_fixed
        else:
            sdf[sdf.resid_col] = custom_resid
        sdf.set_signal_class()
        tdf = TeacherDataFrame.get_teacher_dataframe(
            sdf.get_df_class(),
            class_name_col=sdf.class_name_col,
            n_class_col=sdf.n_class_col,
            time_col=sdf.time_col,
            teacher_id_col=sdf.teacher_id_col,
            signal_class_col=sdf.signal_class_col,
        )
        # STEP2： 最終的に欲しいもの、_signal_precision、_singal_year、sigma_at_dict
        n_total_class = tdf.shape[0]
        n_student = sdf.shape[0]
        n_controls = len(sdf.covariate_cols)
        freedom_correction = (n_student - 1) / (n_student - n_total_class - n_controls + 1)
        variance_resid = sdf[sdf.resid_col].var()  # Var(Ait)
        variance_epsilon = get_variance_student(sdfx=sdf) * freedom_correction  # sigma_epsiplon
        sigma_a0 = get_variance_teacher_using_adjancent(tdf) if charge_type == 'some' else None
        variance_theta = variance_resid - variance_epsilon - sigma_a0 if charge_type == 'some' else None  # sigma_theta
        max_lag = tdf.groupby(tdf.teacher_id_col)[tdf.time_col].nunique().max()
        lag_cols = [encode_lag_column_name(i) for i in range(1, max_lag)]
        df_teacher_info = (
            tdf
            .groupby(tdf.teacher_time_cols)
            .progress_apply(
                signal_year_info,
                signal_class_col=tdf.signal_class_col, n_class_col=tdf.n_class_col,
                variance_resid=variance_resid, variance_epsilon=variance_epsilon,
                variance_theta=variance_theta, sigma_a0=sigma_a0,
                charge_type=charge_type
            )
            .reset_index()
            # lag情報とかここでしか使わんから、こっから下は切り離しだな
            .sort_values(tdf.teacher_time_cols)
            .pipe(get_lag, group_key=tdf.teacher_id_col, target=_singal_year, max_lag=max_lag)
            [tdf.teacher_time_cols + [_singal_year, _signal_precision] + lag_cols]
        )
        sigma_at = df_teacher_info[[_singal_year] + lag_cols].cov()
        sigma_at_dict = {decode_lag_column_name(lag_col): sigma_at.loc[_singal_year, lag_col] for lag_col in lag_cols}
        # STEP3
        time_list = sort(tdf[tdf.time_col].unique())
        result = pd.DataFrame()
        for time in time_list:
            time_list_except_t = time_list[time_list != time]
            time_diff_from_t = np.abs(time_list_except_t - time)  # こことsigma_at_dictはtimeが一様にあることを前提とした作りになっていて不快
            gammma_base = array([sigma_at_dict[x] for x in time_diff_from_t])
            cov_base = get_cov_base(
                length=len(time_list_except_t),
                time_list=time_list_except_t,
                sigma_as_dict=sigma_at_dict
            )
            result = (
                df_teacher_info
                .assign(
                    time_use=lambda dfx: dfx[tdf.time_col]
                )
                .set_index([tdf.teacher_id_col, tdf.time_col])
                [['time_use', _singal_year, _signal_precision]].unstack(-1)
                .assign(
                    # list(np.array)という構造でデータをもち、dfの各々のセルにarrayを格納する
                    is_exist_jt=lambda dfx: list(dfx['time_use'][time_list_except_t].notna().values),
                    a_minus_jt=lambda dfx: list(dfx[_singal_year][time_list_except_t].values),
                    precision_minus_jt=lambda dfx: list(dfx[_signal_precision][time_list_except_t].values)
                )
                [['is_exist_jt', 'a_minus_jt', 'precision_minus_jt']]
                .assign(
                    teacher_effect=lambda dfx: dfx.progress_apply(calc_mu_jt, axis=1, cov_except_diag=cov_base, gamma_full=gammma_base)
                )
                .reset_index()
                .assign(**{tdf.time_col: time})
                .append(result)
            )
        # result.set_index(['teacher_id', 'time'])['mu_jt'].unstack(-1).corr()
        # df.groupby(['teacher_id', 'time'])['teacher_effect'].mean().unstack(-1).corr()
        self.teacher_effect = result

