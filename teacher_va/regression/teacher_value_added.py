import pandas as pd
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from teacher_va.regression.use_r import get_add_str_from_str_list


def create_felm_formula(target, covariate_cols, fixed_effect_cols, factor_cols):
    templete = '{target} ~ {covariate_str} | {fixed_str} | 0 | 0 '
    covariate_str = get_add_str_from_str_list(covariate_cols, factor_cols) if len(covariate_cols) > 0 else ' 0 '
    fixed_str = get_add_str_from_str_list(fixed_effect_cols, factor_cols) if len(fixed_effect_cols) > 0 else ' 0 '
    return templete.format(
        target=target,
        covariate_str=covariate_str,
        fixed_str=fixed_str
    )


class TeacherFixedEffect():
    """
    Using R.
    """
    def __init__(self, fixed_effect_cols=None, factor_cols=None):
        self.fixed_effect_cols = fixed_effect_cols if fixed_effect_cols is not None else []
        self.factor_cols = factor_cols if factor_cols is not None else []
        self.r = ro.r
        self.effect = None
        self.residuals_without_fixed = None
        self.residuals_with_fixed = None

    def fit(
            self, dfx: pd.DataFrame, outcome_col,  covariate_cols, teacher_id_col,**argv
    ):
        covariate_cols_except_fixed = [x for x in covariate_cols if x not in self.fixed_effect_cols]
        fixed_effect_cols_plus_tid = [teacher_id_col] + self.fixed_effect_cols
        dropna_subset_cols = [outcome_col] + covariate_cols + fixed_effect_cols_plus_tid
        formula = create_felm_formula(outcome_col, covariate_cols_except_fixed, fixed_effect_cols_plus_tid, self.factor_cols)
        pandas2ri.activate()
        df_use = dfx.dropna(subset=dropna_subset_cols)
        _res1 = self.r.assign("r_df", pandas2ri.py2rpy(df_use))
        _res2 = self.r("res <- lfe::felm({formula}, r_df)".format(formula=formula))
        bb = self.r("lfe::getfe(res)")
        self.effect = bb
        self.residuals_without_fixed = pd.Series(index=dfx.index)
        self.residuals_without_fixed.loc[df_use.index ,] = self.r("res$r.residuals")[:, 0]
        self.residuals_with_fixed = pd.Series(index=dfx.index)
        self.residuals_with_fixed.loc[df_use.index,] = self.r("res$residuals")[:, 0]
        pandas2ri.deactivate()





def estimation_fixed_effect(outcome_col, time_col, teacher_id_col, class_name_col, covariate_cols, fixed_effect_cols, **argv):

    def create_formula(target, covariate_cols, fixed_effect_cols):
        templete = '{target} ~ {covariate_str} | {fixed_str} | 0 | 0 '
        covariate_str = get_add_str_from_str_list(covariate_cols) if len(covariate_cols) > 0 else ' 0 '
        fixed_str = get_add_str_from_str_list(fixed_effect_cols) if len(fixed_effect_cols) > 0 else ' 0 '
        return templete.format(
            target=target,
            covariate_str=covariate_str,
            fixed_str=fixed_str
        )

    use_cols = [outcome_col, time_col, teacher_id_col, class_name_col] + covariate_cols + fixed_effect_cols
    dropna_subset_cols = [outcome_col, time_col, class_name_col] + covariate_cols + fixed_effect_cols
    fixed_effect_cols_plus_tid = [teacher_id_col] + fixed_effect_cols
    formula = create_formula(outcome_col, covariate_cols, fixed_effect_cols_plus_tid)
    # start
    pd.set_option("display.max_columns", 101)
    df_res = (
        pd.read_csv('./notebook/toda_teacher/df.csv')
            # 小学校だけで推定
            .pipe(lambda dfx: dfx.loc[dfx['year_prime'] >= 2015])
            .pipe(lambda dfx: dfx.loc[dfx['school_id_prime'] < 30000])
        [use_cols]
            .dropna(subset=dropna_subset_cols)
        # .pipe(lambda dfx: pd.get_dummies(dfx, columns=['mst_id'], sparse=True, prefix='mstid'))
    )

    pandas2ri.activate()
    _res1 = ro.r.assign("r_df", pandas2ri.py2rpy(df_res))
    _res2 = ro.r("res <- lfe::felm({formula}, r_df)".format(formula=formula))
    bb = ro.r("lfe::getfe(res)")
    pandas2ri.deactivate()

    effect = (
        bb.reset_index()
        .pipe(lambda dfx: dfx.loc[dfx['fe']==teacher_id_col, ['index', 'effect']])
        .assign(
            **{teacher_id_col: lambda dfx:
                dfx['index'].str.extract('{0}\.(.+)'.format(teacher_id_col)).astype(df_res[teacher_id_col].dtype)}
        )
        [[teacher_id_col, 'effect']]
        .rename(columns = {'effect': 'tva'})
    )
    return effect


