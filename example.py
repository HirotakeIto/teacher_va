def main():
    import pandas as pd
    from rpy2.robjects import pandas2ri
    import rpy2.robjects as ro
    from teacher_va.estimate import TeacherValueAddedEstimator, StudentDataFrame

    def give_group_name(dfx, keys, group_name_col='name'):
        aa = dfx[keys].drop_duplicates().dropna()
        aa[group_name_col] = 1
        aa[group_name_col] = aa[group_name_col].cumsum()
        return (
            dfx
            .merge(aa, on=keys, how='left')
        )

    pd.set_option("display.max_columns", 101)
    df = (
        pd.read_csv('data/math_teacher.csv')
        .pipe(
            give_group_name,
            keys=['year_prime','school_id_prime', 'grade_prime', 'class_prime'],
            group_name_col = 'name'
        )
        [['mst_id', 'name', 'math_level_prime', 'math_level', 'teacher_id', 'year_prime']]
        .dropna(subset=['math_level_prime', 'math_level', 'year_prime'])
        # .pipe(lambda dfx: pd.get_dummies(dfx, columns=['mst_id'], sparse=True, prefix='mstid'))
    )

    # printできないような出力をコールするとしくるから要注意
    pandas2ri.activate()
    r_df = ro.r.assign("r_df", pandas2ri.py2rpy(df))
    aa = ro.r("res <- lfe::felm(math_level ~ math_level_prime | as.factor(mst_id) |0  |0, r_df)")
    bb = ro.r("res$residuals")

    sdf = (
        StudentDataFrame.get_student_dataframe(
            data=df,
            covariate_cols=['math_level_prime'],
            outcome_col='math_level',
            class_name_col='name',  # 1 teacher: 1 class
            time_col='year_prime',
            teacher_id_col='teacher_id',
        )
    )
    sdf.fillna_teacher_id_from_class_cols()
    tvtva = TeacherValueAddedEstimator(effect_type='time_fixed')
    tvtva.fit(sdf=sdf, is_custom_predict=True, custom_resid=bb)
    teacher_effect = tvtva.teacher_effect
    tvtva = TeacherValueAddedEstimator(effect_type='time_varing')
    tvtva.fit(sdf=sdf, is_custom_predict=True, custom_resid=bb)
    teacher_effect2 = tvtva.teacher_effect
    """
         teacher_id  tva_no_shrinkage       tva
    0       23568.0         -0.027097 -0.016627
    1       26958.0         -0.127684 -0.057976
    2       26969.0         -0.326631 -0.146207
    3       27926.0          0.000000  0.000000
    4       71681.0          0.053966  0.024156
    5       83914.0          0.292937  0.224389
    6       87386.0         -0.353937 -0.157323
    7       87622.0         -0.086926 -0.053977
    """