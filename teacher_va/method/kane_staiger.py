from pandas import Series

from teacher_va.dataframe import TeacherDataFrame


def extract_teacher_effect_from_signal(
        tdf: TeacherDataFrame, variance_classroom, variance_student, variance_teacher, effect_by):
    precision_col = 'h_jt'  # signal precision
    weight_col = 'weight_jt'  # signal weight

    def get_teacher_info(dfx, v_t, signal_classroom_col, weight_col=weight_col, precision_col = precision_col):
        tva_no_shrinkage = (dfx[weight_col] * dfx[signal_classroom_col]).sum()
        # import pdb;pdb.set_trace()
        tva = tva_no_shrinkage * (v_t / (v_t + (dfx[precision_col].sum() ** -1)))
        return (
            Series({
                'tva_no_shrinkage': tva_no_shrinkage,
                'tva': tva
            })
        )

    return (
        tdf
        .assign(**{
            precision_col: lambda tdfx: 1 / (variance_classroom + (variance_student / tdfx[tdfx.n_class_col])),
            weight_col: lambda tdfx: tdfx[precision_col] / tdfx.groupby(effect_by)[precision_col].transform('sum')
        })
        .groupby(effect_by)
        .apply(
            get_teacher_info,
            v_t=variance_teacher,
            signal_classroom_col=tdf.signal_class_col
        )
        .reset_index()
    )
