from numpy import stack
from pandas import DataFrame

from teacher_va import StudentDataFrame
from teacher_va.dataframe import TeacherDataFrame
from teacher_va.lib import get_1lag


def get_variance_student(sdfx: StudentDataFrame):
    """ Kane and Steinger type """
    return (sdfx[sdfx.resid_col] - sdfx[sdfx.signal_class_col]).var()


def get_variance_teacher_using_adjancent(tdfx: TeacherDataFrame):
    return (
        tdfx
        .assign(
            adjancent_siganl=lambda tdfxx: tdfxx.groupby(tdfxx.teacher_time_cols)[tdfxx.signal_class_col].shift(1)
        )
        .pipe(lambda tdfxx: tdfxx[[tdfxx.signal_class_col, 'adjancent_siganl']])
        .cov()
        .values[0, 1]
    )


def get_variance_teacher_using_1lag(tdfx: TeacherDataFrame):
    """
    Kane and Steinger type: 1 class correspond 1 teacher
    昨年度のclass signal との分散。だけどこれ1teacher:1クラスを前提にしていて、、、
    """
    # import pdb;pdb.set_tracxce()
    signal_1_lag = get_1lag(
        tdfx,
        value_col=tdfx.signal_class_col,
        id_cols=tdfx.teacher_id_col,
        time_col=tdfx.time_col
    )
    # Todo: 元論文と異なり勝手に0以上にしている: これバリむずい
    x_df = DataFrame(stack((tdfx[tdfx.signal_class_col], signal_1_lag), axis=1))
    # return max(x_df.dropna().cov().values[0, 1], 0)
    return abs(x_df.dropna().cov().values[0, 1])


def get_variance_classroom(sdfx: StudentDataFrame, variance_student, variance_teacher):
    """ Kane and Steinger type """
    return sdfx[sdfx.resid_col].var() - variance_student - variance_teacher