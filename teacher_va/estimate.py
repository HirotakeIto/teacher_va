from teacher_va.lib import get_1lag
from pandas import DataFrame, Series
from numpy import stack
from teacher_va.dataframe import StudentDataFrame, TeacherDataFrame
from functools import reduce


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
    return max(x_df.dropna().cov().values[0, 1], 0)
    return abs(x_df.dropna().cov().values[0, 1])


def get_variance_classroom(sdfx: StudentDataFrame, variance_student, variance_teacher):
    """ Kane and Steinger type """
    return sdfx[sdfx.resid_col].var() - variance_student - variance_teacher


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


class TeacherValueAddedEstimator:
    def __init__(self, effect_type='time_fixed'):
        # """ここも任意にしてbuildとかも自由にさせた方がいいよね。"""
        # self.pipeline = []
        if effect_type=='time_fixed':
            self.get_variance_student = get_variance_student
            self.get_variance_teacher = get_variance_teacher_using_1lag
            self.get_variance_classroom = get_variance_classroom
            self. extract_teacher_effect_from_signal = extract_teacher_effect_from_signal
        if effect_type=='time_varing':
            self.get_variance_student = get_variance_student
            self.get_variance_teacher = get_variance_teacher_using_adjancent
            self.get_variance_classroom = get_variance_classroom
            self. extract_teacher_effect_from_signal = extract_teacher_effect_from_signal
        self.effect_type = effect_type
        self.teacher_effect = None
        self.variance_student = None
        self.variance_teacher = None
        self.variance_classroom = None


    def fit(self, sdf: StudentDataFrame, is_custom_predict = False, custom_resid = None):
        if is_custom_predict is False:
            sdf.set_predict_and_resid()
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
        variance_student = self.get_variance_student(sdf)
        variance_teacher = self.get_variance_teacher(tdf)
        variance_classroom = self.get_variance_classroom(sdf, variance_student, variance_teacher)
        effect_by = tdf.teacher_time_cols if self.effect_type == 'time_varing' else tdf.teacher_id_col
        self.teacher_effect = self.extract_teacher_effect_from_signal(
            tdf=tdf,
            variance_teacher=variance_teacher,
            variance_student=variance_student,
            variance_classroom=variance_classroom,
            effect_by= effect_by
        )
        self.variance_student = variance_student
        self.variance_teacher = variance_teacher
        self.variance_classroom = variance_classroom
        return self
