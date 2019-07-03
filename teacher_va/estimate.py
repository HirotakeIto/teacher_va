from teacher_va.dataframe import StudentDataFrame, TeacherDataFrame
from teacher_va.method.kane_staiger import extract_teacher_effect_from_signal
from teacher_va.method.lib import get_variance_student, get_variance_teacher_using_adjancent, \
    get_variance_teacher_using_1lag, get_variance_classroom


class TeacherValueAddedEstimator:
    """
    Kane and  Staiger type
    """
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
