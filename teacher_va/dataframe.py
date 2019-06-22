import pandas as pd
from sklearn.linear_model import LinearRegression
from teacher_va.lib import predict_with_nan
from functools import reduce


def fillna_series_to_new_series(serires, new_series):
    return serires.mask(serires.isnull(), new_series)


class StudentSeries(pd.Series):
    @property
    def _constructor(self):
        return StudentSeries

    @property
    def _constructor_expanddim(self):
        return StudentDataFrame


class StudentDataFrame(pd.DataFrame):
    # normal properties
    _metadata = [
        'covariate_cols', 'outcome_col', 'class_name_col', 'time_col', 'teacher_id_col'
    ]
    # temporary properties
    _internal_names = pd.DataFrame._internal_names + ['tmp_prop']
    _internal_names_set = set(_internal_names)
    # global properties
    resid_col = 'resid'
    signal_class_col = 'signal_class'
    n_class_col = 'n_class_col'

    @property
    def _constructor(self):
        return StudentDataFrame

    @property
    def _constructor_sliced(self):
        return StudentSeries

    @property
    def class_cols(self):
        return [self.class_name_col, self.time_col]

    @classmethod
    def get_student_dataframe(
            cls,
            data: pd.DataFrame,
            covariate_cols: list,
            outcome_col: str,
            class_name_col: str,
            time_col: str,
            teacher_id_col: str,
    ):
        sdf = cls(data)
        sdf.covariate_cols = covariate_cols
        sdf.outcome_col = outcome_col
        sdf.class_name_col = class_name_col
        sdf.time_col = time_col
        sdf.teacher_id_col = teacher_id_col
        return sdf

    def fillna_teacher_id_from_class_cols(self):
        new_teacher_name = reduce(
            lambda series_x, series_y: series_x.astype(str).str.cat(series_y.astype(str)),
            [self[x] for x in self.class_cols]
        )
        self[self.teacher_id_col] = fillna_series_to_new_series(
            self[self.teacher_id_col],
            new_teacher_name
        )
        return self

    def set_predict_and_resid(self, fit_intercept=True):
        lr = LinearRegression(fit_intercept=fit_intercept)
        # import pdb;pdb.set_trace()
        lr.fit(X=self[self.covariate_cols], y=self[self.outcome_col])
        self[self.resid_col] = self[self.outcome_col] - predict_with_nan(self[self.covariate_cols], lr.predict)
        return self

    def set_signal_class(self):
        self[self.signal_class_col] = \
            self.groupby(self.class_cols)[self.resid_col].transform('mean')
        return self

    # def get_variance_student(self):
    #     """ Kane and Steinger type """
    #     return (self[self.resid_col] - self[self.signal_class_col]).var()
    #
    # def get_variance_classroom(self, variance_student, variance_teacher):
    #     """ Kane and Steinger type """
    #     return self[self.resid_col].var() - variance_student - variance_teacher

    def get_df_class(self):
        def get_class_info(dfx, teacher_id_col: str, signal_classroom_col: str, n_class_col:str):
            return (
                pd.Series({
                    teacher_id_col: dfx[teacher_id_col].values[0],  # 教員は1クラスでユニークのはず
                    signal_classroom_col: dfx[signal_classroom_col].values[0],  # クラスはユニークのはず
                    n_class_col: dfx.shape[0],
                })
            )
        return (
            self
            .groupby(self.class_cols)
            .apply(
                get_class_info,
                teacher_id_col=self.teacher_id_col,
                signal_classroom_col=self.signal_class_col,
                n_class_col=self.n_class_col
            )
            .reset_index()
        )


class TeacherSeries(pd.Series):

    @property
    def _constructor(self):
        return TeacherSeries

    @property
    def _constructor_expanddim(self):
        return TeacherDataFrame


class TeacherDataFrame(pd.DataFrame):
    # normal properties
    _metadata = [
        'class_name_col', 'time_col', 'teacher_id_col', 'signal_class_col', 'n_class_col',
    ]
    # temporary properties
    _internal_names = pd.DataFrame._internal_names + ['tmp_prop']
    _internal_names_set = set(_internal_names)
    # set prop
    signal_class_1lag_col = 'signal_class_col_1lag'
    kane_h_jt = 'h_jt'
    kane_weight_jt = 'weight_jt'
    @property
    def _constructor(self):
        return TeacherDataFrame

    @property
    def _constructor_sliced(self):
        return TeacherSeries

    @property
    def class_time_cols(self):
        return [self.class_name_col, self.time_col]

    @property
    def teacher_time_cols(self):
        return [self.teacher_id_col, self.time_col]

    @classmethod
    def get_teacher_dataframe(
            cls,
            data: pd.DataFrame,
            class_name_col: str,
            time_col: str,
            teacher_id_col: str,
            signal_class_col: str,
            n_class_col: str,
    ):
        tdf = cls(data)
        tdf.class_name_col = class_name_col
        tdf.time_col = time_col
        tdf.teacher_id_col = teacher_id_col
        tdf.signal_class_col = signal_class_col
        tdf.n_class_col = n_class_col
        return tdf

    @property
    def data_type(self):
        val1 = self.groupby(self.class_time_cols)[self.teacher_id_col].nunique().max()
        val2 = self.groupby([self.teacher_id_col, self.time_col])[self.class_name_col].nunique().max()
        print('1 class correspond {0} teacher'.format(val1))
        print('1 teacher correspond {0} class'.format(val2))
        return val1, val2

    # def get_variance_teacher(self):
    #     """
    #     Kane and Steinger type: 1 class correspond 1 teacher
    #     昨年度のclass signal との分散。だけどこれ1teacher:1クラスを前提にしていて、、、
    #     """
    #     self[self.signal_class_1lag_col] = get_1lag(
    #         self,
    #         value_col=self.signal_class_col,
    #         id_cols=self.teacher_id_col,
    #         time_col=self.time_col
    #     )
    #     # import pdb;pdb.set_tracxce()
    #     # Todo: 元論文と異なり勝手に0以上にしている
    #     return max(self[[self.signal_class_col, self.signal_class_1lag_col]].cov().values[0, 1], 0)
    #
    # def set_kane(self, variance_classroom, variance_student, variance_teacher):
    #     self['h_jt'] = 1 / (variance_classroom + (variance_student / self[self.n_class_col]))  # class signalの正確性（分散の逆数）
    #     self['weight_jt'] = self['h_jt'] / self.groupby(self.teacher_id_col)['h_jt'].transform('sum')  # 全体での比重
    #
    #     def get_teacher_info(dfx, v_t, signal_classroom_col=self.signal_class_col, weight_jt_col='weight_jt'):
    #         tva_no_shrinkage = (dfx[weight_jt_col] * dfx[signal_classroom_col]).sum()
    #         # import pdb;pdb.set_trace()
    #         tva = tva_no_shrinkage * (v_t / (v_t + (dfx['h_jt'].sum() ** -1)))
    #         return (
    #             pd.Series({
    #                 'tva_no_shrinkage': tva_no_shrinkage,
    #                 'tva': tva
    #             })
    #         )
    #     df_teacher = (
    #         self
    #         .groupby(self.teacher_id_col)
    #         .apply(get_teacher_info, v_t=variance_teacher)
    #     )
    #     return df_teacher
