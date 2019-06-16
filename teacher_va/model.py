import pandas as pd
from sklearn.linear_model import LinearRegression

def predict_with_nan(X, func_predict):
    slicing = pd.np.isnan(X).sum(axis=1) == 0
    y_hat = pd.np.ones(X.shape[0])
    y_hat[:] = pd.np.nan
    y_hat[slicing] = func_predict(X[slicing])
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
            *,
            student_time_col = 'student_time',
            student_class_col = 'student_class',
            **argv
    ):
        sdf = cls(data)
        sdf.covariate_cols = covariate_cols
        sdf.outcome_col = outcome_col
        sdf.class_name_col = class_name_col
        sdf.time_col = time_col
        sdf.teacher_id_col = teacher_id_col
        return sdf

    def set_predict_and_resid(self):
        lr = LinearRegression(fit_intercept=True)
        # import pdb;pdb.set_trace()
        lr.fit(X=self[self.covariate_cols], y=self[self.outcome_col])
        self[self.resid_col] = self[self.outcome_col] - predict_with_nan(self[self.covariate_cols], lr.predict)
        return self

    def set_signal_class(self):
        self[self.signal_class_col] = \
            self.groupby(self.class_cols)[self.resid_col].transform('mean')
        return self

    def get_variance_student(self):
        """ Kane and Steinger type """
        return (self[self.resid_col] - self[self.signal_class_col]).var()

    def get_variance_classroom(self, variance_student, variance_teacher):
        """ Kane and Steinger type """
        return self[self.resid_col].var() - variance_student - variance_teacher

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
    def class_cols(self):
        return [self.class_name_col, self.time_col]

    @classmethod
    def get_teacher_dataframe(
            cls,
            data: pd.DataFrame,
            class_name_col: str,
            time_col: str,
            teacher_id_col: str,
            signal_class_col: str,
            n_class_col: str,
            *,
            student_time_col = 'student_time',
            student_class_col = 'student_class',
            **argv
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
        val1 = self.groupby(self.class_cols)[self.teacher_id_col].nunique().max()
        val2 = self.groupby([self.teacher_id_col, self.time_col])[self.class_name_col].nunique().max()
        print('1 class correspond {0} teacher'.format(val1))
        print('1 teacher correspond {0} class'.format(val2))
        return val1, val2

    def get_variance_teacher(self):
        """
        Kane and Steinger type: 1 class correspond 1 teacher
        昨年度のclass signal との分散。だけどこれ1teacher:1クラスを前提にしていて、、、
        """
        self[self.signal_class_1lag_col] = get_1lag(
            self,
            value_col=self.signal_class_col,
            id_cols=self.class_name_col,
            time_col=self.time_col
        )
        # import pdb;pdb.set_tracxce()
        return self[[self.signal_class_col, self.signal_class_1lag_col]].cov().values[0, 1]

    def set_kane(self, variance_classroom, variance_student, variance_teacher):
        self['h_jt'] = 1 / (variance_classroom + (variance_student / self[self.n_class_col]))
        self['weight_jt'] = self['h_jt'] / self.groupby(self.teacher_id_col)['h_jt'].transform('sum')

        def get_teacher_info(dfx, variance_teacher, signal_classroom_col=self.signal_class_col, weight_jt_col='weight_jt'):
            tva_no_shrinkage = (dfx[weight_jt_col] * dfx[signal_classroom_col]).sum()
            tva = tva_no_shrinkage * variance_teacher / (variance_teacher + (dfx['h_jt'].sum() ** -1))
            return (
                pd.Series({
                    'tva_no_shrinkage': tva_no_shrinkage,
                    'tva': tva
                })
            )
        df_teacher = (
            self
            .groupby(self.teacher_id_col)
            .apply(get_teacher_info, variance_teacher=variance_teacher)
        )
        return df_teacher


def kane_steinger_method():
    pass

def main():
    def get_class_info(dfx, teacher_id_col = 'teacher_id', signal_classroom_col = 'signal_classroom'):
        return (
            pd.Series({
                teacher_id_col: dfx[teacher_id_col].values[0],  # 教員はユニークのはず
                signal_classroom_col: dfx[signal_classroom_col].values[0],  # クラスはユニークのはず
                'n_classroom': dfx.shape[0],
            })
        )

    df = (
        pd.read_csv('data/seed.csv')
        .assign(
            dummy = lambda dfx: pd.np.random.normal(size=dfx.shape[0]),
            class_name = lambda dfx: dfx['teacher_id'],
        )
    )

    sdf = StudentDataFrame.get_student_dataframe(
        data=df,
        covariate_cols=['student_effect'],
        outcome_col='output',
        class_name_col= 'class_name', # 1 teacher: 1 class
        time_col='time',
        teacher_id_col = 'teacher_id',
        # student_time_col = 'time',
        # student_class_col = 'teacher_id',
    )
    sdf.set_predict_and_resid().set_signal_class()
    variance_student = sdf.get_variance_student()
    sdf.get_df_class()

    tdf = TeacherDataFrame.get_teacher_dataframe(
        sdf.get_df_class(),
        class_name_col=sdf.class_name_col,
        n_class_col=sdf.n_class_col,
        time_col=sdf.time_col,
        teacher_id_col=sdf.teacher_id_col,
        signal_class_col=sdf.signal_class_col,
    )
    tdf.data_type
    variance_teacher = tdf.get_variance_teacher()
    variance_classroom = sdf.get_variance_classroom(variance_student, variance_teacher)
    aa = tdf.set_kane(
        variance_classroom=variance_classroom,
        variance_student=variance_student,
        variance_teacher=variance_teacher
    )
    df.groupby('teacher_id')['teacher_effect'].mean()




    df_class = (
        pd.read_csv('data/seed.csv')
        .assign(
            teacher_class = lambda dfx: dfx['teacher_id']
        )
        .groupby(['time', 'teacher_class'])
        .apply(get_class_info, signal_classroom_col='output')
        .reset_index()
    )



    aa = TeacherDataFrame.get_teacher_dataframe(
        df_class,
        teacher_id_col='teacher_id',
        teacher_time_col='time',
        teacher_class_col='teacher_class',
        class_signal_col='output',
        n_teacher_class_col='n_classroom'

    )

    get_variance_student = lambda dfx: (dfx['resid'] - dfx['signal_classroom']).var()
    get_variance_teacher = lambda dfx: dfx[['signal_classroom', 'signal_classroom_1lag']].cov().values[0, 1]
    get_variance_classroom = lambda dfx, v_s, v_t: dfx['resid'].var() - v_s - v_t
    variance_student = get_variance_student(df_student)
    variance_teacher = get_variance_teacher(df_class)
    variance_classroom= get_variance_classroom(df_student, variance_student, variance_teacher)

    df_class['h_jt'] = 1 /(variance_classroom + (variance_student/df_class['n_classroom']))
    df_class['weight_jt'] = df_class['h_jt']/df_class.groupby(teacher_id)['h_jt'].transform('sum')
