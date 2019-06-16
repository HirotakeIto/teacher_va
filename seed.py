import pandas as pd
import numpy as np

def get_covariance_like_time_series(size, auto_cov):
    cov_teacher_effect2 = np.eye(size)
    for distance in range(size):
        for i in range(size):
            if 0 < i + distance < size:
                cov_teacher_effect2[i, i + distance] = auto_cov ** distance
                cov_teacher_effect2[i + distance, i] = auto_cov ** distance
    return cov_teacher_effect2


n_teacher = 5
n_student = 100
n_time = 2
is_multiple_class = False
cov_across_time = 1

# teacher
mean_teacher_effect = np.kron(
    np.zeros(shape=n_teacher),
    np.ones(n_time)
)
# mean_teacher_effect[0: n_time] = 10
cov_teacher_effect = np.kron(
    np.eye(n_teacher),
    get_covariance_like_time_series(size=n_time, auto_cov=cov_across_time)
)
teacher_effect = np.random.multivariate_normal(
    mean=mean_teacher_effect,
    cov=cov_teacher_effect
)
time_teacher = np.kron(
    np.ones(shape=n_teacher),
    np.arange(n_time)
)
teacher_id = np.kron(
    np.arange(n_teacher),
    np.ones(shape=n_time)
)
df_teacher = pd.DataFrame(
    np.c_[teacher_id, time_teacher, teacher_effect],
    columns=['teacher_id', 'time', 'teacher_effect']
)
# check
aa = (
    df_teacher
    .set_index(['teacher_id', 'time'])
    ['teacher_effect']
    .unstack(-1)
    .cov()
)
print(aa)



# student
mean_student_effect = np.kron(
    np.random.normal(size=n_student),
    np.ones(n_time)
)
cov_student_effect = np.kron(
    np.eye(n_student),
    get_covariance_like_time_series(size=n_time, auto_cov=1)
)
student_effect = np.random.multivariate_normal(
    mean=mean_student_effect,
    cov=cov_student_effect
) * 1000
time_student = np.kron(
    np.ones(shape=n_student),
    np.arange(n_time)
)
student_id = np.kron(
    np.arange(n_student),
    np.ones(shape=n_time)
)
student_teacher = np.random.choice(np.arange(n_teacher), size=n_student*n_time)
df_student = pd.DataFrame(
    np.c_[student_id, time_student, student_teacher, student_effect],
    columns=['student_id', 'time', 'student_teacher', 'student_effect']
)

df = (
    df_student
    .rename(columns={'student_teacher': 'teacher_id'})
    .merge(
        df_teacher,
        on=['time', 'teacher_id'],
        how='left'
    )
    .assign(
        output = lambda dfx: dfx['student_effect'] + dfx['teacher_effect']
    )
)
df.to_csv('data/seed.csv', index=False)


