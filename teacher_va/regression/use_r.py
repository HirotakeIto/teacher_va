def convert_str_list_including_factor(str_list, factor_cols):
    return [x if x not in factor_cols else 'as.factor({0})'.format(x) for x in str_list]


def get_add_str_from_str_list(str_list, factor_cols=None):
    factor_cols = [] if factor_cols is None else factor_cols
    return ' + '.join(convert_str_list_including_factor(str_list, factor_cols))