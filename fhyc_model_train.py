# -*- coding: utf-8 -*—
import pandas as pd
import sys
import datetime
import fhyc_global_list as fgl
import fhyc_mutil_model as fmm


def model_train():
    run_time = sys.argv[2]
    version_ = sys.argv[3]
    model_type = sys.argv[4]

    # 线性回归
    if model_type == '1':
        iseval= sys.argv[5]
        train_dicts = {'model_name': 'LinearRegression'}
    # 随机森林
    elif model_type == '2':
        n_estimators=sys.argv[5]
        max_depth = sys.argv[6]
        min_samples_split = sys.argv[7]
        min_samples_leaf = sys.argv[8]
        iseval = sys.argv[9]
        train_dicts = {'n_estimators': int(n_estimators), 'max_depth': int(max_depth),
                       'model_name': 'RandomForestRegressor', 'min_samples_split': int(min_samples_split),
                       'min_samples_leaf': int(min_samples_leaf)}
    elif model_type == '3':
        max_depth = sys.argv[5]
        min_samples_split = sys.argv[6]
        min_samples_leaf = sys.argv[7]
        iseval = sys.argv[8]
        train_dicts = {'model_name': 'ExtraTreeRegressor',  'max_depth': int(max_depth),
                       'min_samples_split': int(min_samples_split), 'min_samples_leaf': int(min_samples_leaf)}
    # 是否评测
    if iseval == '1':
        eval_flag = True
    elif iseval == '0':
        eval_flag = False
    try:
        transinfo = pd.read_csv(fgl.TRANSINFO_FORM, encoding='utf-8')
    except Exception:
        transinfo = pd.read_csv(fgl.TRANSINFO_FORM, encoding='gbk')
    print('Start Training Model...', 'Time:', datetime.datetime.now())
    train_set = fmm.feature_engineer(transinfo,version_)
    fmm.muti_trainer(run_time, train_set, train_dicts, version_, iseval=eval_flag)
    print('train over!')


if __name__ == '__main__':
    model_train()