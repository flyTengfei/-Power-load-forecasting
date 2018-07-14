import pandas as pd
import gc
from time import ctime
from sklearn.externals import joblib    # 在训练模型后将模型保存
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import datetime
import os
import numpy as np
from dateutil.parser import parse

import fhyc_global_list as fgl
# import fhyc_oracle_unit as fou
# 全局变量定义


# 获取测试集合列表
def get_test_(df, n):
    re = []
    df = df[df.isnull().any(axis=1) == False]
    for p, q in df.groupby('CLDBH'):
        q = q.sort_values(by=["date"], ascending=False)
        for i, j in enumerate(q.index):
            try:
                if q.loc[j, 'date'] == q.loc[q.index[i+1], 'date']+datetime.timedelta(days=1):
                    re.append([str(q.loc[j, 'date']), int(p)])
                    break
            except Exception:
                break
        if len(re) == n:
            break
    return re


# 删除指定列
# 输入：待删除数据集和del_df， 待删除的特征列表del_list
def columns_deleter(del_df, del_list):
    for item in del_list:
        try:
            del del_df[item]
        except Exception:
            continue
    gc.collect()
    return del_df


"""
1. 数据表融合
"""


def cedian_form_read(cd_dir):
    print('Start time:', datetime.datetime.now())
    # 0:cldbh, 1:sjsj, 2:ptbb, 3:ctbb, 11:yggl
    try:
        cd_form = pd.read_csv(cd_dir, sep='\t', header=None, usecols=[0, 1, 2, 3, 11])
    except Exception:
        cd_form = pd.read_csv(cd_dir, sep=',', header=None, usecols=[0, 1, 2, 3, 11])
    print('2016File read :', datetime.datetime.now())
    cd_form.columns = ['CLDBH', 'date', 'ptbb', 'ctbb', 'yggl']
    cd_form['date'] = pd.to_datetime(cd_form['date'].str.split(' ').str[0])

    cd_form["yggl"].fillna(-999, inplace=True)  # 为防止出现空值，先对有功功率进行空值填充
    cd_form = cd_form.ix[cd_form.groupby(["CLDBH", "date"]).apply(lambda x: x['yggl']
                                                                  .idxmax()).values, :]  # 筛选除每天的最大值样本
    print('Time used for getMax:', datetime.datetime.now())
    cd_form['yggl'].replace(-999, np.nan, inplace=True)
    cd_form['yggl'] = cd_form['yggl'] * cd_form["ptbb"] * cd_form["ctbb"]
    del cd_form["ctbb"]
    del cd_form["ptbb"]
    gc.collect()
    print('Split completed!', 'Time used:', datetime.datetime.now())
    return cd_form


def weather_combination(weather_df, weather_dir):
    # 读取天气表
    df5 = pd.read_csv(weather_dir, encoding='gbk')
    # df5 = pd.read_csv('/home/hyf/IdeaProjects/overload/Data/fhyc/merge_weather_bmbh_zdh.csv', encoding='gbk')
    df5['TIME'] = pd.to_datetime(df5['TIME'])
    del df5["ZDH"]
    del df5["unit_id"]
    # merge the two file by 'BMBH','date'
    dfs = pd.merge(weather_df, df5, how='left', left_on=['BMBH', 'date'], right_on=['bmbh', 'TIME'])
    del df5
    del dfs['TIME']
    del dfs['bmbh']
    gc.collect()
    print('WeatherInfo Combination completed!', 'Time:', datetime.datetime.now())
    return dfs


def taizhang_combination(tz_df, tz_dir):
    # 选择需要的台帐信息
    use_cols = ['CLDBH',  'BYQRL', 'BMBH', 'CKDYDJ', 'JLFS', 'CZLX', 'BYQLB',
                'BYQTYPE', 'YHS', 'ISCHK', 'TRANSTYPE', 'DELFLAG', 'QUYU', 'SBXZ', 'YC_RL']

    df2 = pd.read_csv(tz_dir, usecols=use_cols, encoding='gbk')

    # 处理空值，连续型变量用均值填充，离散型变量用默认值填充
    df2.fillna({'CKDYDJ': 50, 'JLFS': 20, 'BYQLB': -1, 'YHS': df2['YHS'].mean(), 'QUYU': -1, 'YC_RL': -1},
               inplace=True)

    # 将配变测点数据和配变台帐数据进行合并
    df2 = pd.merge(tz_df, df2, how="inner", on="CLDBH")

    print('TaiZhang Combination completed!', 'Time:', datetime.datetime.now())
    return df2


def volication_combination(volication_df, volication_dir):
    # vacation_info = pd.read_csv('/home/hyf/IdeaProjects/overload/Data/fhyc/HOLIDAY.csv')
    holiday_info = pd.read_csv(volication_dir)
    holiday_info['RDAY'] = pd.to_datetime(holiday_info['RDAY'])
    dfs = pd.merge(volication_df, holiday_info, how='left', left_on='date', right_on='RDAY')
    del dfs['RDAY']
    del holiday_info
    gc.collect()
    print('holiday_info Combination completed!', 'Time:', datetime.datetime.now())
    return dfs


def economic_combination(economic_df, economic_dir):
    economic_info = pd.read_csv(economic_dir)
    economic_df['date'] = pd.to_datetime(economic_df['date'])
    economic_df['year'] = economic_df['date'].apply(lambda x: x.year)
    dfs = pd.merge(economic_df, economic_info, how='left', on='year')
    del dfs['year']
    del economic_info
    print('economic_info Combination completed!', 'Time:', datetime.datetime.now())
    return dfs


def updata_check(uc_dir):
    for item in uc_dir:
        if not os.path.exists(item):
            return False
    return True


def data_combination(cedian_objs, taizhang_dir, weather_dir, holiday_dir, jinji_dir):
    """
    # 读取变压器测点数据，来自Hive-刘建模
    """
    dfs = []

    for cd_index in range(len(cedian_objs)):
        dfs.append(cedian_form_read(cedian_objs[cd_index]))
    dfs = pd.concat(dfs, axis=0)

    print('Cedian completed!', 'Time used:', datetime.datetime.now())
    """
    # 合并变压器台帐信息，来自Hive-刘建模
    """
    dfs = taizhang_combination(dfs.copy(), taizhang_dir)

    """
    合并天气属性
    """
    dfs = weather_combination(dfs.copy(), weather_dir)

    """
    # 合并节假日信息
    """
    dfs = volication_combination(dfs.copy(), holiday_dir)

    """
    # 合并经济增长信息
    """
    dfs = economic_combination(dfs.copy(), jinji_dir)
    gc.collect()
    dfs.dropna(inplace=True)
    return dfs


# 数据更新
def updata_transinfo(transinfo_dir, updata_dir, taizhang_dir, weather_dir, holiday_dir, jinji_dir):
    try:
        df = pd.read_csv(transinfo_dir, encoding='utf-8')
    except Exception:
        df = pd.read_csv(transinfo_dir, encoding='gbk')

    if updata_check(updata_dir):
        updata_path = updata_dir
        df1 = data_combination(updata_path, taizhang_dir, weather_dir, holiday_dir, jinji_dir)

        df1['date'] = pd.to_datetime(df1['date'])
        df['date'] = pd.to_datetime(df['date'])

        df['CLDBH'] = df['CLDBH'].astype(int)
        df1['CLDBH'] = df1['CLDBH'].astype(int)

        trans_info_updata = pd.concat([df, df1], axis=0)
        trans_info_updata.drop_duplicates(keep='last', inplace=True, subset=['CLDBH', 'date'])
        trans_info_updata.to_csv(transinfo_dir, index=False)
        return trans_info_updata.shape[0], pd.DataFrame(trans_info_updata['date']).max()['date'], trans_info_updata
    else:
        print('updata path is not exist,please input the true path!')


"""
2. 特征工程
"""


def history_statistics(x, dict_, q, type):

    if np.isnan(q.loc[x, type]):
        return pd.np.nan

    else:
        hs_temp = int(q.loc[x, type])
        if dict_[hs_temp]==None:
            #            print("is none ,now add")
            if not np.isnan(q.loc[x, "yggl"]):
                dict_[hs_temp] = [q.loc[x, "yggl"], 1]
            return pd.np.nan
        else:
            temp = dict_[hs_temp][0]
            if not np.isnan(q.loc[x, "yggl"]):
                dict_[hs_temp] = [(dict_[hs_temp][0]*dict_[hs_temp][1] +
                                          q.loc[x, "yggl"])/(dict_[hs_temp][1]+1), dict_[hs_temp][1]+1]
            return temp


def history_statistics_test(hst_df, hst_type, hst_column,version_):
    temp_df = hst_df.groupby(['CLDBH', hst_type])['yggl'].mean().reset_index()
    temp_df = temp_df.rename(columns={'yggl': hst_column})
    temp_df.to_pickle(fgl.MODEL_DIR+'/Feature_'+hst_column+str(version_)+'.pkl')


# 处理天气特征
def transform_weather(x):
    if str(x).find("雪") != -1:
        return 0
    elif str(x).find("雷阵雨") != -1:
        return 1
    elif str(x).find("阵雨") != -1:
        return 2
    elif str(x).find("多云") != -1:
        return 3
    elif str(x).find("晴") != -1:
        return 4
    else:
        return -1


# 处理区域特征
def transform_quyu(x):
    if str(x) == "城区":
        return 1
    elif str(x) == "农村":
        return 0
    elif str(x) == "工业园":
        return 2
    else:
        return-1


# 变量名，类型，作用
# df, DataFrame, 需要做特征工程的数据集合
def feature_engineer(df,version_):
    print('Strat ', 'Time:', datetime.datetime.now())
    df = columns_deleter(df, ['BMBH'])  # 此时df.shape[0] = 2966343
    df["TQ_A"] = df["TQ_A"].apply(transform_weather)
    df["QUYU"] = df["QUYU"].apply(transform_quyu)

    df["yggl"] = abs(df["yggl"])/df['BYQRL']
    df["date"] = pd.to_datetime(df["date"])
    # 删除异常数据,此时df.shape[0] = 2641586，共过滤324757条样本
    df.drop(df[(df['yggl'] > 1.5) | (df['yggl'] < 0.15)].index, inplace=True)
    print('Start feature engineer:')

    """
    测点数据特征构造
    """
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['week'] = df['date'].apply(lambda x: x.weekday())
    df['gw_dispersed'] = df['GW_A'].apply(lambda x: 0 if x < 10 else (1 if x < 20 else (2 if x < 30 else 3)))
    df['dw_dispersed'] = df['DW_A'].apply(lambda x: 0 if x < 5 else (1 if x < 15 else (2 if x < 25 else 3)))
    df = columns_deleter(df, ['GW_A', 'DW_A'])
    # df['UNWORKDAY'] = df['UNWORKDAY'].astype(int)
    # df['WEEKEND'] = df['WEEKEND'].astype(int)
    # df['HOLIDAY'] = df['HOLIDAY'].astype(int)

    # 为测试集构造不同月份的有功功率水平指标
    history_statistics_test(df, 'week', 'meanw',version_)
    history_statistics_test(df, 'month', 'meanm',version_)
    history_statistics_test(df, 'gw_dispersed', 'meangw',version_)
    history_statistics_test(df, 'dw_dispersed', 'meandw',version_)
    history_statistics_test(df, 'TQ_A', 'meantq',version_)
    history_statistics_test(df, 'UNWORKDAY', 'mean_UNWORKDAY',version_)
    history_statistics_test(df, 'WEEKEND', 'mean_WEEKEND',version_)
    history_statistics_test(df, 'HOLIDAY', 'mean_HOLIDAY',version_)

    # 构造前n天有功功率特征
    templist = []
    temp_groupby = df.groupby('CLDBH')
    del df
    gc.collect()

    count = 0
    for p, q in temp_groupby:
        weekdict = {}
        for i in range(7):
            weekdict[i] = None
        monthdict = {}
        for i in range(1, 13):
            monthdict[i] = None
        gwdict = {}
        for i in range(4):
            gwdict[i] = None
        tqdict = {}
        for i in range(-1, 5, 1):
            tqdict[i] = None
        isworkdict = {}
        for i in range(2):
            isworkdict[i] = None

        q = q.sort_values("date")
        times = pd.date_range(q["date"].iloc[0], q["date"].iloc[-1], freq='1D')
        q = pd.merge(left=pd.DataFrame(times, columns=["date"]), right=q, how="left", on="date")

        q.reset_index(inplace=True, drop=True)
        q.reset_index(inplace=True)
        # 构造前n天有功功率特征
        q["meanw"] = q["index"].apply(history_statistics, args=(weekdict, q, 'week'))
        q["meanm"] = q["index"].apply(history_statistics, args=(monthdict, q, 'month'))
        q["meangw"] = q["index"].apply(history_statistics, args=(gwdict, q, 'gw_dispersed'))
        q["meandw"] = q["index"].apply(history_statistics, args=(gwdict, q, 'dw_dispersed'))
        q["meantq"] = q["index"].apply(history_statistics, args=(tqdict, q, 'TQ_A'))
        q["mean_UNWORKDAY"] = q["index"].apply(history_statistics, args=(isworkdict, q, 'UNWORKDAY'))
        q["mean_WEEKEND"] = q["index"].apply(history_statistics, args=(isworkdict, q, 'WEEKEND'))
        q["mean_HOLIDAY"] = q["index"].apply(history_statistics, args=(isworkdict, q, 'HOLIDAY'))

        for delay in range(1, 33):
            q["yggl_feature_"+str(delay)] = q["yggl"].shift(delay)
        templist.append(q)
        count += 1
        if count % 100 == 0:
            print("处理了{}个".format(count))
            print(ctime())
    del temp_groupby
    gc.collect()
    df = pd.concat(templist, axis=0)
    del templist
    columns_deleter(df, ['index'])
    gc.collect()
    df.dropna(subset=['CLDBH'], how='any', inplace=True)
    # 此时df.shape[0] = 3416171
    df['CLDBH'] = df['CLDBH'].astype(int)
    return df


"""
3. 训练测试集划分
"""


def data_split(df, split_data='2017-08-01'):
    train_temp = df[df['date'] < split_data]
    test_temp = df[df['date'] >= split_data]
    return train_temp, test_temp


def get_evaluate_train(get_df, n):
    get_df['date'] = pd.to_datetime(get_df['date'])
    train_set = feature_engineer(get_df)
    # 实际预测时获取预测目标昨天/今天的yggl，根据时间扩充31天记录，并根据CLDBH和date不冲
    test_list = get_test_(train_set.copy(), n)

    test_set = pd.DataFrame([], columns=['CLDBH', 'date'])
    for item in test_list:
        temp = pd.DataFrame([])
        # temp['CLDBH'] = item[1]
        temp['date'] = pd.date_range(end=item[0], periods=31, freq='1D')
        temp['CLDBH'] = item[1]
        test_set = pd.concat([test_set, temp], axis=0)

    test_set = pd.merge(test_set, get_df, on=['CLDBH', 'date'], how='left')

    temp = pd.DataFrame([], columns=train_set.columns.values)

    temp['CLDBH'] = test_set['CLDBH']
    temp['date'] = test_set['date']
    train_set = pd.concat([train_set, temp], axis=0)
    train_set.drop_duplicates(['CLDBH', 'date'], keep=False, inplace=True)

    return train_set, test_set


"""
4. 模型训练
"""


# 变量名，类型，作用
# train_set, DataFrame, 用作模型训练的数据集合
# params_dict, dict, 字典中存储模型训练用到的算法名称和应用该模型时的参数
# version_, int/str, 用于标注具体哪个版本的模型
def muti_trainer(r_date, train_set, params_dicts, version_, iseval=True):
    feature_list = []
    print('data is ok!', 'Time:', datetime.datetime.now())
    for i in range(1, 31):
        feature = list(train_set.loc[:, :"yggl_feature_1"].columns.values)
        feature.remove("yggl_feature_1")
        # feature = [x for x in train_set.columns.values if x.find("yggl") == -1]
        feature.append("yggl_feature_"+str(i))
        feature.append("yggl_feature_"+str(i+1))
        feature.append("yggl_feature_"+str(i+2))
        # feature.append('yggl')
        feature_list.append(feature)
    # print(feature_list)
    for delay in range(30):
        print('training:', delay)
        if iseval:
            temp = train_set.loc[:, feature_list[delay]]
            temp.dropna(inplace=True)

            x_train, x_test = data_split(temp, split_data='2017-06-01')
            y_train = x_train['yggl']
            del x_train['yggl']
            y_test = x_test['yggl']
            del x_test['yggl']

            del_list = ['CLDBH', 'date']
            x_train = columns_deleter(x_train, del_list)
            m_name, m_dir, m_usecols = model_train(x_train, y_train, delay, params_dicts, version_)
            x_test = columns_deleter(x_test, del_list)
            res = model_predict(x_test, m_dir, r_date)
            score, res = get_score(pd.DataFrame(y_test), res)
            m_train_date = str(datetime.datetime.now()).split('.')[0]
            print(score, 'Time:', datetime.datetime.now())
            # 模型记录文件，模型名，模型路径，
            if os.path.exists(fgl.RESULT_DIR+'/ModelRecord.csv'):
                pd.DataFrame([[m_name, m_dir, m_usecols, score, m_train_date]],
                             columns=['ModelName', 'ModelPath', 'TrainFeature', 'Reliability', 'ModelTrainDate']).\
                    to_csv(fgl.RESULT_DIR+'/ModelRecord.csv', index=False, encoding='utf-8', mode='a', header=None)
            else:
                pd.DataFrame([[m_name, m_dir, m_usecols, score, m_train_date]],
                             columns=['ModelName', 'ModelPath', 'TrainFeature', 'Reliability', 'ModelTrainDate']). \
                    to_csv(fgl.RESULT_DIR+'/ModelRecord.csv', index=False, encoding='utf-8')

        else:
            temp = train_set.loc[:, feature_list[delay]]
            temp.dropna(inplace=True)
            test_temp = temp['yggl']
            del temp['yggl']
            gc.collect()
            del_list = ['CLDBH', 'date']
            temp = columns_deleter(temp, del_list)
            m_name, m_dir, m_usecols = model_train(temp, test_temp, delay, params_dicts, version_)
            # 模型记录文件，模型名，模型路径，
            if os.path.exists(fgl.RESULT_DIR+'/ModelRecord.csv'):
                pd.DataFrame([[m_name, m_dir, m_usecols, np.NAN, m_train_date]],
                             columns=['ModelName', 'ModelPath', 'TrainFeature', 'Reliability', 'ModelTrainDate']). \
                    to_csv(fgl.RESULT_DIR+'/ModelRecord.csv', index=False, encoding='utf-8', mode='a', header=None)
            else:
                pd.DataFrame([[m_name, m_dir, m_usecols, np.NAN, m_train_date]],
                             columns=['ModelName', 'ModelPath', 'TrainFeature', 'Reliability', 'ModelTrainDate']). \
                    to_csv(fgl.RESULT_DIR+'/ModelRecord.csv', index=False, encoding='utf-8')


# 变量名，类型，作用
# x, DataFrame, 用作模型训练的特征集合
# y, DataFrame, 用作模型训练的目标集合
# delay, int, 用作标注第几个模型
# params_dict, dict, 字典中存储模型训练用到的算法名称和应用该模型时的参数
# version_, int/str, 用于标注具体哪个版本的模型
def model_train(x, y, delay, params_dict, version_):
    if params_dict['model_name'] == 'LinearRegression':
        model = LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)    # 1. 线性回归
    elif params_dict['model_name'] == 'RandomForestRegressor':
        model = RandomForestRegressor(n_estimators=params_dict['n_estimators'], n_jobs=7, random_state=2017,
                                      max_depth=params_dict['max_depth'], oob_score=True,
                                      min_samples_split=params_dict['min_samples_split'],
                                      min_samples_leaf=params_dict['min_samples_leaf'])
    elif params_dict['model_name'] == 'ExtraTreeRegressor':
        model = ExtraTreeRegressor(max_depth=params_dict['max_depth'], random_state=2017,
                                   min_samples_split=params_dict['min_samples_split'],
                                   min_samples_leaf=params_dict['min_samples_leaf'])     # 9. ExtraTree极端随机树回归
    else:
        model = RandomForestRegressor()
    model_n = 'Model_'+str(delay+1)+'_'+params_dict['model_name']+'_V'+str(version_)+'.model'
    model_dir = fgl.MODEL_DIR+'/'+model_n
    print('Start Training Model '+str(delay+1)+'...', 'Time:', datetime.datetime.now())
    print(list(x.columns))
    best_model = model.fit(x, y)
    # print(best_model.feature_importances_)
    print('Model '+str(delay+1)+'is ok!', 'Time:', datetime.datetime.now())
    joblib.dump(best_model, model_dir)
    print('Model '+str(delay+1)+'saved!', 'Time:', datetime.datetime.now())
    return model_n, best_model, '-'.join(list(x.columns.values))


"""
5. 模型预测
"""


# 变量名，类型，作用
# input_, DataFrame, 用作模型预测的变压器集合决策表
# byqrl, DataFrame, 用作将预测结果还原的变压器容量信息
# delays, int, input_的日期与预测起点时间的偏差天数
# model_name, str, 预测时用到的算法模型名称
# version_, int/str, 用于标注具体哪个版本的模型
def thirty_day_predicter(input_, byqrl, delays, model_name, version_, r_date, job_parameter):
    # 构建时间特征
    input_['date'] = pd.to_datetime(input_['date'])
    input_['year'] = input_['date'].apply(lambda x_: x_.year)
    input_['month'] = input_['date'].apply(lambda x_: x_.month)
    input_['day'] = input_['date'].apply(lambda x_: x_.day)
    input_['week'] = input_['date'].apply(lambda x_: x_.weekday())
    input_['gw_dispersed'] = input_['GW_A'].apply(lambda x: 0 if x < 10 else (1 if x < 20 else (2 if x < 30 else 3)))
    input_['dw_dispersed'] = input_['DW_A'].apply(lambda x: 0 if x < 5 else (1 if x < 15 else (2 if x < 25 else 3)))
    input_ = columns_deleter(input_, ['GW_A', 'DW_A'])

    # 处理区域天气及yggl特征
    input_ = columns_deleter(input_, ['BMBH'])
    input_["TQ_A"] = input_["TQ_A"].apply(transform_weather)
    input_["QUYU"] = input_["QUYU"].apply(transform_quyu)

    input_["yggl"] = abs(input_["yggl"])/input_['BYQRL']

    # 检查文件是否存在
    check_list = {'week': fgl.MODEL_DIR+'/Feature_meanw'+str(version_)+'.pkl',
                  'month': fgl.MODEL_DIR+'/Feature_meanm'+str(version_)+'.pkl',
                  'TQ_A': fgl.MODEL_DIR+'/Feature_meantq'+str(version_)+'.pkl',
                  'gw_dispersed': fgl.MODEL_DIR+'/Feature_meangw'+str(version_)+'.pkl',
                  'dw_dispersed': fgl.MODEL_DIR+'/Feature_meandw'+str(version_)+'.pkl',
                  'WEEKEND': fgl.MODEL_DIR+'/Feature_mean_WEEKEND'+str(version_)+'.pkl',
                  'UNWORKDAY': fgl.MODEL_DIR+'/Feature_mean_UNWORKDAY'+str(version_)+'.pkl',
                  'HOLIDAY':fgl.MODEL_DIR+'/Feature_mean_HOLIDAY'+str(version_)+'.pkl'
                  }
    for (check_feature, check_dir) in check_list.items():
        if os.path.exists(check_dir):
            temp_df = pd.read_pickle(check_dir)
            input_ = pd.merge(input_, temp_df, on=['CLDBH', check_feature], how='left')
        else:
            print_str = "特征文件缺失。请检查"+check_dir+"是否存在！"
            print(print_str, 'Time:', datetime.datetime.now())
            # fou.update_data(pd.DataFrame([[r_date, 1, print_str]], columns=['rundate', 'state', 'error']))
            state_ = pd.DataFrame([[r_date, 1, print_str, job_parameter, 'predict.sh',
                                    str(datetime.datetime.now()).split('.')[0]]])
            state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
            sys.exit(1)

    start_date = True

    # 测试数据规整
    temp_list = []
    for p, q in input_.groupby('CLDBH'):
        q = q.sort_values(by=["date"], ascending=True)
        q.reset_index(drop=True, inplace=True)
        q['yggl_feature'] = q.loc[0, 'yggl']
        q['yggl_feature_x'] = q.loc[0, 'yggl_x']
        q['yggl_feature_y'] = q.loc[0, 'yggl_y']
        if start_date:
            start_date = q.date[0]
        temp_list.append(q)
    input_ = pd.concat(temp_list, axis=0)
    print('数据特征规整完成！', 'Time:', datetime.datetime.now())
    res_ = []
    # 预测
    for i in range(delays):
        i = i + (30-delays)
        print('开始预测第'+str(i+1-(30-delays))+'天！', 'Time:', datetime.datetime.now())
        delta = datetime.timedelta(days=i+1)

        x = input_[input_['date'] == start_date+delta]
        z = x.loc[:, ['CLDBH', 'date']].reset_index(drop=True)

        del_list = ['CLDBH', 'date', 'yggl', 'yggl_x', 'yggl_y']
        x = columns_deleter(x, del_list)

        x['meanw'].fillna(method='pad', inplace=True)
        x['meanm'].fillna(method='pad', inplace=True)
        x['meantq'].fillna(method='pad', inplace=True)
        x['meandw'].fillna(method='pad', inplace=True)
        x['yggl_feature_x'].fillna(method='pad', inplace=True)
        x['yggl_feature_y'].fillna(method='pad', inplace=True)
        # x['gw_dispersed'].fillna(method='pad', inplace=True)

        # del_list = ['BYQRL', 'CKDYDJ', 'JLFS', 'CZLX', 'BYQLB', 'BYQTYPE', 'YHS', 'ISCHK', 'TRANSTYPE',
        #             'QUYU', 'SBXZ', 'YC_RL', 'UNWORKDAY', 'WEEKEND', 'HOLIDAY',
        #             'economic_growth', 'year', 'month', 'day', 'week', 'gw_dispersed',
        #             'meanw', 'meangw', 'mean_UNWORKDAY', 'mean_WEEKEND', 'mean_HOLIDAY', 'meanm']
        # x = columns_deleter(x, del_list)

        model_dir = fgl.MODEL_DIR+'/'+'Model_'+str(i+1)+'_'+model_name+'_V'+str(version_)+'.model'
        print('预测开始，Time：', datetime.datetime.now())
        res = model_predict(x, model_dir, r_date)
        res = res.reset_index(drop=True)

        print('预测完成，Time：', datetime.datetime.now())
        res = pd.concat([z, res], axis=1)
        res['suanfaname'] = 'Model_'+str(i+1)+'_'+model_name+'_V'+str(version_)+'.model'
        res['rundate'] = r_date
        print('还原预测结果')
        res = pd.merge(res, byqrl, on='CLDBH', how='left')
        res['predict_yggl'] = res['predict_yggl'] * res['BYQRL']
        del res['BYQRL']
        res_.append(res)
    return pd.concat(res_, axis=0)


def model_predict(x, model_dir, r_date):
    try:
        model_pred = joblib.load(model_dir)
    except Exception:
        model_pred = model_dir
    try:
        res = pd.DataFrame(model_pred.predict(x), columns=['predict_yggl'])
    except Exception as err:
        print(err, 'Time:', datetime.datetime.now())
        res = pd.DataFrame({'predict_yggl': [np.nan]*x.shape[0]})
    return res


"""
6. 模型评估
"""


def get_score(true_, pre_):
    true_.reset_index(drop=True, inplace=True)
    pre_.reset_index(drop=True, inplace=True)
    res_ = pd.concat([pre_, true_], axis=1)
    res_['l1'] = res_['predict_yggl'] - res_['yggl']
    res_['error_rate'] = res_['l1']/res_['yggl']
    res_['label'] = res_['error_rate'].apply(lambda x: 1 if (x <= 0.20) & (x > -0.15) else 0)
    score = res_['label'].sum()/res_.shape[0]
    del_list = ['l1', 'label']
    res_ = columns_deleter(res_, del_list)
    return score, res_


def predict_oneday_onecldbh(CLDBH,byqrl,data_date,pred_date,model_name,version_,r_date, job_parameter):
    pred_date = parse(pred_date)
    data_date = parse(data_date)
    time_delta = pred_date-data_date
    days = time_delta.days
    if days > 30:
        print_str = '预测目标过远。应不超过基准时间30天！'
        print(print_str, 'Time:', datetime.datetime.now())
        # fou.update_data(pd.DataFrame([[r_date, 1, print_str]], columns=['rundate', 'state', 'error']))
        state_ = pd.DataFrame([[r_date, 1, print_str, job_parameter, 'predict.sh',
                                str(datetime.datetime.now()).split('.')[0]]])
        state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
        sys.exit(1)
    delta = datetime.timedelta(days=0)
    input_dir1 = fgl.INPUT_DIR+'/PredictInput'+''.join(str(data_date-delta).split(' ')[0].split('-'))+'.csv'
    input_dir2 = fgl.INPUT_DIR+'/PredictInput'+''.join(str(data_date-2*delta).split(' ')[0].split('-'))+'.csv'
    input_dir3 = fgl.INPUT_DIR+'/PredictInput'+''.join(str(data_date-3*delta).split(' ')[0].split('-'))+'.csv'

    if os.path.exists(input_dir1) and os.path.exists(input_dir2) and os.path.exists(input_dir3):

        # 通过CLDBH字段拼接target_cldbh和input_
        """
        注：有功功率输入文件input_dir*存于“**/fhyc/Data/Input/”中，若文件不存在则返回警告文件不存在
        """
        input_usecols = ['CLDBH', 'date', 'yggl']
        input_1 = pd.read_csv(input_dir1, encoding='utf-8', usecols=input_usecols)   # 读取输入文件信息
        input_2 = pd.read_csv(input_dir2, encoding='utf-8', usecols=input_usecols)
        input_2.rename(columns={"yggl":"yggl_x"}, inplace=True)
        del input_2["date"]
        input_3 = pd.read_csv(input_dir3, encoding='utf-8', usecols=input_usecols)
        input_3.rename(columns={"yggl": "yggl_y"}, inplace=True)
        del input_3["date"]
        input_ = pd.merge(input_1, input_2, how="inner", on='CLDBH')
        input_ = pd.merge(input_, input_3, how="inner", on='CLDBH')

        input_ = input_[input_["CLDBH"]==CLDBH]

        del input_["date"]
        input_["date"] = pred_date

        # 合并变压器台帐信息，来自Hive-刘建模
        if os.path.exists(fgl.TAIZHANG_FORM):
            input_ = taizhang_combination(input_.copy(), fgl.TAIZHANG_FORM)
        else:
            print_str = 'Error code 102：缺少输入文件。无法完整预测。请将设备台账文件上传至路径:'+fgl.TAIZHANG_FORM+'！'
            print(print_str, 'Time:', datetime.datetime.now())
            # fou.update_data(pd.DataFrame([[r_date, 1, print_str]], columns=['rundate', 'state', 'error']))
            state_ = pd.DataFrame([[r_date, 1, print_str, job_parameter, 'predict.sh',
                                    str(datetime.datetime.now()).split('.')[0]]])
            state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
            sys.exit(1)

        # 合并天气属性
        w_input = fgl.INPUT_DIR+'/WeatherForecast'+''.join(str(data_date-delta).split(' ')[0].split('-'))+'.csv'
        if os.path.exists(fgl.WEATHER_FORM):
            input_ = weather_combination(input_.copy(), fgl.WEATHER_FORM)
        elif os.path.exists(w_input):
            input_ = weather_combination(input_.copy(), w_input)
        else:
            print_str = 'Error code 103：缺少输入文件。无法完整预测。请将天气文件上传至路径:'+w_input+'！'
            print(print_str, 'Time:', datetime.datetime.now())
            # fou.update_data(pd.DataFrame([[r_date, 1, print_str]], columns=['rundate', 'state', 'error']))
            state_ = pd.DataFrame([[r_date, 1, print_str, job_parameter, 'predict.sh',
                                    str(datetime.datetime.now()).split('.')[0]]])
            state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
            sys.exit(1)

        # 合并节假日信息
        if os.path.exists(fgl.HOLIDAY_FORM):
            input_ = volication_combination(input_.copy(), fgl.HOLIDAY_FORM)
        else:
            print_str = 'Error code 104：缺少输入文件。无法完整预测。请将设备台账文件上传至路径:'+fgl.HOLIDAY_FORM+'！'
            print(print_str, 'Time:', datetime.datetime.now())
            # fou.update_data(pd.DataFrame([[r_date, 1, print_str]], columns=['rundate', 'state', 'error']))
            state_ = pd.DataFrame([[r_date, 1, print_str, job_parameter, 'predict.sh',
                                    str(datetime.datetime.now()).split('.')[0]]])
            state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
            sys.exit(1)

        # 合并经济增长信息
        if os.path.exists(fgl.ECONOMIC_FORM):
            input_ = economic_combination(input_.copy(), fgl.ECONOMIC_FORM)
        else:
            print_str = 'Error code 105：缺少输入文件。无法完整预测。请将设备台账文件上传至路径:'+fgl.ECONOMIC_FORM+'！'
            print(print_str, 'Time:', datetime.datetime.now())
            # fou.update_data(pd.DataFrame([[r_date, 1, print_str]], columns=['rundate', 'state', 'error']))
            state_ = pd.DataFrame([[r_date, 1, print_str, job_parameter, 'predict.sh',
                                    str(datetime.datetime.now()).split('.')[0]]])
            state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
            sys.exit(1)

        input_['date'] = pd.to_datetime(input_['date'])
        input_['year'] = input_['date'].apply(lambda x_: x_.year)
        input_['month'] = input_['date'].apply(lambda x_: x_.month)
        input_['day'] = input_['date'].apply(lambda x_: x_.day)
        input_['week'] = input_['date'].apply(lambda x_: x_.weekday())
        input_['gw_dispersed'] = input_['GW_A'].apply(lambda x: 0 if x < 10 else (1 if x < 20 else (2 if x < 30 else 3)))
        input_['dw_dispersed'] = input_['DW_A'].apply(lambda x: 0 if x < 5 else (1 if x < 15 else (2 if x < 25 else 3)))
        input_ = columns_deleter(input_, ['GW_A', 'DW_A'])

        # 处理区域天气及yggl特征
        input_ = columns_deleter(input_, ['BMBH'])
        input_["TQ_A"] = input_["TQ_A"].apply(transform_weather)
        input_["QUYU"] = input_["QUYU"].apply(transform_quyu)

        input_["yggl"] = abs(input_["yggl"])/input_['BYQRL']

        # 检查文件是否存在
        check_list = {'week': fgl.MODEL_DIR+'/Feature_meanw'+str(version_)+'.pkl',
                      'month': fgl.MODEL_DIR+'/Feature_meanm'+str(version_)+'.pkl',
                      'TQ_A': fgl.MODEL_DIR+'/Feature_meantq'+str(version_)+'.pkl',
                      'gw_dispersed': fgl.MODEL_DIR+'/Feature_meangw'+str(version_)+'.pkl',
                      'dw_dispersed': fgl.MODEL_DIR+'/Feature_meandw'+str(version_)+'.pkl',
                      'WEEKEND': fgl.MODEL_DIR+'/Feature_mean_WEEKEND'+str(version_)+'.pkl',
                      'UNWORKDAY': fgl.MODEL_DIR+'/Feature_mean_UNWORKDAY'+str(version_)+'.pkl',
                      'HOLIDAY':fgl.MODEL_DIR+'/Feature_mean_HOLIDAY'+str(version_)+'.pkl'
                      }
        for (check_feature, check_dir) in check_list.items():
            if os.path.exists(check_dir):
                temp_df = pd.read_pickle(check_dir)
                input_ = pd.merge(input_, temp_df, on=['CLDBH', check_feature], how='left')
            else:
                print_str = "特征文件缺失。请检查"+check_dir+"是否存在！"
                print(print_str, 'Time:', datetime.datetime.now())
                # fou.update_data(pd.DataFrame([[r_date, 1, print_str]], columns=['rundate', 'state', 'error']))
                state_ = pd.DataFrame([[r_date, 1, print_str, job_parameter, 'predict.sh',
                                        str(datetime.datetime.now()).split('.')[0]]])
                state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
                sys.exit(1)

        z = input_.loc[:, ["CLDBH", "date"]]
        # del_list = ["CLDBH", "date", 'BYQRL', 'CKDYDJ', 'JLFS', 'CZLX', 'BYQLB', 'BYQTYPE', 'YHS', 'ISCHK', 'TRANSTYPE',
        #             'QUYU', 'SBXZ', 'YC_RL', 'UNWORKDAY', 'WEEKEND', 'HOLIDAY',
        #             'economic_growth', 'year', 'month', 'day', 'week', 'gw_dispersed',
        #             'meanw', 'meangw', 'mean_UNWORKDAY', 'mean_WEEKEND', 'mean_HOLIDAY', 'meanm']
        del_list = ["CLDBH", "date"]
        input_ = columns_deleter(input_, del_list)

        if input_.isnull().any().any():
            print_str = 'Error ：该变压器数据存在异常空值。无法预测！'
            print(print_str, 'Time:', datetime.datetime.now())
            # fou.update_data(pd.DataFrame([[r_date, 1, print_str]], columns=['rundate', 'state', 'error']))
            state_ = pd.DataFrame([[r_date, 1, print_str, job_parameter, 'predict.sh',
                                    str(datetime.datetime.now()).split('.')[0]]])
            state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
            sys.exit(1)

        model_dir = fgl.MODEL_DIR+'/'+'Model_'+str(days)+'_'+model_name+'_V'+str(version_)+'.model'
        print('预测开始，Time：', datetime.datetime.now())
        res = model_predict(input_, model_dir, r_date)
        print('预测结束，Time：', datetime.datetime.now())
        res = res.reset_index(drop=True)

        res = pd.concat([z, res], axis=1)
        res['suanfaname'] = 'Model_'+str(days)+'_'+model_name+'_V'+str(version_)+'.model'
        res['rundate'] = r_date
        print('还原预测结果，Time：', datetime.datetime.now())
        res = pd.merge(res, byqrl, on='CLDBH', how='left')
        res['predict_yggl'] = res['predict_yggl'] * res['BYQRL']
        del res['BYQRL']
        return res
    else:
        print_str = 'Warning：缺少输入文件。无法完整预测。请将指定日期'+str(data_date-delta).split(' ')[0]+'的有功功率文件上传至路径:'+fgl.INPUT_DIR+'！'
        print(print_str, 'Time:', datetime.datetime.now())
        # fou.update_data(pd.DataFrame([[r_date, 1, print_str]], columns=['rundate', 'state', 'error']))
        state_ = pd.DataFrame([[r_date, 1, print_str, job_parameter, 'predict.sh',
                                str(datetime.datetime.now()).split('.')[0]]])
        state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
        sys.exit(1)
