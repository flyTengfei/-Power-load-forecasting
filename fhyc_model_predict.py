# -*- coding: utf-8 -*—
import pandas as pd
import sys
import gc
import os
import datetime

import fhyc_global_list as fgl
import fhyc_mutil_model as fmm
# import fhyc_oracle_unit as fou

if __name__ == '__main__':

    """
    接收shell脚本传入参数：预测基准时间start_data, 模型版本号version_， 预测模式predict_way
    """
    # 获取预测执行时间
    # run_date = str(datetime.datetime.now()).split('.')[0]     # 记录预测作业开始时间，作为该次作业的唯一标注
    run_date = str(sys.argv[2]).split('.')[0].replace(',', ' ')     # 记录预测作业开始时间，作为该次作业的唯一标注

    # start_data = '2017-07-31'  # 获取模型预测基准日期，格式：年-月-日
    start_data = str(sys.argv[3])  # 获取模型预测基准日期，格式：年-月-日

    """
    River: 此处需判断该模型版本是否存在
    """
    version_ = 2017122201  # 获取模型预测输入的第一个参数：模型版本号,例如"月+日"
    # version_ = str(sys.argv[4])  # 获取模型预测输入的第一个参数：模型版本号

    """
    River：模型选择功能拓展
    """
    model_name = 'RandomForestRegressor'    # 预测使用模型名

    predict_way = int(sys.argv[5])
    # predict_way = 205396

    target_date = str(sys.argv[6])
    # target_date = '2017-08-08'

    job_parameter = '_'.join([start_data, str(version_), str(predict_way), str(target_date)])

    print('Start Predicting...', 'Time:', datetime.datetime.now())
    state_ = pd.DataFrame([[run_date, 2, 'Abnormal interruption!',
                            job_parameter, 'predict.sh', str(datetime.datetime.now()).split('.')[0]]])
    state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
    # fou.insert_data(pd.DataFrame([[run_date, '_'.join([start_data, str(version_), str(predict_way), str(target_date)])]]
    #                              , columns=['rundate', 'parameter']), fgl.FHYC_FORECAST_JOB_STATE)

    # 判断时间格式是否正确
    d1 = None
    try:
        d1 = datetime.datetime.strptime(start_data, '%Y-%m-%d')  # str(d1) = '2017-01-01 00:00:00'
    except Exception as err:
        print(err, 'Time:', datetime.datetime.now())
        # fou.update_data(pd.DataFrame([[run_date, 1, err]], columns=['rundate', 'state', 'error']))
        state_ = pd.DataFrame([[run_date, 1, err, job_parameter, 'predict.sh',
                                str(datetime.datetime.now()).split('.')[0]]])
        state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
        sys.exit(1)

    byqrl = None
    try:
        byqrl = pd.read_csv(fgl.TAIZHANG_FORM, usecols=['CLDBH', 'BYQRL'], encoding='gbk')        # 用于预测结果还原的变压器容量信息
    except Exception as err:
        print(err, 'Time:', datetime.datetime.now())
        # fou.update_data(pd.DataFrame([[run_date, 1, err]], columns=['rundate', 'state', 'error']))
        state_ = pd.DataFrame([[run_date, 1, err, job_parameter, 'predict.sh',
                                str(datetime.datetime.now()).split('.')[0]]])
        state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
        sys.exit(1)

    byqbh = None
    try:
        byqbh = pd.read_csv(fgl.TAIZHANG_FORM, usecols=['CLDBH', 'BYQBH'], encoding='gbk')        # 用于预测结果还原的变压器容量信息
    except Exception as err:
        print(err, 'Time:', datetime.datetime.now())
        # fou.update_data(pd.DataFrame([[run_date, 1, err]], columns=['rundate', 'state', 'error']))
        state_ = pd.DataFrame([[run_date, 1, err, job_parameter, 'predict.sh',
                                str(datetime.datetime.now()).split('.')[0]]])
        state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
        sys.exit(1)

    target_cldbh = pd.DataFrame(byqrl['CLDBH'])     # 待预测测量点编号

    # 输出文件路径定义
    output_dir = fgl.OUTPUT_DIR+'/PredictOutput'+''.join(str(d1).split(' ')[0].split('-'))+'_'+str(predict_way)+'_'+str(target_date)+'.csv'

    if predict_way != -1:
        if predict_way in byqbh['CLDBH'].values:
            pass
        elif predict_way in byqbh['BYQBH'].values:
            predict_way = byqbh[byqbh['BYQBH'] == predict_way]['CLDBH']
        else:
            print_str = 'Warning：暂不支持预测指定变压器，请输入6029台已训练的变压器。'
            print(print_str, 'Time:', datetime.datetime.now())
            # fou.update_data(pd.DataFrame([[run_date, 1, print_str]], columns=['rundate', 'state', 'error']))
            state_ = pd.DataFrame([[run_date, 1, print_str, job_parameter, 'predict.sh',
                                    str(datetime.datetime.now()).split('.')[0]]])
            state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
            sys.exit(1)
        target_cldbh = target_cldbh[target_cldbh['CLDBH'] == int(predict_way)]
        if target_date != '-1':
            try:
                datetime.datetime.strptime(target_date, '%Y-%m-%d')  # str(d1) = '2017-01-01 00:00:00'
            except Exception as err:
                print(err, 'Time:', datetime.datetime.now())
                # fou.update_data(pd.DataFrame([[run_date, 1, err]], columns=['rundate', 'state', 'error']))
                state_ = pd.DataFrame([[run_date, 1, err, job_parameter, 'predict.sh',
                                        str(datetime.datetime.now()).split('.')[0]]])
                state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
                sys.exit(1)
            res = fmm.predict_oneday_onecldbh(predict_way, byqrl, start_data, target_date, model_name, version_,
                                              run_date, job_parameter)
            res = pd.merge(res, byqbh, on='CLDBH', how='left')
            res.to_csv(output_dir, index=False, encoding='utf-8', header=None)
            # fou.insert_data(res, fgl.FHYC_FORECAST_JOB_RESULT)
            print('预测结果文件写于：'+output_dir, 'Time:', datetime.datetime.now())
            # fou.update_data(pd.DataFrame([[run_date, 0, '']], columns=['rundate', 'state', 'error']))
            state_ = pd.DataFrame([[run_date, 0, '', job_parameter, 'predict.sh',
                                    str(datetime.datetime.now()).split('.')[0]]])
            state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
            sys.exit(0)

    total_num = target_cldbh.shape[0]   # 需要预测的总变压器数量
    temp_num = 0    # 已经预测的总变压器数量
    # for循环控制执行预测的次数
    for i in range(1):
        # 日期偏置项
        delta = datetime.timedelta(days=i)
        # 定义输入文件路径
        input_dir1 = fgl.INPUT_DIR+'/PredictInput'+''.join(str(d1-delta).split(' ')[0].split('-'))+'.csv'
        input_dir2 = fgl.INPUT_DIR+'/PredictInput'+''.join(str(d1-2*delta).split(' ')[0].split('-'))+'.csv'
        input_dir3 = fgl.INPUT_DIR+'/PredictInput'+''.join(str(d1-3*delta).split(' ')[0].split('-'))+'.csv'

        # 检测是否还有变压器需要预测，没有的话跳出循环
        if target_cldbh.empty:
            print('所有变压器都全力预测！', 'Time:', datetime.datetime.now())
            break
        # 基于当前输入文件预测未来30天，如果输入文件存在则预测，否则给出没有输入文件的标志
        if os.path.exists(input_dir1) and os.path.exists(input_dir2) and os.path.exists(input_dir3):
            print('第'+str(i+1)+'次预测，'+'预测'+str(30-i)+'天', 'Time:', datetime.datetime.now())

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

            input_ = pd.merge(pd.DataFrame(target_cldbh['CLDBH']), input_, on='CLDBH', how='left')
            input_['date'] = pd.to_datetime(input_['date'])

            # 根据前三天的数据是否有空值判断该变压器是否可以预测，没空值的部分传给input_，空值部分传给target_cldbh进入待预测状态
            target_cldbh = pd.DataFrame(input_.loc[input_.loc[:, ['yggl', 'yggl_x', 'yggl_y']]
                                        .isnull().any(axis=1)==True, :]['CLDBH'])  # 下一批待预测测量点编号
            input_ = input_.loc[input_.loc[:, ['yggl', 'yggl_x', 'yggl_y']].isnull().any(axis=1)==False, :]    # 当前预测变压器集合

            # # 过滤变压器
            # filter_ = pd.read_csv('Original/predict_cldbh_0.5.csv', usecols=['CLDBH'])
            # input_ = pd.merge(filter_, input_, on='CLDBH', how='inner')

            # 判断当前input_集合是否需要预测
            if input_.empty:
                # 历史三天yggl输入存在空值
                print_str = '该变压器历史三天yggl输入存在空值。0台变压器被预测！'
                print(print_str, 'Time:', datetime.datetime.now())
                # fou.update_data(pd.DataFrame([[run_date, 1, print_str]], columns=['rundate', 'state', 'error']))
                state_ = pd.DataFrame([[run_date, 1, print_str, job_parameter, 'predict.sh',
                                        str(datetime.datetime.now()).split('.')[0]]])
                state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
                sys.exit(1)

            temp_cldbh = pd.DataFrame(input_['CLDBH'])  # 当前预测测量点编号
            input_.reset_index(drop=True, inplace=True)     # CLDBH date yggl,可预测的数据
            temp_num = temp_num + input_.shape[0]   # 记录已预测变压器数量

            # 对input_扩充待预测的30天的时间记录
            temp_cldbh['temp'] = 1
            temp_date = pd.DataFrame([], columns=['date', 'temp'])
            temp_date['date'] = pd.date_range(start=str(d1-delta).split(' ')[0], periods=31, freq='1D')
            temp_date['temp'] = 1
            test_set = pd.merge(temp_cldbh, temp_date, on='temp', how='left')
            del temp_cldbh['temp']
            del test_set['temp']
            del temp_date
            gc.collect()

            input_ = pd.merge(test_set, input_, on=['CLDBH', 'date'], how='left')     # 合并有功功率
            del test_set
            gc.collect()

            # 合并变压器台帐信息，来自Hive-刘建模
            if os.path.exists(fgl.TAIZHANG_FORM):
                input_ = fmm.taizhang_combination(input_.copy(), fgl.TAIZHANG_FORM)
            else:
                print_str = 'Error code 102：缺少输入文件。无法完整预测。请将设备台账文件上传至路径:'+fgl.TAIZHANG_FORM+'！'
                print(print_str, 'Time:', datetime.datetime.now())
                # fou.update_data(pd.DataFrame([[run_date, 1, print_str]], columns=['rundate', 'state', 'error']))
                state_ = pd.DataFrame([[run_date, 1, print_str, job_parameter, 'predict.sh',
                                        str(datetime.datetime.now()).split('.')[0]]])
                state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
                sys.exit(1)

            # 合并天气属性
            w_input = fgl.INPUT_DIR+'/WeatherForecast'+''.join(str(d1-delta).split(' ')[0].split('-'))+'.csv'
            if os.path.exists(fgl.WEATHER_FORM):
                input_ = fmm.weather_combination(input_.copy(), fgl.WEATHER_FORM)
            elif os.path.exists(w_input):
                input_ = fmm.weather_combination(input_.copy(), w_input)
            else:
                print_str = 'Error code 103：缺少输入文件。无法完整预测。请将天气文件上传至路径:'+w_input+'！'
                print(print_str, 'Time:', datetime.datetime.now())
                # fou.update_data(pd.DataFrame([[run_date, 1, print_str]], columns=['rundate', 'state', 'error']))
                state_ = pd.DataFrame([[run_date, 1, print_str, job_parameter, 'predict.sh',
                                        str(datetime.datetime.now()).split('.')[0]]])
                state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
                sys.exit(1)

            # 合并节假日信息
            if os.path.exists(fgl.HOLIDAY_FORM):
                input_ = fmm.volication_combination(input_.copy(), fgl.HOLIDAY_FORM)
            else:
                print_str = 'Error code 104：缺少输入文件。无法完整预测。请将设备台账文件上传至路径:'+fgl.HOLIDAY_FORM+'！'
                print(print_str, 'Time:', datetime.datetime.now())
                # fou.update_data(pd.DataFrame([[run_date, 1, print_str]], columns=['rundate', 'state', 'error']))
                state_ = pd.DataFrame([[run_date, 1, print_str, job_parameter, 'predict.sh',
                                        str(datetime.datetime.now()).split('.')[0]]])
                state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
                sys.exit(1)

            # 合并经济增长信息
            if os.path.exists(fgl.ECONOMIC_FORM):
                input_ = fmm.economic_combination(input_.copy(), fgl.ECONOMIC_FORM)
            else:
                print_str = 'Error code 105：缺少输入文件。无法完整预测。请将设备台账文件上传至路径:'+fgl.ECONOMIC_FORM+'！'
                print(print_str, 'Time:', datetime.datetime.now())
                # fou.update_data(pd.DataFrame([[run_date, 1, print_str]], columns=['rundate', 'state', 'error']))
                state_ = pd.DataFrame([[run_date, 1, print_str, job_parameter, 'predict.sh',
                                        str(datetime.datetime.now()).split('.')[0]]])
                state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
                sys.exit(1)

            # 进行预测
            res = fmm.thirty_day_predicter(input_, byqrl, 30-i, model_name, version_, run_date, job_parameter)
            print(str(temp_num)+'台变压器完成预测，总共'+str(total_num)+'台', 'Time:', datetime.datetime.now())
            res = pd.merge(res, byqbh, on='CLDBH', how='left')
            res.to_csv(output_dir, index=False, encoding='utf-8', header=None)
            # fou.insert_data(res, fgl.FHYC_FORECAST_JOB_RESULT)
            print('预测结果文件写于：'+output_dir, 'Time:', datetime.datetime.now())
            # fou.update_data(pd.DataFrame([[run_date, 0, '']], columns=['rundate', 'state', 'error']))
            state_ = pd.DataFrame([[run_date, 0, '', job_parameter, 'predict.sh',
                                    str(datetime.datetime.now()).split('.')[0]]])
            state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
            sys.exit(0)

        else:
            print_str = 'Warning：缺少输入文件。无法完整预测。请将指定日期'+str(d1-delta).split(' ')[0]+'及前三天的的有功功率文件上传至路径:'+fgl.INPUT_DIR+'！'
            print(print_str, 'Time:', datetime.datetime.now())
            # fou.update_data(pd.DataFrame([[run_date, 1, print_str]], columns=['rundate', 'state', 'error']))
            state_ = pd.DataFrame([[run_date, 1, print_str, job_parameter, 'predict.sh',
                                    str(datetime.datetime.now()).split('.')[0]]])
            state_.to_csv(fgl.OUTPUT_DIR+'/job_state.csv', encoding='utf-8', header=None, index=False)
            sys.exit(1)
    # if target_cldbh.empty:
    #     print('所有变压器都全力预测！', 'Time:', datetime.datetime.now())
    # else:
    #     # log_write(pd.DataFrame(target_cldbh['CLDBH']), 0, log_dir)
    #     print(str(target_cldbh.shape[0])+'台变压器过去30天都无输入文件，无法进行预测！', 'Time:', datetime.datetime.now())
