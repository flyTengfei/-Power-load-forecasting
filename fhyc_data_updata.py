# -*- coding: utf-8 -*—

import sys
import pandas as pd
from dateutil.parser import parse
import datetime

import fhyc_global_list as fgl
import fhyc_mutil_model as fmm
# import fhyc_oracle_unit as fou


def data_split(start_time, df):
    start_time = parse(start_time)
    delta = datetime.timedelta(days=-2)
    end_time = start_time+delta

    cd_ = df.loc[:, ['CLDBH', 'date', 'yggl']]
    cd_['date'] = pd.to_datetime(cd_['date'])
    cd_ = cd_[(cd_['date'] >= str(end_time)) & (cd_['date'] <= str(start_time))]

    for p, q in cd_.groupby('date'):
        input_dir = fgl.INPUT_DIR+'/PredictInput'+''.join(str(p).split(' ')[0].split('-'))+'.csv'
        q.to_csv(input_dir, index=False, encoding='utf-8')


def data_updata():

    # 获取预测执行时间
    # run_date = str(datetime.datetime.now()).split('.')[0]     # 记录预测作业开始时间，作为该次作业的唯一标注
    run_date = str(sys.argv[2]).split('.')[0]     # 记录预测作业开始时间，作为该次作业的唯一标注

    # 获取更新模式：updata_dir为一个路径则为定期数据更新，为-1则为日常数据更新，即将每天的input数据更新至TransformerInfo.csv
    updata_dir = str(sys.argv[3])

    if updata_dir != '-1':
        updata_dir = [updata_dir]
    else:
        updata_dir = [fgl.WORK_LIST + '/Original/temp1.csv',
                      fgl.WORK_LIST + '/Original/temp2.csv',
                      fgl.WORK_LIST + '/Original/temp3.csv']
    print('data updata is start.....', 'Time:', datetime.datetime.now())
    size, last_date, transformerinfo_df = fmm.updata_transinfo(fgl.TRANSINFO_FORM, updata_dir, fgl.TAIZHANG_FORM, fgl.WEATHER_FORM, fgl.HOLIDAY_FORM, fgl.ECONOMIC_FORM)
    # fou.insert_data(pd.DataFrame([[run_date, size, last_date]], columns=['update_time', 'data_size', 'last_date']), fgl.FHYC_TRANSFORMERINFO_STATE)
    print('data updata is over!', 'Time:', datetime.datetime.now())

    if updata_dir == -1:
        data_split(last_date, transformerinfo_df)


if __name__ == '__main__':
    data_updata()
