# -*- coding: utf-8 -*-


"""
@author River
Date 20171103
"""

"""
定义全局变量
"""
import sys


# 定义工作空间和基础特征表路径
#WORK_LIST = '/home/ncu/fhyc/Data'
# WORK_LIST = '.'
WORK_LIST = sys.argv[1]

CEDIAN_FORM_2017 = WORK_LIST + '/th_data_sslsd_2017_hive.csv'   # 2017年测点数据表
CEDIAN_FORM_2016 = WORK_LIST + '/th_data_sslsd_2016_hive.csv'   # 2016年测点数据表
TAIZHANG_FORM = WORK_LIST + '/Original/PBYC6092data.csv'     # 设备台账信息表
WEATHER_FORM = WORK_LIST + '/Original/weather4.csv'      # 天气信息表
HOLIDAY_FORM = WORK_LIST + '/Original/holiday.csv'       # 节假日信息表
ECONOMIC_FORM = WORK_LIST+'/Original/economic_growth.csv'    # 经济增长信息表
TRANSINFO_FORM = WORK_LIST+'/TransformerInfo.csv'  # 负荷预测数据挖掘信息总表

# 模型训练结果存储
MODEL_DIR = WORK_LIST + '/Model'
RESULT_DIR = WORK_LIST + '/Result'

# 模型预测结果存储
INPUT_DIR = WORK_LIST+'/Input'
OUTPUT_DIR = WORK_LIST+'/Output'

"""
    # oracle数据库中的表名为大写
    # FHYC_JOB_STATE：预测作业执行状态表
    # FHYC_FORECAST_JOB_RESULT：预测作业执行结果表
    # FHYC_TRANSFORMERINFO_STATE: 负荷预测数据状态表
    
"""
# ORACLE数据库表配置
FHYC_DATABASE_URL = 'pbdb/tellhow_bigdata@192.168.14.29/PMCSDB'
FHYC_JOB_STATE = 'FHYC_FORECAST_JOB_STATE'
FHYC_FORECAST_JOB_RESULT = 'FHYC_FORECAST_JOB_RESULT'
FHYC_TRANSFORMERINFO_STATE = 'FHYC_TRANSFORMERINFO_STATE'

