# -*- coding: utf-8 -*-


"""
@author River
Date 20171218
"""
import cx_Oracle
import fhyc_global_list as fgl
import pandas as pd


"""
资源获取与关闭
"""


def open_oracle():
    conn = cx_Oracle.connect(fgl.FHYC_DATABASE_URL)
    cursor = conn.cursor()

    table_list = [fgl.FHYC_FORECAST_JOB_STATE, fgl.FHYC_FORECAST_JOB_RESULT, fgl.FHYC_TRANSFORMERINFO_STATE]
    create_list = {
        table_list[0]: "create table "+fgl.FHYC_FORECAST_JOB_STATE+"( rundate DATE not null primary key, "
                                                                   "parameter VARCHAR(60) NOT NULL, "
                                                                   "state NUMBER(2) DEFAULT 2, error  "
                                                                   "VARCHAR(100) DEFAULT 'Abnormal "
                                                                   "interruption!', jobtype VARCHAR(30) "
                                                                   "not null, endtime DATE not null)",
        table_list[1]: "create table "+fgl.FHYC_FORECAST_JOB_RESULT+"(CLDBH INTEGER NOT NULL , date_ DATE NOT NULL , "
                       "predict_yggl FLOAT NOT NULL, suanfaname VARCHAR(60) NOT NULL, rundate DATE not null, "
                       "CONSTRAINT PK_ PRIMARY KEY (CLDBH, date_, rundate))",
        table_list[2]: "CREATE TABLE "+fgl.FHYC_TRANSFORMERINFO_STATE+"( update_time DATE NOT NULL PRIMARY KEY, "
                       "date_size VARCHAR(20) NOT NULL, last_date  DATE NOT NULL)"
    }
    for item in table_list:
        sql = "select count(*) from user_tables where table_name = '"+item+"'"
        cursor.execute(sql)
        if cursor.fetchone()[0] == 0:
            try:
                cursor.execute(create_list[item])
                if item is fgl.FHYC_TRANSFORMERINFO_STATE:
                    df = pd.read_csv(fgl.TRANSINFO_FORM, usecols=['date'])

                    sql = "INSERT INTO FHYC_TRANSFORMERINFO_STATE VALUES (to_date('" + df.max()['date'] +\
                          "','yyyy-mm-dd,hh24:mi:ss'), "+str(df.shape[0])+", to_date('2017-01-01','yyyy-mm-dd,hh24:mi:ss'))"
                    cursor.execute(sql)
                    conn.commit()
            except Exception as err:
                print("ERROR:", err)
    return conn, cursor


def close_oracle(conn, cursor):
    cursor.close()
    conn.close()


"""
数据插入逻辑
"""


def insert(x, table_name, insert_cursor):
    sql = None
    try:
        # print(table_name is fgl.FHYC_FORECAST_JOB_RESULT)
        if table_name is fgl.FHYC_FORECAST_JOB_RESULT:
            sql = "INSERT INTO "+table_name+" VALUES (" + \
                  str(x['CLDBH'])+",to_date('"+str(x['date']) + \
                  "','yyyy-mm-dd,hh24:mi:ss'),"+str(x['predict_yggl']) + \
                  ",'"+x['suanfaname']+"',to_date('"+str(x['rundate']) + \
                  "','yyyy-mm-dd,hh24:mi:ss'))"
        elif table_name is fgl.FHYC_FORECAST_JOB_STATE:
            sql = "INSERT INTO "+table_name+" (rundate, parameter) VALUES (to_date('" + \
                  str(x['rundate'])+"','yyyy-mm-dd,hh24:mi:ss'), '" + x['parameter'] + "')"
        elif table_name is fgl.FHYC_TRANSFORMERINFO_STATE:
            sql = "INSERT INTO "+table_name+" VALUES (to_date('" + \
                  str(x['update_time']) + "','yyyy-mm-dd,hh24:mi:ss'), " + \
                  str(x['data_size']) + ", to_date('" + \
                  str(x['last_date'])+"','yyyy-mm-dd,hh24:mi:ss'))"
        insert_cursor.execute(sql)
    except Exception as err:
        print('ERROR', err)


def insert_data(id_df, table_name):
    id_conn, id_cursor = open_oracle()
    id_df.apply(insert, axis=1, args=(table_name, id_cursor))
    id_conn.commit()
    close_oracle(id_conn, id_cursor)


"""
数据更新逻辑
"""


def update(x, update_cursor, conn):
    try:
        sql = "UPDATE "+fgl.FHYC_FORECAST_JOB_STATE+" SET STATE="+str(x['state'])+", ERROR='"+str(x['error']) + \
              "' WHERE rundate = to_date('"+str(x['rundate'])+"','yyyy-mm-dd,hh24:mi:ss')"
        # print(sql)
        update_cursor.execute(sql)
    except Exception as err:
        print('ERROR', err)


def update_data(ud_df):
    ud_conn, ud_cursor = open_oracle()
    ud_df.apply(update, axis=1, args=(ud_cursor, 1))
    ud_conn.commit()
    close_oracle(ud_conn, ud_cursor)
