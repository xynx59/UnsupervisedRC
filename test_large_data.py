# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:45:32 2019

@author: 93986
"""

import pandas as pd
import numpy as np
from Speedtest_data_preprocess import SpeedTestDataTransform, DataDeepTransform
from Speedtest_data_preprocess import numerization_part, numerization_test
from Unsupervised_RC_Generic import RC_Analysis
from UFS import UFS_MI

label_pre = True
label_imp = True
label_numer = True
label_missing = True
label_fs = True
label_analysis = False
IGR_th = 0.05
imp_feature = ["DOWNLOAD_NNI_RTT_MS", "DOWNLOAD_UNI_RTT_MS", \
                "DOWNLOAD_NNI_PLR", "DOWNLOAD_UNI_PLR", \
                "UP_DATA_RTT_MS", \
                "PPPOE_DELAY_MS", \
                "DOWN_DATA_RTT_MS", \
                "END_DEVICE_UPTIME_HR", \
                "UPLOAD_NNI_RTT_MS", "UPLOAD_UNI_RTT_MS", \
                "UPLOAD_NNI_PLR", "UPLOAD_UNI_PLR", \
                "APPLICATION_PROTOCOL", "REMOTE_IP", "DEVICE_TYPE", \
                "WAN_TYPE_DYNAMIC", "DEVICE", "PORT", \
                "EDSG_VALID", "CPE_VENDOR", "CPE_MODEL", "CPE_TYPE", \
                "CITY", "SPONSOR", "SERVER_URL", \
                "USER_PER_PORT", "BW_PER_PORT", "BW_PER_PORT2", \
                "USER_PER_DEVICE", "BW_PER_DEVICE", "BW_PER_DEVICE2", \
                "DNS_OWNER" \
                ]


if label_pre:
    """ Data Preprocessing """
    csv_file = './sp0613/raw_DL_BW_201_to_10000.csv'
    df = pd.read_csv(csv_file, index_col=[0])
    obj = SpeedTestDataTransform(df)
    obj.intial_cleanup()
    obj.filter_machine_test(thresh=20)
    obj.convert_result()
    obj.convert_uptime(date_format="%Y-%m-%d %H:%M:%S", check_format=True)
    obj.update_port()
    obj.keep_based_on_columns()
    obj.deep_cleanup(precise=False, date_format="%Y-%m-%d %H:%M:%S")
    obj.df.drop(columns=['ACCOUNT', 'DATE_TIME', 'DAY', 'HOUR'], inplace=True)
    obj.remove_sparse_col(sparse_th=0.7)
    deep_feature=['LOCAL_IP', 'REMOTE_UP', 'IP_TYPE', 'IP_TYPE2', 'DEVICE_TYPE', \
                  'BRAS', 'WAN_TYPE_DYNAMIC', 'DEVICE', 'PORT', 'APPLICATION_PROTOCOL', \
                  'CC', 'COUNTRY', 'CITY', 'SPONSOR', 'SERVER_URL', 'EDSG', 'EDSG_VALID', \
                  'CPE_VENDOR', 'CPE_MODEL', 'CPE_TYPE',  'UE_TYPE', 'DNS_OWNER']
    N = obj.df.shape[0]
    obj2 = DataDeepTransform(obj.df)
    obj2.label_nan_cata(features=deep_feature)
    obj2.remove_minor_cata(nth=np.int(N*0.01), features=deep_feature)
    pd_pre = obj.df
    pd_pre.to_csv('./sp0613/data_pre.csv')
else:
    pd_pre = pd.read_csv('./sp0613/data_pre.csv')
    
if label_imp:
    """ Only consider important features """
    Col = pd_pre.columns.values
    Col_slct = imp_feature
    if 'RESULT' in Col:
        Col_slct.append('RESULT')
    if 'FAULTCAUSE' in Col:
        Col_slct.append('FAULTCAUSE')
    pd_pre =pd_pre[Col_slct]
    
if label_numer:
    """ Numerize Catagorical Data """
    pd_numer, pd_bound, cata_name = numerization_part(pd_pre)
    pd_numer.to_csv('./sp0613/data_pre.csv')
    if label_missing:
        """ Handle Missing Value """
        pd_numer = pd_numer.fillna(-1)
    drop_list=['RESULT']
    if 'FAULTCAUSE' in pd_numer.columns.values:
        drop_list.append('FAULTCAUSE')
    pd_feature = pd_numer.drop(drop_list, axis=1)
    X = pd_feature.values
    Col = pd_feature.columns.values
    y = pd_numer[['RESULT']].values
    cata_info = []
    M = X.shape[1]
    for it in range(M):
        if Col[it] in cata_name:
            cata_info.append(it)

if label_fs:
    """ Feature Selection """
    N_fs = np.max([X.shape[0], 500])
    Idx = np.random.choice(X.shape[0], N_fs, replace=True)
    X_fs = X[Idx, :]
    ufs_mi = UFS_MI(norm_label=True)
    ufs_mi.training(X_fs, cata_info)
    M_imp = 0
    M = len(ufs_mi.all_IGR)
    for it in range(M):
        if ufs_mi.all_IGR[it] < IGR_th:
            M_imp = it
            break
    Col_slct = Col[ufs_mi.all_idx[:M_imp]]
    Col_slct = np.append(Col_slct, ["RESULT"])
    pd_training = pd_numer[Col_slct]
    
if label_analysis:
    """ RC Analysis """
    RC = RC_Analysis()
    RC.training(max_depth=4, pd_data=pd_training, cata_info=cata_info)# raw_DL_BW_100_UL_BW_50 #plan100_50_drop_duplicate_test
    #RC.KMeans_DT_clustering(n_clusters=10, max_depth=4, min_impurity_split=0.05, min_samples_split=5)
    thresh_cnt = 200
    imp_th = 0.01
    RC.DT_Stage_I_Clustering(max_depth=4, min_impurity_split=0.05, min_samples_split=thresh_cnt)
    RC.FP_Stage_II_Clustering(imp_th=imp_th)
#    RC.test_DT_FP(file_name='./sp0613/exp2_test.csv', outfile = './exp2_predict.csv', thresh_cnt = thresh_cnt)
#    RC.validate_DT_FP(file_name='./sp0613/exp2_test.csv', outfile='./exp2_RC_predict.csv', thresh_cnt = thresh_cnt)
