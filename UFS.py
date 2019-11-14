# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mutual_info_score

def MI_cont(X, y, k=3):
    if len(X.shape) == 1:
        N = X.shape[0]
        X = X.reshape((N,1))
    return mutual_info_regression(X, y, n_neighbors=k)

def MI_cata(X, y):
    if len(X.shape) == 1:
        N = X.shape[0]
    elif X.shape[0] == 1:
        N = X.shape[1]
    elif X.shape[1] == 1:
        N = X.shape[0]
    X = X.reshape(N)
    y = y.reshape(N)
    return mutual_info_score(X,y)

class UFS_MI:
    def __init__(self, norm_label=False):
        self.norm_label = norm_label
        pass
    def training(self, X, cata_info):
        self.X = X
        self.cata_info = cata_info
        [N, M] = X.shape
        M_cata = len(cata_info)
        M_cont = M - M_cata
        self.M = M
        self.M_cata = M_cata
        self.M_cont = M_cont
        self.N = N
        
#        if M_cata == 0:
#            return self.UFS_MI_Cont(X)
#        if M_cata == M:
#            return UFS_MI_Cata(X)
        
        cont_info = []
        for it in range(M):
            if not (it in cata_info):
                cont_info.append(it)
        self.cont_info = cont_info
        
        X_cont = X[:,cont_info]
        X_cata = X[:,cata_info]
        if self.norm_label == True:
            X_cont_mean = np.mean(X_cont, axis=0).reshape((1,M_cont))
            X_cont_std = np.std(X_cont, axis=0).reshape((1,M_cont))
            X_cont = (X_cont-X_cont_mean) / X_cont_std
            
        self.X_cont = X_cont
        self.X_cata = X_cata
        (cata_idx, cata_IGR) = self.training_cata()
        (cont_idx, cont_IGR) = self.training_cont()
        self.cata_idx = cata_idx
        self.cata_IGR = cata_IGR
        self.cont_idx = cont_idx
        self.cont_IGR = cont_IGR
        
        all_idx = np.zeros(M, dtype=np.int)
        all_IGR = np.zeros(M, dtype=np.float)
        it_cont = 0
        it_cata = 0
        it = 0
        while it_cont < M_cont or it_cata < M_cata: 
            if (it_cont == M_cont):
                all_idx[it] = cata_info[cata_idx[it_cata]]
                all_IGR[it] = cata_IGR[it_cata]
                it_cata += 1
            elif (it_cata >= M_cata) or (cata_IGR[it_cata] < cont_IGR[it_cont]):
                all_idx[it] = cont_info[cont_idx[it_cont]]
                all_IGR[it] = cont_IGR[it_cont]
                it_cont += 1
            else:
                all_idx[it] = cata_info[cata_idx[it_cata]]
                all_IGR[it] = cata_IGR[it_cata]
                it_cata += 1
            it += 1
        self.all_idx = all_idx
        self.all_IGR = all_IGR
        return (all_idx, all_IGR)
    
    def training_cont(self):
        if self.M_cont == 0:
            return ([],[])
        IMI_cont = np.zeros((self.M_cont, self.M_cont))
        IR_cont = np.zeros(self.M_cont)
        IRed_cont = np.zeros((self.M_cont, self.M_cont))
        for it in range(self.M_cont):
            for jt in range(self.M_cont):
                IMI_cont[it,jt] = MI_cont(self.X_cont[:,it],self.X_cont[:,jt])
        for it in range(self.M_cont):
            for jt in range(self.M_cont):
                IR_cont[it] += IMI_cont[it,jt]/self.M_cont
        for it in range(self.M_cont):
            for jt in range(self.M_cont):
                IRed_cont[it,jt] = IMI_cont[it,jt] / IMI_cont[jt,jt] * IR_cont[jt]
        self.IMI_cont = IMI_cont
        self.IR_cont = IR_cont
        self.IRed_cont = IRed_cont
        
        cont_idx = np.zeros(self.M_cont, dtype=np.int)
        cont_IGR = np.zeros(self.M_cont)
        cont_idx[0] = np.argmax(IR_cont)
        cont_IGR[0] = IR_cont[cont_idx[0]]
        slct_set = set()
        slct_set.add(cont_idx[0])
        for it in range(1, self.M_cont):
            max_IGR = -1e8
            max_idx = -1
            for jt in range(self.M_cont):
                if not jt in slct_set:
                    max_IRed = -1.0
                    for kt in slct_set:
                        if IRed_cont[jt,kt] > max_IRed:
                            max_IRed = IRed_cont[jt,kt]
                    #print(max_IRed)
                    tmp_IGR = IR_cont[jt] - max_IRed
                    if tmp_IGR > max_IGR:
                        max_IGR = tmp_IGR
                        max_idx = jt
            cont_idx[it] = max_idx
            cont_IGR[it] = max_IGR
            slct_set.add(max_idx)
        return (cont_idx,cont_IGR)
    
    def training_cata(self):
        if self.M_cata == 0:
            return ([],[])
        IMI_cata = np.zeros((self.M_cata, self.M_cata))
        IR_cata = np.zeros(self.M_cata)
        IRed_cata = np.zeros((self.M_cata, self.M_cata))
        for it in range(self.M_cata):
            for jt in range(self.M_cata):
                IMI_cata[it,jt] = MI_cata(self.X_cata[:,it],self.X_cata[:,jt])
        for it in range(self.M_cata):
            for jt in range(self.M_cata):
                IR_cata[it] += IMI_cata[it,jt]/self.M_cata
        for it in range(self.M_cata):
            for jt in range(self.M_cata):
                IRed_cata[it,jt] = IMI_cata[it,jt] / IMI_cata[jt,jt] * IR_cata[jt]
        self.IMI_cata = IMI_cata
        self.IR_cata = IR_cata
        self.IRed_cata = IRed_cata
        
        cata_idx = np.zeros(self.M_cata, dtype=np.int)
        cata_IGR = np.zeros(self.M_cata)
        cata_idx[0] = np.argmax(self.IR_cata)
        cata_IGR[0] = IR_cata[cata_idx[0]]
        slct_set = set()
        slct_set.add(cata_idx[0])
        for it in range(1, self.M_cata):
            max_IGR = -1e8
            max_idx = -1
            for jt in range(self.M_cata):
                if not jt in slct_set:
                    max_IRed = -1.0
                    for kt in slct_set:
                        if IRed_cata[jt,kt] > max_IRed:
                            max_IRed = IRed_cata[jt,kt]
                    tmp_IGR = IR_cata[jt] - max_IRed
                    if tmp_IGR > max_IGR:
                        max_IGR = tmp_IGR
                        max_idx = jt
            cata_idx[it] = max_idx
            cata_IGR[it] = max_IGR
            slct_set.add(max_idx)
        return (cata_idx,cata_IGR)

if __name__ == '__main__':
    file_raw = 'D:/Duke/Huawei/SpeedTest/data_for_VTS/exp1/exp1_new.csv'
    pd_raw = pd.read_csv(file_raw, index_col=0)
    imp_col = ["DOWNLOAD_NNI_RTT_MS", "DOWNLOAD_UNI_RTT_MS", \
                            "DOWNLOAD_NNI_PLR", "DOWNLOAD_UNI_PLR", \
                            "UP_DATA_RTT_MS", \
                            "PPPOE_DELAY_MS", \
                            "DOWN_DATA_RTT_MS", \
                            "HOUR", \
                            "UPLOAD_NNI_RTT_MS", "UPLOAD_UNI_RTT_MS", \
                            "UPLOAD_NNI_PLR", "UPLOAD_UNI_PLR", \
                            "APPLICATION_PROTOCOL", "REMOTE_IP", "DEVICE_TYPE", \
                            "WAN_TYPE_DYNAMIC", "DEVICE", "PORT", \
                            "EDSG_VALID", "CPE_VENDOR", "CPE_MODEL", "CPE_TYPE", \
                            "CC", "COUNTRY", "CITY", "SPONSOR", "SERVER_URL", \
                            "USER_PER_PORT", "BW_PER_PORT", "BW_PER_PORT2", \
                            "USER_PER_DEVICE", "BW_PER_DEVICE", "BW_PER_DEVICE2", \
                            "DNS_OWNER" \
                            ]
    cata_info = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 33]
    pd_raw = pd_raw[imp_col]
    pd_raw.dropna(axis=0, how='any',inplace=True)
    X = pd_raw.values#[:1000,:]
    #print(X.shape)
#    X = np.random.randint(100,size=(100,10))
#    X_cata = np.random.randint(5,size=(100,10))
#    X = np.concatenate([X,X,X_cata,X_cata],axis=1)
    ufs_mi = UFS_MI(norm_label=True)
    ufs_mi.training(X,cata_info)