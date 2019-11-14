# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:36:55 2019

@author: r84130171
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy
import string
import seaborn as sns
from data_preprocess import SpeedTestDataTransform, DataDeepTransform
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
#from data_preprocessing import *
from sklearn.tree import DecisionTreeClassifier as DTC
from dt import DT_cata

#from export_speedtest import plot_tree
import time
from failure_pattern_mining_renjian import FeqPattern
from scipy.stats import binom_test
from collections import Counter

from evaluation_metrics import cluster_purity, cluster_NMI, cluster_AMI, intra_dist
from clustering_metrics import sklearn_AMI

import warnings
warnings.filterwarnings('ignore')

class RC_Analysis(object):

    
    def __init__(self):
        self.pd_data_raw = None
        self.pd_data = None
        self.np_feature = None
        self.np_results_raw = None
        self.np_results = None
        self.clf_DT_clustering = None
        self.cata_set_list = None
        
    def preprocessing(self):
        pd_feature = self.pd_data.drop(['RESULT'],axis=1)
        self.pd_feature = pd_feature
        self.np_feature = pd_feature.values
        self.np_results = self.pd_data['RESULT'].values
        self.imp_feature = self.pd_feature.columns.values
        self.N = self.np_feature.shape[0]
        self.M = self.np_feature.shape[1]
        self.M_cata = np.int(len(self.cata_info))
        if self.pd_bounds == []:
            self.np_bounds = np.zeros((np.int(self.N*0.1),self.M))
            for it in self.cata_info:
                max_B = np.int(np.max(self.np_bounds[:,it]))
                for jt in range(max_B):
                    self.np_bounds[jt,it] = jt
            self.pd_bounds = pd.DataFrame(data=self.np_bounds, columns=self.imp_feature)
    
    def training(self, method='DT', max_depth=8, file_name=None, pd_data=None, cata_info=[], pd_bounds=[]):
        if not file_name ==None:
            pd_data = pd.read_csv(file_name, index_col=[0])
        self.pd_data = pd_data
        self.cata_info = cata_info
        self.pd_bounds = pd_bounds
        self.preprocessing()
        if method == 'DT':
            self.training_DT(max_depth = max_depth)
        if method == 'Freq':
            self.training_Freq()
        if method == 'all':
            self.training_DT(max_depth = max_depth)
            self.training_Freq()

    def training_DT(self, max_depth=8, min_impurity_split=0.05, min_samples_split=100):
        """
            Classification
        """
        X_train = self.np_feature
        y_train = self.np_results
        features = self.imp_feature
        self.clf_DT = DT_cata(max_depth = max_depth, min_impurity_split=min_impurity_split, \
                              min_samples_split = min_samples_split, feature_name = features, \
                              cata_info = self.cata_info, cata_list = self.pd_bounds[self.imp_feature].values)
        self.clf_DT.fit(X_train, y_train)
        #self.RC_DT()
        
    def DT_Stage_I_Clustering(self, max_depth = 4, min_impurity_split=0.05, min_samples_split=100):
        X_train = self.np_feature
        y_train = self.np_results
        features = self.imp_feature
        cata_info = self.cata_info
        (N, M) = X_train.shape
        
        self.clf_DT_clustering = DT_cata(max_depth = max_depth, min_impurity_split=min_impurity_split,\
                                         min_samples_split = min_samples_split, \
                                         feature_name = features, cata_info = cata_info, \
                                         cata_list = self.pd_bounds[features].values)
        self.clf_DT_clustering.fit(X_train, y_train)
        
        queue = [self.clf_DT_clustering.trunk]
        queue_str = [""]
        tmp_set = set()
        queue_feature = [tmp_set]
        tmp_valid_set = set()
        queue_valid_feature = [tmp_valid_set]
        
        head = 0
        tail = 0
        tot_cluster = 0
        self.dict_str = dict()
        self.dict_node = dict()
        self.map_node_str = dict()
        self.map_node_feature = dict()
        self.dict_feature = dict()
        self.idx_to_feature = dict()
        self.idx_to_feature[0] = self.imp_feature
        while head <= tail:
            node = queue[head]
            node_str = queue_str[head]
            node_feature = queue_feature[head]
            node_valid_feature = queue_valid_feature[head]
            tmp_nf = copy.deepcopy(node_feature)
            if node.Feature == None:
                pass
            else:
                tmp_nf.add(features[node.Feature])
            
            if len(node.children) == 0:
                print(node_feature)
                if (node.ratio > 0.5) and (node.total > 50):
                    tot_cluster += 1
                    self.dict_str[node_str] = tot_cluster
                    self.dict_node[node] = tot_cluster
                    self.map_node_str[node] = node_str
                    self.dict_feature[node] = node_feature
                    self.idx_to_feature[tot_cluster] = node_feature
                    self.map_node_feature[node] = node_valid_feature
                    
                else:
                    self.dict_str[node_str] = 0
                    self.dict_node[node] = 0
                    self.map_node_str[node] = node_str
                    self.dict_feature[node] = node_feature
                    self.map_node_feature[node] = node_valid_feature
                    
            else:
                for child in node.children:
                    tail += 1
                    queue.append(child)
                    queue_str.append(node_str+'  <br>  '+child.name)
                    tmp_valid_feature = copy.deepcopy(node_valid_feature)
                    if True:#'<br> >' in child.name:
                        tmp_valid_feature.add(child.name[:child.name.find('<br>')])
                    queue_feature.append(tmp_nf)
                    queue_valid_feature.append(tmp_valid_feature)
            head += 1
                
        tot_cluster += 1
        self.tot_cluster = tot_cluster
        
        self.pd_cluster_list = []
        self.ratio_1st = np.zeros(N)
        self.idx_cluster = np.zeros((N), dtype=np.int)
        for it in range(N):
            node = self.clf_DT_clustering.trunk
            while not node.Feature == None:
                if X_train[it][node.Feature] <= node.TH:
                    node = node.TrueSon
                else:
                    node = node.FalseSon
            self.ratio_1st[it] = node.ratio
            self.idx_cluster[it] = self.dict_node[node]
            
        self.dict_idx_feat = dict()
        for node in self.dict_node.keys():
            node_idx = self.dict_node[node]
            self.dict_idx_feat[node_idx] = self.dict_feature[node]

        self.idx_list = []
        tmp_idx = np.zeros((N,tot_cluster), dtype=np.int)
        tmp_tot = np.zeros((tot_cluster), dtype=np.int)
        for it in range(N):
            cluster = self.idx_cluster[it]
            tmp_idx[tmp_tot,cluster] = it
            tmp_tot[cluster] += 1
            
        for cluster in range(tot_cluster):
            self.idx_list.append(tmp_idx[:tmp_tot[cluster], cluster])
        
    def anomaly_detection(self, max_depth=5, min_impurity_split=0.05, min_samples_split=100):
        if self.clf_DT_clustering == None:
            self.DT_clustering(max_depth=max_depth, min_impurity_split=min_impurity_split, \
                               min_samples_split=min_samples_split)
        self.ad_idx = []
        for it in self.idx_list[0]:
            if self.np_results[it] == 0:
                self.ad_idx.append(it)
        return self.ad_idx
    
    def FP_Stage_II_Clustering(self, label_feature='2nd', nth=10, imp_th=0.03):
        self.cluster_FP = []
        self.cluster_FP_pd_data = []
        self.cluster_binning_ref = []
        tmp_pd = copy.deepcopy(self.pd_data)
        obj_all = SpeedTestDataTransform(tmp_pd)
        obj_all.convert_float2cat()  # only add this process when needed, continuous data lost resolution through binning
        pd_data = obj_all.df
        self.cluster_binning_ref = obj_all.binning_ref
        self.cluster_feature_pool = []
        
        for cluster in range(self.tot_cluster):
            #slct_feature = copy.deepcopy(self.feature_2nd)
            feature_already = self.idx_to_feature[cluster]
            slct_feature = []
            for feature in self.imp_feature:
                if not feature in feature_already:
                    slct_feature.append(feature)
            tmp_feature = copy.deepcopy(slct_feature)
            self.cluster_feature_pool.append(tmp_feature)
            slct_feature.append('RESULT')
            cata_info_slct = []
            cata_feat_slct = []
            jt = 0
            for col in self.imp_feature:
                if (not jt in self.cata_info) or (not col in slct_feature):
                    pass
                else:
                    cata_info_slct.append(jt)
                    cata_feat_slct.append(col)
                jt += 1
            pd_cluster_raw = pd_data[slct_feature]
            pd_cluster_raw = pd_cluster_raw.iloc[self.idx_list[cluster],:]
            obj = SpeedTestDataTransform(pd_cluster_raw)
#            obj.convert_float2cat()  # only add this process when needed, continuous data lost resolution through binning
#            self.cluster_binning_ref.append(obj.binning_ref)
            obj2 = DataDeepTransform(obj.df)
            obj2.remove_minor_cata(nth=nth, features=cata_feat_slct)
            X_train = obj2.df.reset_index(drop=True)
            X_fail = X_train[X_train['RESULT'] == 'Fail'].reset_index(drop=True)
    
            tmp_FP = FeqPattern(X_train, support=nth, item_conf=0.5)
            self.cluster_FP_pd_data.append(X_train)
            tmp_FP.get_top_item()
            
            if len(tmp_FP.top_col_list) == 0:
                self.cluster_FP.append(None)
                continue

            tmp_FP.valid_dict = dict()
            tmp_FP.conf_dict = dict()
            tmp_FP.p_dict = dict()
            tmp_FP.impurity_diff_dict = dict()
            for it in range(len(tmp_FP.list_col)):
                col = tmp_FP.list_col[it]
                val = tmp_FP.list_v[it]
                tmp_key = col + '@' + str(val)
                if not tmp_key in tmp_FP.top_item_dict:
                    continue
                if str(val) == 'others' or str(val) == 'Others' or str(val) == 'Unknown' or str(val) == 'unknown':
                    tmp_FP.valid_dict[tmp_key] = False
                    tmp_FP.conf_dict[tmp_key] = 0
                    tmp_FP.impurity_diff_dict[tmp_key] = 0
                    tmp_FP.p_dict[tmp_key] = dict()
                    continue
                
                n_pf = X_train[X_train[col].notnull()].shape[0]
                n_fail = X_fail[X_fail[col].notnull()].shape[0]
                n_all = pd_data[pd_data[col].notnull()].shape[0]

                tmp_fail = tmp_FP.top_item_dict[tmp_key]['COUNT']
                tmp_pf = tmp_FP.top_item_dict[tmp_key]['COUNT'] / tmp_FP.top_item_dict[tmp_key]['RATIO']
                tmp_all = Counter(pd_data[col].values)[val]
                
                p_fail = np.float(tmp_fail)/n_fail
                p_pf = np.float(tmp_pf)/n_pf
                p_all = np.float(tmp_all)/n_all
#                
                imp_fail = p_fail*(1.0-p_fail)
                imp_pf = p_pf*(1.0-p_pf)
                imp_all = p_all*(1.0-p_all)
                
                max_diff = 0
                if p_fail > p_pf:
                    max_diff = np.max([np.abs(imp_pf-imp_fail), max_diff])
                if p_pf > p_all:
                    max_diff = np.max([np.abs(imp_all-imp_pf), max_diff])
                
                p_in = binom_test(tmp_fail, n_fail, np.float(tmp_pf)/n_pf)
                p_out = binom_test(tmp_pf, n_pf, np.float(tmp_all)/n_all)
                #print(np.float(tmp_fail), np.float(tmp_pf))
                #print(np.float(tmp_fail)/n_fail, np.float(tmp_pf)/n_pf, np.float(tmp_all)/n_all)
                tmp_FP.conf_dict[tmp_key] = 1.0-np.min([p_in, p_out])
                tmp_FP.p_dict[tmp_key] = {'Fail':(p_fail, tmp_fail, n_fail), 'PF':(p_pf, tmp_pf, n_pf), 'All':(p_all, tmp_all, n_all)}
                tmp_FP.impurity_diff_dict[tmp_key] = max_diff
                if max_diff > imp_th:
                    tmp_FP.valid_dict[tmp_key] = True
                else:
                    tmp_FP.valid_dict[tmp_key] = False
             
            fpt = tmp_FP.grow_fp_tree(patterns_to_grow='sequential')
            #fpt.prune_tree(thresh_cnt=30, thresh_conf=0.7)
            self.cluster_FP.append(tmp_FP)
            
    def test_DT_FP(self, file_name = None, pd_test = None, \
                   thresh_cnt=50, thresh_conf=0.95, outfile = './test_result.csv'):
        if not file_name == None:
            pd_test = pd.read_csv(file_name, index_col=[0])
        pd_test = pd_test.fillna(-1)
        self.RC_list = pd_test['FAULTCAUSE'].values.astype(np.int)
        #np_results = pd_test["RESULT"].values.astype(np.int)
        np_data = pd_test[self.imp_feature].values
        
        self.np_test_data = np_data
        (N, M) = np_data.shape
        self.np_test_node = self.clf_DT_clustering.predict_node(np_data)
        self.np_test_str = []
        self.np_test_fp = []
        self.np_test_dt = []
        self.np_test_pval = []
        self.np_test_pdist = []
        self.np_test_fp_raw = []
        self.np_test_pval_raw = []
        self.np_test_pdist_raw = []
        

        print(N)
        for it in range(N):
            #print(it)
            num_cluster = np.int(self.dict_node[self.np_test_node[it]])
            self.np_test_str.append(self.map_node_str[self.np_test_node[it]])
            self.np_test_dt.append(str(self.map_node_feature[self.np_test_node[it]]))
            if self.cluster_FP[num_cluster] == None:
                self.np_test_fp.append('')
                self.np_test_pval.append('')
                self.np_test_pdist.append('')
                continue
            
            #print(num_cluster)
            slct_feature = self.cluster_feature_pool[num_cluster]
            
            case = pd_test[slct_feature].loc[it]
            case_new = []
            binning_ref = self.cluster_binning_ref
            for item in case.iteritems():
                feature, value = item
                tmp_value = value
                if feature in binning_ref:
                    for k, v in binning_ref[feature].items():
                        pass
                        if value <= v:
                            tmp_value = k
                            break  
                case_new.append(tmp_value)
            case = pd.Series(data=case_new, index=case.index)
            #print(case.to_dict())
            fp = self.cluster_FP[num_cluster]
            pattern = fp.prep_pattern(case.to_dict()) 
            #result = case.at['DL_RESULT_RATIO']
            #print(pattern)
            matched, cnt, conf = fp.fail_fp_tree.get_longest_prefix(pattern, cnt_th=thresh_cnt)
            #print(matched, cnt, conf)
            matched_new = []
            matched_dict = dict()
            matched_pdist = dict()
            matched_dict_raw = dict()
            matched_pdist_raw = dict()
            for it in matched:
                if fp.valid_dict[it]:
                    matched_new.append(it)
                    matched_dict[it] = fp.impurity_diff_dict[it]
                    matched_pdist[it] = fp.p_dict[it]
                matched_dict_raw[it] = fp.impurity_diff_dict[it]
                matched_pdist_raw[it] = fp.p_dict[it]
            self.np_test_fp.append(str(matched_new))
            self.np_test_pval.append(str(matched_dict))
            self.np_test_pdist.append(str(matched_pdist))
            self.np_test_fp_raw.append(str(matched))
            self.np_test_pval_raw.append(str(matched_dict_raw))
            self.np_test_pdist_raw.append(str(matched_pdist_raw))
            
        self.pd_test_res = pd.concat([pd.DataFrame(data=self.np_test_str, columns=['Decison Path']), \
                                      pd.DataFrame(data=self.np_test_dt, columns=['Decison Feature']), \
                                      pd.DataFrame(data=self.np_test_fp, columns=['Frequent Pattern']), \
                                      pd.DataFrame(data=self.np_test_pval, columns=['FP Impurity Diff']), \
                                      pd.DataFrame(data=self.np_test_pdist, columns=['FP Dist']), \
                                      pd.DataFrame(data=self.np_test_fp_raw, columns=['Raw Frequent Pattern']), \
                                      pd.DataFrame(data=self.np_test_pval_raw, columns=['Raw FP Impurity Diff']), \
                                      pd.DataFrame(data=self.np_test_pdist_raw, columns=['Raw FP Dist']), \
                                      pd_test], \
                                      axis=1)
        self.pd_test_res.to_csv(outfile)
                    
    def validate_DT_FP(self, outfile='./db_fm/RC_predict.csv', thresh_cnt=50):

        tot_RC = np.int(np.max(self.RC_list)+1)
        N = len(self.RC_list)
        self.encode_DT_FP = dict()
        self.DT_FP_idx = np.zeros(N, dtype=np.int)
        tot_DT_FP = 0
        for it in range(N):
            tmp_str = self.np_test_str[it]+self.np_test_fp[it]
            if not tmp_str in self.encode_DT_FP:
                self.encode_DT_FP[tmp_str] = tot_DT_FP
                tot_DT_FP += 1
            self.DT_FP_idx[it] = self.encode_DT_FP[tmp_str]
        self.DT_FP_counter = Counter(self.DT_FP_idx)
        for it in range(N):
            if self.DT_FP_counter[self.DT_FP_idx[it]] < thresh_cnt:
                self.np_test_fp[it] = "[]"
        
        self.encode_DT_FP = dict()
        tot_DT_FP = 0
        for it in range(N):
            tmp_str = self.np_test_str[it]+self.np_test_fp[it]
            if not tmp_str in self.encode_DT_FP:
                self.encode_DT_FP[tmp_str] = tot_DT_FP
                tot_DT_FP += 1
            self.DT_FP_idx[it] = self.encode_DT_FP[tmp_str]

        self.stat_RC = []
        self.pred_RC = np.zeros(N, dtype=np.int)
        for fp in range(tot_DT_FP):
            tmp_stat_RC = np.zeros(np.int(tot_RC))
            for it in range(N):
                if not self.DT_FP_idx[it] == fp:
                    continue
                tmp_stat_RC[self.RC_list[it]] += 1
            self.stat_RC.append(tmp_stat_RC)
            tmp_pred = np.argmax(tmp_stat_RC)
            for it in range(N):
                if not self.DT_FP_idx[it] == fp:
                    continue
                self.pred_RC[it] = tmp_pred
        self.n_RC = np.zeros(tot_RC)
        self.n_RC_acc = np.zeros(tot_RC)
        self.n_FP = 0
        self.n_NC = 0
        self.n_FN = 0
        self.n_PC = 0
        self.n_E = 0
        
        for it in range(N):
            self.n_RC[self.RC_list[it]] += 1
            if self.RC_list[it] == self.pred_RC[it]:
                self.n_RC_acc[self.RC_list[it]] += 1
            else:
                if self.RC_list[it] > tot_RC-2 and self.pred_RC[it] < tot_RC-1:
                    self.n_FP +=1
                if self.RC_list[it] < tot_RC-1 and self.pred_RC[it] > tot_RC-2:
                    self.n_FN +=1
                if self.RC_list[it] < tot_RC-1 and self.pred_RC[it] < tot_RC-1:
                    self.n_E += 1
                                    
            if self.RC_list[it] > tot_RC-2:
                self.n_NC += 1
            else:
                self.n_PC += 1
        self.RC_acc = self.n_RC_acc / self.n_RC
        self.FPR = np.float(self.n_FP) / self.n_NC
        self.FNR = np.float(self.n_FN) / self.n_PC
        self.DER = np.float(self.n_E) / self.n_PC
        
        
        print(self.RC_acc)
        print(self.n_RC)
        print(self.FPR, self.FNR, self.DER)
        
        self.purity = cluster_purity(self.RC_list, self.DT_FP_idx)
        self.NMI = cluster_NMI(self.RC_list, self.DT_FP_idx)
        self.AMI = sklearn_AMI(self.RC_list, self.DT_FP_idx)
        print(self.purity, self.NMI,  self.AMI)
        
                
        tmp_np = self.np_test_data
        self.mean_all = tmp_np.mean(axis=0)
        self.std_all = tmp_np.std(axis=0)
        for it in range(N):
            tmp_np[it,:] = (tmp_np[it,:]-self.mean_all)/self.std_all
        
        self.silhouette = silhouette_score(tmp_np, self.DT_FP_idx)
        
        self.intra_dist = intra_dist(tmp_np, self.DT_FP_idx)
        
        print(self.silhouette, self.intra_dist)
        
        self.pd_RC = pd.DataFrame(data=RC.DT_FP_idx, columns=['DT_FP'])
        self.pd_RC['RC'] = self.RC_list
        self.pd_RC['pred_RC'] = self.pred_RC
        self.pd_RC.to_csv(outfile)


class Clustering_Analysis(object): 
    
    def __init__(self):
        pass

if __name__ == '__main__':
    RC = RC_Analysis()
    cata_info = []
    RC.training(max_depth=4, file_name='./sp0613/exp2_train.csv', cata_info=cata_info)# raw_DL_BW_100_UL_BW_50 #plan100_50_drop_duplicate_test
    #RC.KMeans_DT_clustering(n_clusters=10, max_depth=4, min_impurity_split=0.05, min_samples_split=5)
    thresh_cnt = 150
    imp_th = 0.03
    RC.DT_Stage_I_Clustering(max_depth=4, min_impurity_split=0.05, min_samples_split=thresh_cnt)
    RC.FP_Stage_II_Clustering(imp_th=imp_th)
    RC.test_DT_FP(file_name='./sp0613/exp2_test.csv', outfile = './exp2_predict.csv', thresh_cnt = thresh_cnt)
    RC.validate_DT_FP(file_name='./sp0613/exp2_test.csv', outfile='./exp2_RC_predict.csv', thresh_cnt = thresh_cnt)
