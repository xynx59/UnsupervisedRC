import pandas as pd
import numpy as np
import logging
import time
import copy
import string

class DataSegmentation:
    """ Split data set by specific features """
    def __init__(self, data):
        self.df = pd.DataFrame(data)
    
    def seperate_by_BW(self, file_dir='./sp0613/', feature_BW=['DL_BW', 'UL_BW']):
        df_BW = self.df[feature_BW].drop_duplicates()
        Col = list(df_BW.columns.values)
        idx_BW = []
        for feature in feature_BW:
            idx_BW.append(Col.index(feature))
        np_BW = df_BW.values
        M_BW = df_BW.shape[0]
        (N, M) = df_BW.shape
        for m in range(M_BW):
            tmp_file = file_dir+'raw_'+feature_BW[0]+'_'+str(int(np_BW[m,0]))\
                +'_'+feature_BW[1]+'_'+str(int(np_BW[m,1]))+'.csv'
            tmp_df = self.df[(self.df[feature_BW[0]] == np_BW[m,0]) & \
                             (self.df[feature_BW[1]] == np_BW[m,1])]
            tmp_df.reset_index(drop=True, inplace=True)
            tmp_df.to_csv(tmp_file)
    
    def seperate_by_feature(self, file_dir='./sp0613/', feature='DL_BW', DL_bins = [-1, 10, 30, 50, 100, 200, 10000]):
        M_bins = len(DL_bins)
        for m in range(M_bins-1):
            tmp_file = file_dir+'raw_'+feature+'_'+str(DL_bins[m]+1)+'_to_'+str(DL_bins[m+1])+'.csv'
            tmp_df = self.df[(self.df[feature] <= DL_bins[m+1]) & (self.df[feature] > DL_bins[m])]
            tmp_df.reset_index(drop=True, inplace=True)
            tmp_df.to_csv(tmp_file)


class SpeedTestDataTransform:
    """Ad hoc transform, transform in place"""
    """must have col: ACCOUNT, DATE_TIME, DL_RESULT_RATIO,"""
    def __init__(self, data):
        self.df = pd.DataFrame(data)
        self.binning_ref = dict()

    def initial_cleanup(self, dashboard=False):
        """basic clean up"""
        # drop columns if all the values are nan
        self.df.dropna(axis=1, how='all', inplace=True)
        # convert all columns name to upper letter
        self.df.rename(str.upper, axis='columns', inplace=True)
        # drop useless columns
        dash_use = ['DOWNLOAD_RATE_KBPS', 'DL_BW', 'UL_BW']
        discard = ['DOWNLOAD_TRAFFIC_KB', 'DOWNLOAD_DURATION', 'DOWN_BANDWIDTH',
                   'UPLOAD_TRAFFIC_KB', 'UPLOAD_DURATION', 'UPLOAD_RATE_KBPS', 'UP_BANDWIDTH',
                   'UL_RESULT_RATIO',
                   'TIMESTAMP_STR', 'REMOTE_IP_DECIMAL', 'EXPORT_TIME', 'NAT64_IP', 'NAS_PORT', 'ASN', 'RESULT'
                   'IP_TYPE2', 'IP_LOCATION', 'IP_LOCATION_DETAIL', 'ASN_LOCATION', 'ASN_IP_PREFIX', 'MAC', 'PACKAGE',
                   'GPU_INFO', 'DNS_PCT']
        if not dashboard:
            discard = discard + dash_use
        checked_discard = [i for i in discard if i in self.df.columns]
        logging.debug(f'drop total of {len(checked_discard)} cols: {checked_discard}')
        self.df.drop(columns=checked_discard, inplace=True)
        # drop rows DL_RESULT_RATIO
        if 'DL_RESULT_RATIO' in self.df.columns.values:
            self.df = self.df[self.df['DL_RESULT_RATIO'].notnull()]
            
    def remove_sparse_col(self, sparse_th=0.2):
        Col = self.df.columns.values
        drop_Col = []
        N = self.df.shape[0]
        for col in Col:
            tmp_th = 1.0 - np.float(self.df[col].isna().sum()) / N
            if tmp_th < sparse_th:
                drop_Col.append(col)
        self.df.drop(drop_Col, axis=1, inplace=True)

    def remove_missing(self):
        self.df.dropna(axis=0, how='any', inplace=True)
        
    def keep_based_on_columns(self, Col=["UP_DATA_RTT_MS", "PKG_DL_BW", "PKG_UL_BW"]):
        self.df.dropna(subset=Col, inplace=True)
        self.df.reset_index(inplace=True, drop=True)

    def filter_machine_test(self, thresh=20):
        if 'ACCOUNT' in self.df.columns:
            account_size = self.df.groupby(['ACCOUNT']).size()
            user_account = list(account_size[account_size < thresh].index)
            # filter out account has been tested for more than thresh times
            self.df = self.df[self.df['ACCOUNT'].isin(user_account)]
        else:
            logging.info('No ACCOUNT info in current data set')

    def convert_result(self, col='DL_RESULT_RATIO', p_th=0.5):
        """DL_RESULT_RATIO represent speedtest results, if <0.5 fail, >0.8 pass, in between is marginal
           convert values to label, 3 classes, and rename the column
        """
        self.df[col] = self.df['DL_RESULT_RATIO'].apply(lambda x: 'Fail' if x < p_th else 'Pass')
        self.df.rename(columns={col: "RESULT"}, inplace=True)
        return self.df

    def convert_uptime(self, col='BOOT_TIME', test_time_col='DATE_TIME', rename_col='END_DEVICE_UPTIME_HR', date_format=None, check_format=True):
        """ calculate the difference between test timestamp and boot timestamp
            better set datetime format to be fast
        """
        if col in self.df.columns and test_time_col in self.df.columns:
            boot_time = self.df[col].values
            test_time = self.df[test_time_col].values
            if date_format is None:  # use generic way, 3X slower
                up_time = [round((pd.to_datetime(a).value / 1e9 - b) / 3600, 1) for a, b in zip(test_time, boot_time)]
            else:  # given specific format, fast
                if check_format:
                    N = len(test_time)
                    for it in range(N):
                        if '/' in test_time[it]:
                            tmp_str = copy.deepcopy(test_time[it])
                            b1 = tmp_str.find('/')
                            tmp_str = tmp_str[b1+1:]
                            b2 = tmp_str.find('/')
                            tmp_str = tmp_str[b2+1:]
                            b3 = tmp_str.find(':')
                            tmp_str = tmp_str[b3+1:]
                            b4 = tmp_str.find(' ')
                            b2 += b1+1
                            b3 += b2+1
                            b4 += b3+1
                            str_y = test_time[it][0:b1]
                            str_m = test_time[it][b1+1:b2]
                            str_d = test_time[it][b2+1:b3]
                            str_h = test_time[it][b3+1:b4]
                            str_min = test_time[it][b4+1:]
                            if len(str_m) == 1:
                                str_m = '0'+str_m
                            if len(str_d) == 1:
                                str_d = '0'+str_d
                            if len(str_h) == 1:
                                str_d = '0'+str_h
                            if len(str_min) == 1:
                                str_d = '0'+str_min
                            test_time[it] = str_y+'-'+str_m+'-'+str_d+' '+str_h+':'+str_min+':00'
                    
                up_time = [round((time.mktime(time.strptime(a, date_format)) - b)/3600, 1)
                           for a, b in zip(test_time, boot_time)]
            N = len(up_time)
            for it in range(N):
                up_time[it] = np.max([up_time[it], 0])
            self.df.drop(columns=[col], inplace=True)
            self.df[rename_col] = up_time
        else:
            logging.info(f'Cannot calculation device uptime, either {col} or {test_time_col} does not exist')

    def convert_float2cat(self, skip_col=('ACCOUNT', 'DATE_TIME', 'UL_BW', 'DL_BW'), equi_label=False, B=None):
        for col in self.df.columns:
            if col in skip_col:
                continue
            tmp_idx = self.df[col].first_valid_index()
            tmp_val = self.df[col][tmp_idx]
            if isinstance(tmp_val, (int, np.int, np.int8, np.int32, np.int64, float, np.float, np.float32, np.float64)):
                #print(col)
                if not equi_label:
                    self.df[col] = self._four_bin_conversion(self.df[col].values, col_name=col)
                    if self.df.iloc[0][col] == 'Stable':
                            self.df.drop(columns=col, inplace=True)
                            logging.info(f'Drop {col}, values have small variation, consider it as stable')
                else:
                    self.df[col] = self._equi_depth_conversion(self.df[col].values, col_name=col, B=B)

    def update_port(self):
        """Attach the device info"""
        """the following syntax has been optimized for speed
           use df.apply(lambda x: x.a + x.b, axis = 1) kind of style is slow, since it takes the whole df
        """
        self.df['PORT'] = list(map('__'.join, zip(self.df['DEVICE'].values, self.df['PORT'].values)))
        return self.df

    def update_remote_ip(self):
        return self.df

    def _equi_depth_conversion(self, data, min_zero=True, col_name='KPI', B=4):
        """auto binning, default is four bin, if min max not not much of difference,"""
        # trick "i is np.nan" doesn't work, has to use np.isnan() function
        num = data[~np.isnan(data)]  # remove nan for split value calculation
        # num = [i for i in data if ~np.isnan(i)]  # alternative way is slower
        if np.abs(np.std(num)/np.mean(num)) < 0.0:  # small variation don't bother
            return ['Stable'] * len(data)
        # p1 could either be 0 or the first 20 percentile
        min_value = max(min(num), -1e10)
        v = np.zeros(B+1)
        for it in range(B):
            v[it] = np.percentile(num, 100.0/B*it)
        v_uni, idx_uni = np.unqiue(v, return_index=True)
        q = np.linspace(0.0, 1.0, B+1)
        q_uni = q[idx_uni]
        B_uni = len(q_uni)-1
        ret = pd.qcut(data, q=q_uni, labels=range(B_uni).astype(np.float))
        return ret
        #self.binning_ref[col_name] = {'Min': p1, 'Low': p2, 'Medium': p3, 'Max': max_value}
        

    def _four_bin_conversion(self, data, min_zero=True, col_name='KPI'):
        """auto binning, default is four bin, if min max not not much of difference,"""
        # trick "i is np.nan" doesn't work, has to use np.isnan() function
        num = data[~np.isnan(data)]  # remove nan for split value calculation
        # num = [i for i in data if ~np.isnan(i)]  # alternative way is slower
        if np.abs(np.std(num)/np.mean(num)) < 0.0:  # small variation don't bother
            return ['Stable'] * len(data)
        # p1 could either be 0 or the first 20 percentile
        min_value = max(min(num), -1e10)
        p1 = np.percentile(num, 20)# if not min_zero or min_value > 0 else 0
        p2 = np.percentile(num, 50)
        p3 = np.percentile(num, 90)
        max_value = max(num)
        ret = [i if np.isnan(i) else 'Min' if i <= p1 else 'Low' if i <= p2 else 'Medium' if i <= p3 else 'Max'
               for i in data]
        self.binning_ref[col_name] = {'Min': p1, 'Low': p2, 'Medium': p3, 'Max': max_value}
        return ret
    
    def deep_cleanup(self, precise = False, date_format=None):
        """ For data with the same account, remove data with the same/similar features """
        self.df.sort_values(by=['ACCOUNT'], inplace = True)
        self.df.reset_index(drop = True, inplace = True)
        Col = self.df.columns.values
        np_data = self.df.values
        (N, M) = np_data.shape
        Idx_list = -np.ones((N), dtype = np.int)
        tot_Idx = 0
        Acc = np.nan
        Res = np.nan
        Dat = np.nan
        Ser = np.nan
        for it in range(M):
            if Col[it] == 'ACCOUNT':
                Acc = it
            if Col[it] == 'RESULT':
                Res = it
            if Col[it] == 'DATE_TIME':
                Dat = it
            if Col[it] == 'DATE_TIME':
                Dat = it
            if Col[it] == 'SERVER_URL':
                Ser = it
            
        self.Acc = Acc
        self.Res = Res
        self.Dat = Dat
        self.Ser = Ser
        
        idx_start = 0
        for it in range(1,N):
            idx_end = it
            if np_data[it, Acc] == np_data[it-1, Acc] and it < N-1:
                continue
            """ Maintain unique data """
            if idx_end == idx_start+1:
                Idx_list[tot_Idx] = idx_start
                tot_Idx += 1
                idx_start = it
                continue
            if precise:
                """ For each account clustering data by DATE_TIME, and maintain unique data """
                df_tmp = copy.deepcopy(self.df.iloc[idx_start:idx_end, :])
                df_tmp['Idx'] = range(idx_start, idx_end)
                df_tmp.sort_values(by=['DATE_TIME'], inplace = True)
                df_tmp.reset_index(drop = True, inplace = True)
                np_tmp = df_tmp.values
                (tmpN, tmpM) = df_tmp.shape
                if date_format == None:
                    lasttime = pd.to_datetime(np_tmp[0,Dat]).value * 1e-9
                else:
                    lasttime = time.mktime(time.strptime(np_tmp[0,Dat], date_format))
                chck = dict()
                Idx_list[tot_Idx] = np_tmp[0,-1]
                tot_Idx += 1
                
                chck[tuple(np_tmp[0,[Res,Ser]])] = True
                for jt in range(1, tmpN):
                    if date_format == None:
                        tmptime = pd.to_datetime(np_tmp[jt,Dat]).value * 1e-9
                    else:
                        tmptime = time.mktime(time.strptime(np_tmp[jt,Dat], date_format))
                    if (tmptime-lasttime) < 3600*3:
                        if not (tuple(np_tmp[jt,[Res,Ser]]) in chck):
                           Idx_list[tot_Idx] = np_tmp[jt,-1]
                           tot_Idx += 1
                           chck[tuple(np_tmp[jt,[Res,Ser]])] = True
                    else:
                        chck = dict()
                        Idx_list[tot_Idx] = np_tmp[jt,-1]
                        tot_Idx += 1
                        chck[tuple(np_tmp[jt,[Res,Ser]])] = True
            else:
                """ For each account, and maintain one data for each type of result """
                chck = {'Pass':False, 'Fail':False, 'Marginal':False}
                for jt in range(idx_start, idx_end):
                    if not (tuple(np_data[jt,[Res,Ser]]) in chck):
                        Idx_list[tot_Idx] = jt
                        tot_Idx += 1
                        chck[tuple(np_data[jt,[Res,Ser]])] = True
                
            idx_start = it
        
        print(N)
        print(tot_Idx)
        self.df = self.df.iloc[Idx_list[:tot_Idx],:].reset_index(drop=True)
        
        
class DataDeepTransform:
    def __init__(self, data):
        self.df = pd.DataFrame(data)
    
    def remote_ip_sperate(self, remote_ip = '49.231.68.150'):
        """ Select data for specific remote ip """
        self.df = self.df[self.df['REMOTE_IP']==remote_ip].reset_index(drop = True)
    
    def add_cnt(self, features = ["DEVICE", "PORT", "REMOTE_IP", "SERVER_URL"]):
        """ Add columns to count the # of cases share the same feature """
        (N, M) = self.df.shape
        for feature in features:
            tmp_dict = dict(self.df[feature].value_counts())
            tmp_np = self.df[feature].values
            tmp_cnt = np.zeros((N), np.int)
            for it in range(N):
                if type(tmp_np[it]) == str:
                    tmp_cnt[it] = tmp_dict[tmp_np[it]]
                else:
                    tmp_cnt[it] = 0
            self.df['N_'+feature] = tmp_cnt
    
    def label_nan_cata(self, features = ["REMOTE_IP", "DEVICE_TYPE", "WAN_TYPE_DYNAMIC", "DEVICE", "PORT", \
                                         "APPLICATION_PROTOCOL", "CC", "CITY", \
                                         "SPONSOR", "SERVER_URL", "EDSG_VALID", \
                                         "CPE_VENDOR", "CPE_MODEL", "CPE_TYPE", "DNS_OWNER"]):
        (N, M) = self.df.shape
        for feature in features:
            tmp_np = self.df[feature].values
            for it in range(N):
                if not (type(tmp_np[it]) == str):
                    tmp_np[it] = 'Unknown'
            self.df[feature] = tmp_np
    
    def remove_minor_cata(self, nth = 10, cata_set_list=None, features = ["REMOTE_IP", "DEVICE", "PORT", \
                                                      "CC", "CITY", \
                                                      "SPONSOR", "SERVER_URL", \
                                                      "CPE_MODEL"]):
        (N, M) = self.df.shape
        if not cata_set_list == None:
            mt = 0
            for feature in features:
                tmp_np = self.df[feature].values
                for it in range(N):
                    if type(tmp_np[it])==str:
                        if not (tmp_np[it] in cata_set_list[mt]):
                            tmp_np[it] = 'Others'
                self.df[feature] = tmp_np
                mt += 1
        else:    
            cata_set_list = []
            for feature in features:
                tmp_set = set()
                tmp_dict = dict(self.df[feature].value_counts())
                tmp_np = self.df[feature].values
                for it in range(N):
                    if type(tmp_np[it])==str:
                        if (tmp_dict[tmp_np[it]] <= nth):
                            tmp_np[it] = 'Others'
                        else:
                            tmp_set.add(tmp_np[it])
                self.df[feature] = tmp_np
                cata_set_list.append(tmp_set)
        return cata_set_list

def numerization_part(pd_data = None):
    
    (N, M) = pd_data.shape
    Cols = pd_data.columns.values
    N_bounds = 0
    for col in Cols:
        tmp_series = pd_data[col].unique()
        if type(tmp_series[0]) != str and np.isnan(tmp_series[0]):
            tmp_series = tmp_series[1:]
        if tmp_series.shape[0] > N_bounds:
            N_bounds = tmp_series.shape[0]
    #print(N_bounds)
    pd_bounds = pd.DataFrame(data = np.zeros((N_bounds, M)), columns = Cols).astype(str)
    np_numer = np.zeros((N, M))
    
    jt = 0
    cata_name = []
    for col in Cols:
        tmp_data = pd_data[col].values
        tmp_idx = pd_data[col].first_valid_index()
        tmp_val = pd_data[col][tmp_idx]
        if isinstance(tmp_val, (int, np.int, np.int8, np.int32, np.int64, float, np.float, np.float32, np.float64)):
            for it in range(N):
                np_numer[it][jt] = tmp_data[it]
        else:
            if col == 'RESULT':
                tmp_series = np.array(['Fail', 'Pass'])
            else:
                cata_name.append(col)
                tmp_series = pd_data[col].unique()
                if type(tmp_series[0]) != str and np.isnan(tmp_series[0]):
                    tmp_series = tmp_series[1:]
            #print(col, tmp_series)
            #tmp_series.sort()
            tmp_tot = tmp_series.shape[0]
            tmp_dict = dict()
            for it in range(tmp_tot):
                tmp_dict[tmp_series[it]] = it
                pd_bounds[col][it] = tmp_series[it]
            for it in range(N):
                tmp_xx = tmp_data[it]
                if type(tmp_xx) != str and np.isnan(tmp_xx):
                    np_numer[it][jt] = np.nan
                else:
                    np_numer[it][jt] = tmp_dict[tmp_xx]
        jt += 1
    pd_numer = pd.DataFrame(data = np_numer, columns = Cols)
    return pd_numer, pd_bounds, cata_name

def numerization_test(pd_data = None, pd_bounds = None):
    
    (N, M) = pd_data.shape
    Cols = pd_data.columns.values
    np_numer = np.zeros((N, M))
    
    jt = 0
    for col in Cols:
        tmp_data = pd_data[col].values
        tmp_idx = pd_data[col].first_valid_index()
        print(tmp_idx)
        tmp_val = pd_data[col][tmp_idx]
        if isinstance(tmp_val, (int, np.int, np.int8, np.int32, np.int64, float, np.float, np.float32, np.float64)):
            for it in range(N):
                np_numer[it][jt] = tmp_data[it]
        else:
            if col == 'RESULT':
                tmp_series = np.array(['Fail', 'Pass'])
            else:
                tmp_series = pd_data[col].unique()
                if type(tmp_series[0]) != str and np.isnan(tmp_series[0]):
                    tmp_series = tmp_series[1:]
            #tmp_series.sort()
            tmp_tot = tmp_series.shape[0]
            tmp_dict = dict()
            for it in range(tmp_tot):
                tmp_dict[tmp_series[it]] = it
                pd_bounds[col][it] = tmp_series[it]
            for it in range(N):
                tmp_xx = tmp_data[it]
                if type(tmp_xx) != str and np.isnan(tmp_xx):
                    np_numer[it][jt] = np.nan
                elif not tmp_xx in tmp_dict:
                    np_numer[it][jt] = np.nan
                else:
                    np_numer[it][jt] = tmp_dict[tmp_xx]
        jt += 1
    pd_numer = pd.DataFrame(data = np_numer, columns = Cols)
    return pd_numer


class DataNormalization:
    def __init__(self, data):
        self.df = pd.DataFrame(data)

    def standard(self):
        df1 = self.df
        return df1

    def min_max(self):
        df1 = self.df
        return df1


def example_load_clean_data(csv_file='../../uploads/demo.csv', output_file='./cleanup.csv'):
    T0 = time.time()
    df = pd.read_csv(csv_file, index_col=[0])
    T1 = time.time()
    print("Reading time:", T1-T0)
    obj = SpeedTestDataTransform(df)
    T2 = time.time()
    print("Initial time:", T2-T1)
    obj.initial_cleanup()
    T3 = time.time()
    print("Clean time:", T3-T2)
    obj.filter_machine_test(thresh=20)
    T4 = time.time()
    print("Filter time:", T4-T3)
    obj.convert_result()
    T5 = time.time()
    print("Convert time:", T5-T4)
    obj.convert_uptime(date_format="%Y-%m-%d %H:%M:%S")
    T6 = time.time()
    print("Convert uptime time:", T6-T5)
    obj.update_port()
    T7 = time.time()
    print("Update port time:", T7-T6)
    #obj.convert_float2cat()  # only add this process when needed, continuous data lost resolution through binning
    T8 = time.time()
    print("Float2cat time:", T8-T7)
    obj.deep_cleanup(precise = False, date_format="%Y-%m-%d %H:%M:%S")
    T9 = time.time()
    print("Deep cleanup time:", T9-T8)
    obj.df.drop(columns=['ACCOUNT', 'DATE_TIME'], inplace=True)  # finally drop ID and TIME info
    T10 = time.time()
    print("Drop time:", T10-T9)
    obj.df.to_csv(output_file)
    pd.DataFrame.from_dict(obj.binning_ref, orient='index').to_csv('./binning_ref.csv')
    T11 = time.time()
    print("Save file time:", T11-T10)
    return obj.df

def example_deep_transform_data(csv_file = './cleanup_19148.csv', output_file='./deep.csv'):
    df = pd.read_csv(csv_file, index_col=[0])
    obj = DataDeepTransform(df)
    #obj.add_cnt()
    obj.label_nan_cata()
    obj.remove_minor_cata(nth = 10)
    obj.df.to_csv(output_file)
    return obj.df

def example_sep_data(csv_file='./sp0613/raw.csv'):
    df = pd.read_csv(csv_file)
    ds=DataSegmentation(df)
    ds.seperate_by_feature()
    ds.seperate_by_BW()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    #file = './demo.csv'
    #df = example_load_clean_data(file)
    
    #file = './cleanup_19148.csv'
#    df = example_deep_transform_data(file)    
#    pd_numer, pd_bounds = numerization_part(df)
#    pd_numer.to_csv('./numer_19148.csv',index = False)
#    pd_bounds.to_csv('./bounds_19148.csv')
    
    file = './sp0613/raw.csv'
    #example_sep_data(file)
    example_load_clean_data(csv_file='./sp0613/raw_DL_BW_100_UL_BW_50.csv', output_file='./sp0613/cleanup_100_50.csv')
    example_deep_transform_data(csv_file='./sp0613/cleanup_100_50.csv', output_file='./sp0613/deep_100_50.csv')
    





