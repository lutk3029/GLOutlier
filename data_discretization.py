#coding:utf-8
import numpy as np
import pickle
import gc
from sklearn.cluster import KMeans  
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn import mixture
path = './file/'


def update_diff_mean(a):
    return (a[len(a) - 1] - a[0]) / (len(a) - 1)

def get_act_data():
    fr = open(path + 'data_normal_v1_pickle.txt', 'rb')
    dataDict = pickle.load(fr)
    fr.close()
    print(dataDict['values'].shape)
    values = dataDict['values']
    values = values[15000:, :]#去除前面不稳定的部分
    name = dataDict['name']
    del dataDict
    gc.collect()
    sensor_name_list = []
    actuator_name_list = []
    actuator_num_list = []
    len1_name_list = []
    nan_name_list = []
    for i in range(len(name)):
        flag = True
        for j in range(len(values)):
            if np.isnan(values[j, i]):
                nan_name_list.append(name[i])
                flag = False
                break
        if flag:
            leni = len(set(values[:, i]))
            if leni < 10:
                if leni == 1:
                    len1_name_list.append(name[i])#只有一个值的列的集合
                else:
                    actuator_name_list.append(name[i])#值的数量小于10又不是1，将其视为actuator
                    actuator_num_list.append(leni)
            else:
                sensor_name_list.append(name[i])#值大于等于10的视为传感器
    name_dict = {}
    name_dict['nan_name_list'] = nan_name_list
    name_dict['len1_name_list'] = len1_name_list
    name_dict['actuator_name_list'] = actuator_name_list
    name_dict['actuator_num_list'] = actuator_num_list
    name_dict['sensor_name_list'] = sensor_name_list
    print('nan_name_list:', len(nan_name_list), nan_name_list)
    print('len1_name_list:', len(len1_name_list), len1_name_list)
    print('actuator_name_list:', len(actuator_name_list), actuator_name_list)
    print('actuator_num_list:', len(actuator_num_list), actuator_num_list)
    print('sensor_name_list:', len(sensor_name_list), sensor_name_list)
    fw = open(path + 'name_dict_pickle.txt', 'wb')#只考虑name
    pickle.dump(name_dict, fw)
    fw.close()

    # 生成执行器数据
    actuator_data_list_set = []
    temp_and_set = set([])
    for i in range(values.shape[0]):#shape[0]就是shape的第一维，在这里是48000
        actuator_temp = set([])
        for n in actuator_name_list:
            j = name.index(n)
            if n not in actuator_p and n not in actuator_p_backup:
                actuator_temp.add(n + ":" + str(int(values[i][j])))#[P101+P102:6,...,...,]
        for j in range(len(actuator_p)):
            index1 = name.index(actuator_p[j])
            index2 = name.index(actuator_p_backup[j])
            actuator_temp.add(
                actuator_p[j] + '+' + actuator_p_backup[j] + ":" + str(int(values[i][index1]) + int(values[i][index2])))#有后备的加到一起了
        if i == 0:
            temp_and_set = actuator_temp
        else:
            temp_and_set = temp_and_set & actuator_temp#集合取交集，得到一直不变的属性a= (1,2,3),b = (2,3,4), a&b =(2,3)
        actuator_data_list_set.append(actuator_temp)
    print('不变的属性:', temp_and_set)
    for i in range(len(actuator_data_list_set)):
        for ss in temp_and_set:
            actuator_data_list_set[i].remove(ss)#去除不变的属性对应的那一列
    fw = open(path + 'actuator_data_list_set_pickle.txt', 'wb')
    pickle.dump(actuator_data_list_set, fw)
    fw.close()


def get_cluster_center(sensor_value_name,sensor_trend_name):
    fr = open(path + 'data_normal_v1_pickle.txt', 'rb')
    dataDict = pickle.load(fr)
    fr.close()
    print(dataDict['values'].shape)
    values = dataDict['values']
    values = values[15000:, :]
    name = dataDict['name']
    del dataDict
    gc.collect()

    fw = open(path + 'name_dict_pickle.txt', 'rb')
    name_dict = pickle.load(fw)
    fw.close()
    sensor_name_list = name_dict['sensor_name_list']

    sensor_value_num = []
    for nm in sensor_value_name:
        if nm in sensor_name_list:
            j = name.index(nm)
            temp = values[:, j:j+1]
            k = 2#2

            centroids = []
            kmeans = KMeans(n_clusters = k, random_state = 1).fit(temp)
            centroids = kmeans.cluster_centers_
            # result = kMeans(temp, k, 20)
            print(nm, centroids)
            sensor_value_num.append(centroids[:, 0].tolist())
    #sensor_value_num = [[0.00457658, 2.54180708], [0.00466761, 2.44563993], [2.86739004, 19.8190794], [0.04794506, 2.21229701], [1.08295423e-03, 1.61334205]]
    fw = open(path + 'sensor_value_num_pickle.txt', 'wb')#save_value_cluster_center
    pickle.dump(sensor_value_num, fw)
    fw.close()

    sen_dic = {}
    for nm in sensor_trend_name:
        if nm in sensor_name_list:
            index = name.index(nm)
            sen = values[:, index].tolist()
            
            
            min_temp = min(sen)
            max_temp = max(sen)
            for i in range(len(sen)):
                sen[i] = (sen[i] - min_temp) / (max_temp - min_temp)

            win_len = 50
            C = 0.001
            if (nm == 'AIT502') or (nm == 'AIT402'):
                C = 0.0005
            temp1 = []
            for ii in range(0, len(sen)):
                j, num, sum = 0, 0, 0
                while j <= win_len // 2 and ii + j < len(sen):
                    sum = sum + sen[ii + j]
                    num += 1
                    j += 1
                j = 1
                while j <= win_len // 2 and ii - j >= 0:
                    sum = sum + sen[ii - j]
                    num += 1
                    j += 1
                temp1.append(sum / num)
            sen = temp1

            # 分隔
            time = [0]
            istart, iend = 0, 1
            diff_seg_mean = sen[iend] - sen[iend - 1]
            while iend < len(sen):
                # print(iend)
                while iend < len(sen) and abs(sen[iend] - sen[iend - 1] - diff_seg_mean) < C:
                    iend += 1
                    diff_seg_mean = update_diff_mean(sen[istart:iend])
                if iend < len(sen):
                    istart = iend - 1
                    diff_seg_mean = sen[iend] - sen[iend - 1]
                    time.append(istart)
            time.append(len(sen) - 1)
            print(len(time))
            temp_w = []
            for ii in range(1, len(time)):
                temp_w.append((sen[time[ii]] - sen[time[ii - 1]]) / (time[ii] - time[ii - 1]))
            k = 4
            if (nm == 'LIT401') : 
                k = 2
            # if (nm == 'LIT101'):
            #     k = 3
            kmeans = KMeans(n_clusters = k, random_state = 1).fit(np.array(temp_w).reshape(-1, 1))#
            centroids = kmeans.cluster_centers_
            print(nm, centroids)
            sen_dic[nm] = centroids[:, 0].tolist()#save_trend_cluster_center

    fw = open(path + 'sen_dic_pickle.txt', 'wb')
    pickle.dump(sen_dic, fw)
    fw.close()
    return 0

def discretization_training_data(sensor_value_name,sensor_trend_name,actuator_p,actuator_p_backup):
    fr = open(path + 'data_normal_v1_pickle.txt', 'rb')
    dataDict = pickle.load(fr)
    fr.close()
    print(dataDict['values'].shape)
    values = dataDict['values']
    values = values[15000:, :]#去除前面不稳定的部分
    name = dataDict['name']
    del dataDict
    gc.collect()
    
    
    fw = open(path + 'name_dict_pickle.txt', 'rb')
    name_dict = pickle.load(fw)
    fw.close()
    
    sensor_name_list = name_dict['sensor_name_list']
    fr = open(path + 'sensor_value_num_pickle.txt', 'rb')
    sensor_value_num = pickle.load(fr)
    fr.close()

    sensor_value_data_list_set = [set([]) for x in range(values.shape[0])]
    sensor_dict = {}
    for nm in sensor_name_list:
        if nm in sensor_value_name:
            index = sensor_value_name.index(nm)
            sensor_dict[nm] = []
            j = name.index(nm)
            for i in range(len(values)):
                temp_list = []
                for jj in sensor_value_num[index]:
                    temp_list.append(abs(values[i][j] - jj))
                sensor_value_data_list_set[i].add(name[j] + ':' + str(temp_list.index(min(temp_list))))
                sensor_dict[nm].append(temp_list.index(min(temp_list)))
    fw = open(path + 'sensor_value_data_list_set_pickle.txt', 'wb')
    pickle.dump(sensor_value_data_list_set, fw)
    fw.close()
    fw = open(path + 'normal_sensor_cluster_pickle.txt', 'wb')
    pickle.dump(sensor_dict, fw)
    fw.close()


    

    fw = open(path + 'name_dict_pickle.txt', 'rb')
    name_dict = pickle.load(fw)
    fw.close()
    sensor_name_list = name_dict['sensor_name_list']
    fr = open(path + 'sen_dic_pickle.txt', 'rb')
    sen_dic = pickle.load(fr)
    fr.close()

    sensor_trend_data_list_set = [set([]) for x in range(values.shape[0])]
    cluster_dict = {}
    # 生成按趋势聚类的传感器数据
    sen_min_max_dict = {}  # 每个属性的最大最小值
    for nm in sensor_trend_name:
        if nm in sensor_name_list:
            index = name.index(nm)
            sen = values[:, index].tolist()
            cluster_dict[nm] = [-1 for _ in range(len(sen))]
            # 归一化
            min_temp = min(sen)
            max_temp = max(sen)
            # for i in range(len(sen)):
            #     sen[i] = (sen[i] - min_temp) / (max_temp - min_temp)
            sen_min_max_dict[nm] = (min_temp, max_temp)
            #平滑化处理(前后25个的均值)
            win_len = 50
            C = 0.001
            temp1 = []
            for ii in range(0, len(sen)):
                j, num, sum = 0, 0, 0
                while j <= win_len // 2 and ii + j < len(sen):
                    sum = sum + sen[ii + j]
                    num += 1
                    j += 1
                j = 1
                while j <= win_len // 2 and ii - j >= 0:
                    sum = sum + sen[ii - j]
                    num += 1
                    j += 1
                temp1.append(sum / num)
            sen = temp1

            # 分隔
            time = [0]
            istart, iend = 0, 1
            diff_seg_mean = sen[iend] - sen[iend - 1]
            while iend < len(sen):
                # print(iend)
                while iend < len(sen) and abs(sen[iend] - sen[iend - 1] - diff_seg_mean) < C:#要拉开iend与istart间的差距
                    iend += 1
                    diff_seg_mean = update_diff_mean(sen[istart:iend])#(a[len(a) - 1] - a[0]) / (len(a) - 1)#start到end之间的平均差距
                if iend < len(sen):
                    istart = iend - 1#此时iend与istart间的增长率大于平均增长率
                    diff_seg_mean = sen[iend] - sen[iend - 1]
                    time.append(istart)
            time.append(len(sen) - 1)

            temp_w = []
            for ii in range(1, len(time)):
                temp_w.append((sen[time[ii]] - sen[time[ii - 1]]) / (time[ii] - time[ii - 1]))

            y_pred = [0] * len(temp_w)
            for i in range(len(y_pred)):
                minnum, minindex = float('inf'), -1
                for key in range(len(sen_dic[nm])):
                    val = sen_dic[nm][key]
                    if abs(val - temp_w[i]) < minnum:
                        minnum = abs(val - temp_w[i])
                        minindex = key
                y_pred[i] = minindex

            for i in range(len(y_pred)):
                for j in range(time[i], time[i + 1]):
                    sensor_trend_data_list_set[j].add(nm + ':' + str(y_pred[i]))
                    cluster_dict[nm][j] = y_pred[i]
            sensor_trend_data_list_set[time[-1]].add(nm + ':' + str(y_pred[-1]))
            cluster_dict[nm][time[-1]] = y_pred[-1]
    print(len(sensor_trend_data_list_set))

    fw = open(path + 'sensor_trend_data_list_set_pickle.txt', 'wb')
    pickle.dump(sensor_trend_data_list_set, fw)
    fw.close()

    fw = open(path + 'sen_min_max_dict_pickle.txt', 'wb')
    pickle.dump(sen_min_max_dict, fw)
    fw.close()

    fw = open(path + 'normal_cluster_pickle.txt', 'wb')
    pickle.dump(cluster_dict, fw)
    fw.close()

    return 0

def discretization_testing_data(sensor_value_name,sensor_trend_name,actuator_p,actuator_p_backup):
    fr = open(path + 'data_attack_v0_pickle.txt', 'rb')
    dataDict = pickle.load(fr)
    fr.close()
    values = dataDict['values']
    name = dataDict['name']
    label = dataDict['label']
    del dataDict
    gc.collect()
    print(len(name))
    print(len(label))
    print(values.shape)

    fr = open(path + 'name_dict_pickle.txt', 'rb')
    name_dict = pickle.load(fr)
    fr.close()
    fr = open(path + 'sensor_value_num_pickle.txt', 'rb')
    sensor_value_num = pickle.load(fr)
    fr.close()
    # fr = open(path + 'sensor_value_cut_dict_pickle.txt', 'rb')
    # sensor_value_cut_dict = pickle.load(fr)
    # fr.close()
    
    ########################if use EM#########################
    # fw = open(path + 'sensor_data_list_set_test_pickle.txt','rb')
    # sensor_data_list_set_test = pickle.load(fw)
    # fw.close()

    data_list_set = []
    for i in range(values.shape[0]):
        temp_set = set([])
        # 执行器
        for j in range(len(name)):
            if name[j] in name_dict['actuator_name_list']:
                if name[j] not in actuator_p and name[j] not in actuator_p_backup:
                    temp_set.add(name[j] + ':' + str(int(values[i][j])))
        for j in range(len(actuator_p)):
            index1 = name.index(actuator_p[j])
            index2 = name.index(actuator_p_backup[j])
            temp_set.add(actuator_p[j] + '+' + actuator_p_backup[j] + ":" +
                         str(int(values[i][index1]) + int(values[i][index2])))
        # 传感器按值的大小分类
        for j in range(len(sensor_value_name)):
            nm = sensor_value_name[j]
            index = name.index(sensor_value_name[j])
            temp_list = []
            cur_val = values[i][index]
            value_index = -1
        
            for jj in sensor_value_num[j]:
                temp_list.append(abs(values[i][index] - jj))
            temp_set.add(sensor_value_name[j] + ':' + str(temp_list.index(min(temp_list))))
        data_list_set.append(temp_set)

    # 按传感器的趋势分类
    fr = open(path + 'sen_min_max_dict_pickle.txt', 'rb')
    sen_min_max_dict = pickle.load(fr)
    fr.close()
    fr = open(path + 'sen_dic_pickle.txt', 'rb')
    sen_dic = pickle.load(fr)
    fr.close()
    cluster_dict = {}
    for nm in sensor_trend_name:
        # if (nm=='LIT401'):
        #     sen_dic[nm].append(0.0)
        index = name.index(nm)
        C = 0.001
        win_len = 50
        sen = values[:, index].tolist()
        cluster_dict[nm] = [-1 for _ in range(len(sen))]
        # 归一化
        min_temp = sen_min_max_dict[nm][0]
        max_temp = sen_min_max_dict[nm][1]
        

        # for i in range(len(sen)):
        #     sen[i] = (sen[i] - min_temp) / (max_temp - min_temp)

        temp1 = []
        for ii in range(0, len(sen)):
            j, num, sum = 0, 0, 0
            while j <= win_len // 2 and ii + j < len(sen):
                sum = sum + sen[ii + j]
                num += 1
                j += 1
            j = 1
            while j <= win_len // 2 and ii - j >= 0:
                sum = sum + sen[ii - j]
                num += 1
                j += 1
            temp1.append(sum / num)
        sen = temp1

        # 分隔
        time = [0]
        istart, iend = 0, 1
        diff_seg_mean = sen[iend] - sen[iend - 1]
        while iend < len(sen):
            # print(iend)
            while iend < len(sen) and abs(sen[iend] - sen[iend - 1] - diff_seg_mean) < C:
                iend += 1
                diff_seg_mean = update_diff_mean(sen[istart:iend])
            if iend < len(sen):
                istart = iend - 1
                diff_seg_mean = sen[iend] - sen[iend - 1]
                time.append(istart)
        time.append(len(sen) - 1)

        temp_w = []
        for ii in range(1, len(time)):
            temp_w.append((sen[time[ii]] - sen[time[ii - 1]]) / (time[ii] - time[ii - 1]))

        y_pred = [-1] * len(temp_w)
        for i in range(len(temp_w)):
            minnum, minindex = float('inf'), -1
            for key in range(len(sen_dic[nm])):
                val = sen_dic[nm][key]
                if abs(val - temp_w[i]) < minnum:
                    minnum = abs(val - temp_w[i])
                    minindex = key
            y_pred[i] = minindex
        sen_list = values[:, index].tolist()
        for i in range(len(y_pred)):
            for j in range(time[i], time[i + 1]):
                # if (sen_list[j]>max_temp):
                #     data_list_set[j].add(nm + ':' + str(5))
                # elif (sen_list[j]<min_temp):
                #     data_list_set[j].add(nm + ':' + str(6))
                # else:
                data_list_set[j].add(nm + ':' + str(y_pred[i]))
                cluster_dict[nm][j] = y_pred[i]
        data_list_set[time[-1]].add(nm + ':' + str(y_pred[-1]))
        cluster_dict[nm][time[-1]] = y_pred[-1]
    ########################if use EM#########################
    # for i in range(len(data_list_set)):
    #     data_list_set[i] = data_list_set[i] | sensor_data_list_set_test[i]

    fw = open(path + 'data_att_pickle.txt', 'wb')
    pickle.dump(data_list_set, fw)
    fw.close()
    fw = open(path + 'label_pickle.txt', 'wb')
    pickle.dump(label, fw)
    fw.close()
    fw = open(path + 'attack_cluster_pickle.txt', 'wb')
    pickle.dump(cluster_dict, fw)
    fw.close()
    return 0


def data_discretization(sensor_value_name,sensor_trend_name,actuator_p,actuator_p_backup):
    get_cluster_center(sensor_value_name,sensor_trend_name)
    discretization_training_data(sensor_value_name,sensor_trend_name,actuator_p,actuator_p_backup)
    discretization_testing_data(sensor_value_name,sensor_trend_name,actuator_p,actuator_p_backup)
    return 0
if __name__ == '__main__':
    actuator_p = ['P101', 'P201', 'P203', 'P205', 'P302', 'P402', 'P403', 'P501']#列出所有成对的，储存的时候成对的存在一起
    actuator_p_backup = ['P102', 'P202', 'P204', 'P206', 'P301', 'P401', 'P404', 'P502']
    sensor_value_name = ['FIT101', 'FIT201','DPIT301', 'FIT301', 'FIT601']# 'FIT101', 'FIT201','DPIT301', 'FIT301', 'FIT601'
# #sensor_trend_name = ['LIT101', 'LIT301', 'LIT401','AIT203','AIT501','AIT502','AIT402']
    sensor_trend_name = ['LIT101','LIT301', 'LIT401']# 
    # get_act_data()
    # get_cluster_center(sensor_value_name,sensor_trend_name)
    data_discretization(sensor_value_name,sensor_trend_name,actuator_p,actuator_p_backup)