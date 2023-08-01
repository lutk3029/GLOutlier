import pickle
import csv
import pandas as pd
import gc
import os
import shutil

path = './file/'
def get_tid_belong(ii,data):#获得该属性出现的区间集合
    l = r = -1
    n = len(data)
    res = []
    flag = 0
    for i in range(n):
        temp = data[i].split(' ')
        if (str(ii) in temp):
            if (flag):
                r = i
            else:
                l = r = i
            flag = 1
        else:
            if (flag):
                res.append([l,r])
                flag = 0
    return res


def get_sup():
    fr = open(path + 'actuator_data_list_set_pickle.txt', 'rb')
    actuator_data_list_set = pickle.load(fr)
    fr.close()
    fr = open(path + 'sensor_value_data_list_set_pickle.txt', 'rb')
    sensor_value_data_list_set = pickle.load(fr)
    fr.close()
    fr = open(path + 'sensor_trend_data_list_set_pickle.txt', 'rb')
    sensor_trend_data_list_set = pickle.load(fr)
    fr.close()
    fr = open(path + 'data_all_list_pickle.txt', 'rb')
    data_all_list = pickle.load(fr)
    fr.close()
    data_all_set = set()
    dataList = []
    for i in range(len(actuator_data_list_set)):# dataList是把分开处理的各列数据整合到一起,data_all_set是所有可能取值的集合
        data_all_set = data_all_set | sensor_trend_data_list_set[i] | actuator_data_list_set[i] | \
                       sensor_value_data_list_set[i]
        dataList.append(list(sensor_trend_data_list_set[i] | actuator_data_list_set[i] | sensor_value_data_list_set[i]))
    all_data = [b for a in dataList for b in a]
    sup_dict = {}
    # for i in range(len(actuator_data_list_set)):# dataList是把分开处理的各列数据整合到一起,data_all_set是所有可能取值的集合
    #     data_all_set = data_all_set | sensor_data_list_set[i] | actuator_data_list_set[i]
    #     dataList.append(list(sensor_data_list_set[i] | actuator_data_list_set[i]))
    # l = len(dataList)
    # dataList = dataList[:int(l * data_len)]#使用多少数据量
    name_list = []
    #data_all_list = sorted(list(data_all_set))
    for item in data_all_list:
        name = item.split(':')[0]
        name_list.append(name)
    print(list(set(name_list)))
    name_dict = {}
    for name in list(set(name_list)):
        name_dict[name] = name_list.count(name)

    for item in data_all_list:
        sup_dict[item] = (all_data.count(item))/len(actuator_data_list_set)
    fw = open(path + 'Sup_single_dict_pickle.txt', 'wb')
    pickle.dump(sup_dict, fw)
    fw.close()
    fw = open(path + 'name_dict_pickle.txt', 'wb')
    pickle.dump(name_dict, fw)
    fw.close()
    print(name_dict)
    print(sup_dict)
    return sup_dict,dataList

def get_global_and_local_dict(LF,sup_dict):
    global_dict = {}
    local_dict = {}
    for item,sup in sup_dict.items():
        if (sup>LF and sup<0.99):
            item_name,index = item.split(':')[0],int(item.split(':')[1])
            if (item_name not in list(global_dict.keys())):
                
                global_dict[item_name] = [index]
            else:
                global_dict[item_name].append(index)
    for item,sup in sup_dict.items():
        if (sup<=LF and (item.split(':')[0]) in global_dict.keys()):
            item_name,index = item.split(':')[0],int(item.split(':')[1])
            if(item_name not in list(local_dict.keys())):
                
                local_dict[item_name] = [index]
            else:
                local_dict[item_name].append(index)

    # print(global_dict)
    # print(local_dict)
    return global_dict,local_dict


def partition_training_data(LF,global_dict,local_dict,dataList):   
    p ='./file/'
    fr = open(p + 'data_all_list_pickle.txt', 'rb')
    data_all_list = pickle.load(fr)
    fr.close()
    fr = open(p + 'data_num' + '1.0' + '.txt')
    data = fr.readlines()
    fr.close()

    fr = open(p + 'data_normal_v1_pickle.txt', 'rb')
    dataDict = pickle.load(fr)
    fr.close()
    print(dataDict['values'].shape)
    values = dataDict['values']
    values = values[15000:, :]
    n = values.shape[0]
    name = dataDict['name']
    del dataDict
    gc.collect()
    # train_data = pd.read_csv('./train.csv',index_col = 0)
    print(name)
    print(data_all_list)
    # item_dict = {'P101+P102':3,'MV201':2,'FIT101':0,'FIT301':0,'MV303':1,'LIT401':1,'MV302':2,'P302+P301':3,'FIT201':0,'P203+P204':3,'MV301':1,'MV101':2,'MV304':1,'DPIT301':0}
    #{'P205+P206':2, 'P101+P102':2, 'P203+P204':2, 'LIT101':3, 'FIT201':1, 'MV201':1, 'LIT301':0, 'MV101':1, 'FIT101':1, 'LIT101':1, 'LIT101':0,'P302+P301':2, 'DPIT301':1, 'FIT301':0,'MV302':1, 'LIT401':0, 'LIT301':1}

    print(1)
#     print(item_dict)
    m = len(global_dict.keys())
    a = list(set(list(global_dict.keys())))
    #a.sort()
    b = list(set(list(local_dict.keys())))
    #b.sort()
    print('a',a)
    print('b',b)
    print('a==b?',a==b)
    value_list1 = [] #无缺失版
    value_list2 = []
    for i in range(len(data)):
        temp = data[i].split()
        t = [-1 for _ in range(len(a))]#a
        t1 = [-1 for _ in range(len(b))]
        for ii in temp:
            temp_name = data_all_list[int(ii)]
            n_l = temp_name.split(':')
            #n_l = temp.split(':')
            na = n_l[0]
            vall = n_l[1]
            #val = n_l[1]
            
            if ('+' in list(na)):
                val = n_l[1]
            else:
                index = name.index(na)
                val = vall#values[i][index]
            
#             print(na,val)
            if (na in a):
                if (int(vall) in global_dict[na]):
                    t[a.index(na)]= val
                else:
                    t[a.index(na)]= -1
            if((na in b)):
                if (int(vall) in local_dict[na]):
                    t1[a.index(na)]= val
                else:
                    t1[a.index(na)]= -1

#             elif((na in cha_list) and (na not in a)):#删掉
#                 t[cha_list.index(na)]=int(val)
        value_list1.append(t)
        value_list2.append(t1)


    file_path=os.path.abspath(r"./")
    file_name=file_path + "\\"+'SWaT_global_'+str(LF)
    if (not os.path.exists(file_name)):
        os.makedirs(file_name)
        
    with open('./SWaT_global_'+str(LF)+'/train'+'.csv','w+',newline='') as f1:
        writer = csv.writer(f1)
        row1 = a
        row1.insert(0,0)

        writer.writerow(row1)

        for i in range(len(value_list1)):
            row = value_list1[i]
            row.insert(0,i)
            writer.writerow(row)
        f1.close()
    file_name=file_path + "\\"+'SWaT_local_'+str(LF)
    if (not os.path.exists(file_name)):
        os.makedirs(file_name)
    with open('./SWaT_local_'+str(LF)+'/train'+'.csv','w+',newline='') as f1:
        writer = csv.writer(f1)
        row1 = b
        row1.insert(0,0)

        writer.writerow(row1)

        for i in range(len(value_list2)):
            row = value_list2[i]
            row.insert(0,i)
            writer.writerow(row)
        f1.close()
    return 0

def partition_testing_data(LF,global_dict,local_dict):
    p ='./file/'
    fr = open(p + 'data_all_list_pickle.txt', 'rb')
    data_all_list = pickle.load(fr)
    fr.close()
    #生成自己的divide_dict
    # fw = open(p + 'Sup_single_dict_pickle.txt', 'rb')
    # sup_dict = pickle.load(fw)
    # fw.close()
    fr = open(p + 'data_att_pickle.txt', 'rb')#测试数据
    all_data_list_set = pickle.load(fr)
    fr.close()
    # divide_dict = {}
    # for i in data_all_list:
    #     divide_dict[i] = get_tid_belong(i,all_data_list_set)

    fr = open(p + 'data_attack_v0_pickle.txt', 'rb')
    dataDict = pickle.load(fr)
    fr.close()
    print(dataDict['values'].shape)
    values = dataDict['values']
    n = values.shape[0]
    name = dataDict['name']
    del dataDict
    gc.collect()

    print(name)
    print(data_all_list)
    # item_dict = {'P101+P102':3,'MV201':2,'FIT101':0,'FIT301':0,'MV303':1,'LIT401':1,'MV302':2,'P302+P301':3,'FIT201':0,'P203+P204':3,'MV301':1,'MV101':2,'MV304':1,'DPIT301':0}
    #{'P205+P206':2, 'P101+P102':2, 'P203+P204':2, 'LIT101':3, 'FIT201':1, 'MV201':1, 'LIT301':0, 'MV101':1, 'FIT101':1, 'LIT101':1, 'LIT101':0,'P302+P301':2, 'DPIT301':1, 'FIT301':0,'MV302':1, 'LIT401':0, 'LIT301':1}



    a = list(set(list(global_dict.keys())))
    b = list(set(list(local_dict.keys())))
    a.sort()
    b.sort()
    value_list1 = [] #无缺失版
    value_list2 = []
    m = len(global_dict.keys())
    cha_list = ['LIT101', 'P101+P102', 'MV201', 'FIT101', 'FIT301', 'MV303', 'LIT301', 'LIT401', 'MV302', 'P302+P301', 'FIT201', 'P203+P204', 'MV301', 'MV101', 'MV304', 'DPIT301', 'P205+P206']
# print(1)
    for i in range(len(all_data_list_set)):
        temp = all_data_list_set[i]
        t = [-1 for _ in range(len(a))]
        t1 = [-1 for _ in range(len(b))]
        for ii in temp:
            n_l = ii.split(':')
            na = n_l[0]
            vall = n_l[1]
            #val = n_l[1]
            
            if ('+' in list(na)):
                val = n_l[1]
            else:
                index = name.index(na)
                val = values[i][index]
            if (na in a):
                if (int(vall) in global_dict[na]):
                    t[a.index(na)]= -1
                else:
                    t[a.index(na)]= vall
            if((na in b)):
                if (int(vall) in local_dict[na]):
                    t1[a.index(na)]= vall
                else:
                    t1[a.index(na)]= -1
#             elif((na in cha_list) and (na not in a)):#删掉
#                 t[cha_list.index(na)]=int(val)
    #             t[a.index(na)]=int(val)
        value_list1.append(t)
        value_list2.append(t1)

    file_path=os.path.abspath(r"./")
    file_name=file_path + "\\"+'SWaT_local_'+str(LF)
    if (not os.path.exists(file_name)):
        os.makedirs(file_name)

    with open('./SWaT_global_'+str(LF)+'/test'+'.csv','w+',newline='') as f1:
        writer = csv.writer(f1)
        row1 = a
        row1.insert(0,0)

        writer.writerow(row1)

        for i in range(len(value_list1)):
            row = value_list1[i]
            row.insert(0,i)
            writer.writerow(row)
        f1.close()
    file_name=file_path + "\\"+'SWaT_local_'+str(LF)
    if (not os.path.exists(file_name)):
        os.makedirs(file_name)
    with open('./SWaT_local_'+str(LF)+'/test'+'.csv','w+',newline='') as f1:
        writer = csv.writer(f1)
        row1 = b
        row1.insert(0,0)

        writer.writerow(row1)

        for i in range(len(value_list2)):
            row = value_list2[i]
            row.insert(0,i)
            writer.writerow(row)
        f1.close()
    return 0

def data_partition(LF):
    sup_dict,dataList = get_sup()
    global_dict,local_dict = get_global_and_local_dict(LF,sup_dict)
    partition_training_data(LF,global_dict,local_dict,dataList)
    partition_testing_data(LF,global_dict,local_dict)#,sup_dict
    shutil.copy('./file/test_label.csv', './SWaT_global_'+str(LF)+'/test_label.csv')

if __name__ == '__main__':
    #get_sup()
    LF = 0.5
    sup_dict,dataList = get_sup()
    print(len(sup_dict.keys()))
    global_dict,local_dict = get_global_and_local_dict(LF,sup_dict)
    print(global_dict)
    print(len(global_dict.keys()))
    print(local_dict)
    print(len(local_dict.keys()))
    partition_training_data(LF,global_dict,local_dict,dataList)
    partition_testing_data(LF,global_dict,local_dict)#,sup_dict
    shutil.copy('./file/test_label.csv', './SWaT_global_'+str(LF)+'/test_label.csv')