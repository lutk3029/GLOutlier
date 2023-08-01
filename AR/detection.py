# -*-coding:utf-8-*-

import sys
import pickle
import time
import collections
import copy

args = sys.argv
arg_l = args[1].split('-')
path = './file/'
rulestxt = path + args[1] + "/rules2.txt"
dettxt2 = path + args[1] +'/det'+'.txt'
#dettxt2 = path + '/det'+str(arg_l[1])+str(arg_l[0]) +'.txt'
#path + args[1]+  '/det'+'.txt'
sensor_trend_name = ['LIT101','LIT301', 'LIT401']
fr = open(path + 'sen_min_max_dict_pickle.txt', 'rb')
sen_min_max_dict = pickle.load(fr)
fr.close()
if __name__ == "__main__":
    det_list = []
    fr = open(rulestxt)
    rulesdata = fr.readlines()
    fr.close()
    fr = open(path + 'data_att_pickle.txt', 'rb')
    all_data_list_set = pickle.load(fr)
    fr.close()
    fr = open(path + 'label_pickle.txt', 'rb')
    og_label = pickle.load(fr)
    fr.close()
    fr = open(path + 'data_attack_v0_pickle.txt', 'rb')
    dataDict = pickle.load(fr)
    fr.close()
    values = dataDict['values']
    name = dataDict['name']
    label = dataDict['label']
    dic_num_l = dict()
    dic_num_r = dict()
    rulesdata1 = copy.deepcopy(rulesdata)
    tol_max = 100
    for i in range(len(rulesdata1)):
        temp = rulesdata1[i].split()
        ll = list(map(int, temp[-1][temp[-1].index("=")+1:].split('-')))
        if (ll[0]>= tol_max or ll[1]>=tol_max):
            rulesdata.remove(rulesdata1[i])

    for i in range(len(rulesdata)):#对规则前，后半部分不同项进行计数
        temp = rulesdata[i].split()#['FIT201:1','-->','MV304:1', 'P101+P102:3','P203+P204:3','FIT601:1','LIT101:0','P602:1','MV201:2','MV301:1','MV303:1','sup=0.706821','conf=0.948225','lift=1.341535','tol=155-16']
        for j in range(temp.index('-->')):#前一部分
            if temp[j] not in dic_num_l.keys():
                dic_num_l[temp[j]] = 0
            else:
                dic_num_l[temp[j]] += 1
        for j in range(temp.index('-->') + 1, len(temp) - 4):
            if temp[j] not in dic_num_r.keys():
                dic_num_r[temp[j]] = 0
            else:
                dic_num_r[temp[j]] += 1
    lis_num_l = []
    lis_num_r = []
    for k, v in dic_num_l.items():#变成有序对形式用数组储存
        lis_num_l.append((k, v))
    for k, v in dic_num_r.items():
        lis_num_r.append((k, v))
    lis_num_l.sort(reverse=True, key=lambda x: x[1])#按出现频率由高到低排序
    lis_num_r.sort(reverse=True, key=lambda x: x[1])
    lis_num_l = [x[0] for x in lis_num_l]#项
    lis_num_r = [x[0] for x in lis_num_r]

    rules_list = []  # 通过学习得到的规则
    rules_l = dict()
    rules_r = dict()
    for i in range(len(rulesdata)):#对所有关联规则左部和右部构造前缀树
        temp = rulesdata[i].split()
        tempset_l = set([])
        tempset_r = set([])
        for j in range(temp.index('-->')):
            tempset_l.add(temp[j])
        for j in range(temp.index('-->') + 1, len(temp) - 4):
            tempset_r.add(temp[j])
        rules_list.append([tempset_l, tempset_r, list(map(int, temp[-1][temp[-1].index("=")+1:].split('-'))),#前半部分是左边规则，后半部分是右边规则，temp(-1)容忍度，形式为155-16 ，分成两部分
                           float(temp[-4][temp[-4].index("=")+1:]), float(temp[-3][temp[-3].index("=")+1:])])#所以接下来是一个容忍度的集合[tol1,tol2].然后是支持度和置信度

        tempset_l = list(tempset_l)
        tempset_r = list(tempset_r)
        tempset_l.sort(key=lambda x : lis_num_l.index(x))#同样是按在规则中的出现频率排序
        tempset_r.sort(key=lambda x: lis_num_r.index(x))
        tree = rules_l
        for l in tempset_l:
            if l not in tree.keys():
                tree[l] = dict()
            tree = tree[l]
        if 'end' in tree.keys():
            tree['end'].add(i)
        else:
            tree['end'] = set()
            tree['end'].add(i)

        tree = rules_r
        for l in tempset_r:
            if l not in tree.keys():
                tree[l] = dict()
            tree = tree[l]
        if 'end' in tree.keys():
            tree['end'].add(i)
        else:
            tree['end'] = set()
            tree['end'].add(i)

    for iii in range(1):  #
        
        a_time = time.time()
        res2 = [0] * len(og_label)#结果集
        rules_dict = [{'status_s': 's', 't': 0} for i in range(len(rules_list))]#存每个规则被违背之前的状和每个规则被连续违背的时间，i代表状态
        old_set = set()#上一时刻违反规则的集合
        T = 0
        a_time = time.time()
        foru = open(dettxt2, "w+")
        for i in range(len(all_data_list_set)):
            # print(i)
            vio_list = []
            temp_set_r = set()
            deque = collections.deque()
            for j, v in rules_r.items():#先遍历右树
                if j in all_data_list_set[i]:
                    deque.append(v)
            while deque:
                temp = deque.popleft()
                for k, v in temp.items():
                    if k == 'end':
                        temp_set_r |= v#当前数据有v代表的规则的右端
                    else:
                        if k in all_data_list_set[i]:
                            deque.append(v)
            if len(temp_set_r) == len(rulesdata):#所有规则右边都在当前数据里，说明其啥规则都不违背
                new_set = set()#违背当前时刻规则的集合
            else:#有一部分不满足的时候，要搜索左树
                temp_set_l = set()
                deque = collections.deque()
                for j, v in rules_l.items():
                    if j in all_data_list_set[i]:
                        deque.append(v)
                while deque:
                    temp = deque.popleft()
                    for k, v in temp.items():
                        if k == 'end':
                            temp_set_l |= v
                        else:
                            if k in all_data_list_set[i]:
                                deque.append(v)
                new_set = temp_set_l - temp_set_r#满足左树但不满足右树，代表当前时刻违背的规则的编号

            for j in new_set:#遍历当前时刻违背规则的集合
                if j in old_set:
                    rules_dict[j]['t'] += 1
                else:#j不在old_set或者old_set为空时
                    rules_dict[j]['t'] = 1
                    if i > 0:
                        if rules_list[j][0] & all_data_list_set[i - 1] == rules_list[j][0]:#满足前一项规则
                            rules_dict[j]['status_s'] = 's'
                        else:
                            rules_dict[j]['status_s'] = 'e'#不相关
                if (rules_dict[j]['status_s'] == 's' and rules_dict[j]['t'] > rules_list[j][2][0]) or \
                        (rules_dict[j]['status_s'] == 'e' and rules_dict[j]['t'] > rules_list[j][2][1]):
                    res2[i] = 1#规则被连续违背的时间大于对应状态的容忍度时算作违背规则
                    vio_list.append(j)
                # if (rules_dict[j]['t'] > max(rules_list[j][2][0],rules_list[j][2][1])):
                #     res2[i] = 1
            old_set = new_set
            flag = 0
            # nm = 'DPIT301'
            # sen = values[i][name.index(nm)]
            # if (sen >30):
            #     flag =1
            # for nm in sensor_trend_name:
            #     min_temp = sen_min_max_dict[nm][0]
            #     max_temp = sen_min_max_dict[nm][1]
            #     sen = values[i][name.index(nm)]
            #     if (sen>max_temp+50 or sen<min_temp-50):
            #         flag = 1
            #         break
                # if (sen_list[j]>max_temp):
                #     data_list_set[j].add(nm + ':' + str(5))
                # elif (sen_list[j]<min_temp):
                #     data_list_set[j].add(nm + ':' + str(6))
            if (res2[i] > T ):#or (i in list(range(227828,263727)))
                print(str(og_label[i]) + ' ' + '1', file=foru)#左边是原始数据，右边是算出来的数据
                det_list.append([i,str(og_label[i]),'1',vio_list])
            else:
                print(str(og_label[i]) + ' ' + '0', file=foru)
                det_list.append([i,str(og_label[i]),'0',vio_list])
        foru.close()
    print(args)
    print('检测结束，时间：', time.time() - a_time)
    fw = open(path + args[1] + '/det_list_pickle.txt', 'wb')
    pickle.dump(det_list, fw)
    fw.close()