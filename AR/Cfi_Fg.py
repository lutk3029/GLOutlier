# -*-coding:utf-8-*-

import pickle
import gc
import copy
import time
import sys
import os

args = sys.argv
args_l = args[1].split('-')
mc = float(args_l[2])

p = './file/'
path = './file/' + args[1] + '/'
if not os.path.exists(path):
    os.mkdir(path)

def get_size_tid(list1):
    ans_sum = 0
    for ll in list1:
        ans_sum += ll[1] - ll[0] + 1
    return ans_sum


def contain(list1, i):
    for l in list1:
        if l[0] <= i <= l[1]:
            return True
        if i < l[0]:
            return False
    return False


def get_tol(list1, list2):
    if list1 == list2:
        return [0, 0, 0, 0]
    else:
        templist0 = copy.deepcopy(list1)
        templist1 = copy.deepcopy(list2)
        ii = 0
        while True:
            if ii >= len(templist0):
                break
            for jj in range(len(templist1)):
                if templist0[ii][1] < templist1[jj][0]:
                    break
                if templist0[ii][0] > templist1[jj][1]:
                    continue
                if templist0[ii][0] >= templist1[jj][0]:
                    if templist0[ii][1] <= templist1[jj][1]:
                        templist0[ii] = [0, -1]
                    else:
                        templist0[ii] = [templist1[jj][1] + 1, templist0[ii][1]]
                else:
                    if templist0[ii][1] > templist1[jj][1]:
                        templist0.insert(ii + 1, [templist1[jj][1] + 1, templist0[ii][1]])
                    templist0[ii] = [templist0[ii][0], templist1[jj][0] - 1]
            ii += 1
        tol = [[0] for _ in range(2)]
        # s:满足规则   e:不相关
        # 0: s e
        # 1: s s
        # 2: e s
        # 3: e e

        # 0: s
        # 1: e
        for x in templist0:
            temp = x[1] - x[0] + 1
            if x == [0, -1]:
                continue
            if x[0] == 0:
                if x[1] == data_len - 1:
                    break
                else:
                    tol[0].append(temp)
                    tol[1].append(temp)
            else:
                if contain(list1, x[0] - 1):
                    tol[0].append(temp)
                else:
                    tol[1].append(temp)
        # templen = [x[1] - x[0] + 1 for x in templist0]
        return [max(x) for x in tol]


if __name__ == '__main__':
    fr = open(path + 'Hash_CFI_pickle.txt', 'rb')
    Hash_CFI = pickle.load(fr)
    fr.close()

    fr = open(path + 'Hash_gen_pickle.txt', 'rb')
    Hash_gen = pickle.load(fr)
    fr.close()

    fr = open(p + 'data_len_pickle' + args_l[0] + '.txt', 'rb')
    data_len = pickle.load(fr)
    fr.close()

    gen_clo_dict = {}
    clo_tid_dict = {}
    for kk, vv in Hash_CFI.items():
        for k in vv:
            clo_tid_dict[k[0]] = (get_size_tid(k[1]), k[1])
    for kk, vv in Hash_gen.items():
        for k in vv:
            for v in Hash_CFI[kk]:
                if k[1] == v[1]:
                    gen_clo_dict[k[0]] = v[0]
    del Hash_CFI, Hash_gen
    gc.collect()
    # for k, v in gen_clo_dict.items():
    #     print(k, v)
    #     break
    # for k, v in clo_tid_dict.items():
    #     print(k, v)
    #     break
    # print(len(gen_clo_dict), len(clo_tid_dict))
    a_time = time.time()
    foru = open(path + 'rules.txt', 'w+')
    tol_dict = {}
    ii = 1
    for k, v in gen_clo_dict.items():
        # print(ii)
        ii += 1
        for kc, vc in clo_tid_dict.items():#CFI2
            if len(k) < len(kc) and v & kc == v:
                sup = vc[0]
                sup1 = clo_tid_dict[v][0]
                sup2 = 0
                for ki, vi in gen_clo_dict.items():
                    if (kc - k) & ki == ki and (kc - k) & vi == (kc - k):
                        sup2 = clo_tid_dict[vi][0]
                        break
                if sup / sup1 < mc or data_len * sup / (sup1 * sup2) <= 1:
                    continue
                for i in k:
                    print(i, end=' ', file=foru)
                print("-->", end=' ', file=foru)
                for i in kc - k:
                    print(i, end=' ', file=foru)
                if frozenset([v, kc]) not in tol_dict.keys():
                    tol_dict[frozenset([v, kc])] = get_tol(clo_tid_dict[v][1], vc[1])#求容忍度,Tid集合
                tolList = tol_dict[frozenset([v, kc])]
                print("sup=%f conf=%f lift=%f tol=%d-%d" % (sup / data_len, sup / sup1,
                                                                  data_len * sup / (sup1 * sup2),
                                                                  tolList[0], tolList[1]),
                      end='\n', file=foru)
    foru.close()
    print('Generating time：',time.time()- a_time)
