import time
import pickle
import sys
import os

args = sys.argv
args_l = args[1].split('-')
ms = float(args_l[1])

p = './file/'
path = './file/' + args[1] + '/'

if not os.path.exists(path):
    os.mkdir(path)

def get_inter_tid(list1, list2):
    q, p = 0, 0
    temp2 = []
    while True:
        if q == len(list1) or p == len(list2):
            break
        if list1[q][0] > list2[p][1]:
            p += 1
            continue
        if list2[p][0] > list1[q][1]:
            q += 1
            continue
        temp2.append([max(list1[q][0], list2[p][0]), min(list1[q][1], list2[p][1])])
        if list2[p][1] <= list1[q][1]:
            p += 1
        else:
            q += 1
    return temp2


def get_size_tid(list1):
    ans_sum = 0
    for ll in list1:
        ans_sum += ll[1] - ll[0] + 1
    return ans_sum


def get_hash(list1):
    ans_sum = 0
    for i in list1:
        ans_sum += i[0]
        ans_sum += i[1]
    return ans_sum % 1000


def getNextGenerator(nodes, curr, other, c, minSup):
    cand_tidset = get_inter_tid(nodes[curr], nodes[other])
    if get_size_tid(cand_tidset) < minSup:
        return -1, -1
    if cand_tidset == nodes[curr] or cand_tidset == nodes[other]:
        return -1, -1
    cand_itemset = curr | other
    num = get_hash(cand_tidset)
    if num in c.keys():
        for l in c[num]:
            if cand_tidset == l[1] and cand_itemset & l[0] == l[0]:
                return -1, -1
    nodes[cand_itemset] = cand_tidset
    return cand_itemset, cand_tidset


def extend(nodes, chi, nei, c, minSup):
    temp_list = []
    for i in range(len(nei)):
        gen1, gen2 = getNextGenerator(nodes, chi, nei[i], c, minSup)
        if gen1 != -1 and gen2 != -1:
            temp_list.append(gen1)
    for i in range(len(temp_list) - 1, -1, -1):
        num = get_hash(nodes[temp_list[i]])
        if num not in c.keys():
            c[num] = [(temp_list[i], nodes[temp_list[i]])]
        else:
            c[num].append((temp_list[i], nodes[temp_list[i]]))
        extend(nodes, temp_list[i], temp_list[i + 1: len(temp_list)], c, minSup)


def talky_g(ip_dict, minSup):
    temp_list = list(ip_dict.keys())
    for k in temp_list:
        if get_size_tid(ip_dict[k]) < minSup:
            del ip_dict[k]
    c = {}
    temp_list = list(ip_dict.keys())
    for i in range(len(temp_list) - 1, -1, -1):
        num = get_hash(ip_dict[temp_list[i]])
        if num not in c.keys():
            c[num] = [(temp_list[i], ip_dict[temp_list[i]])]
        else:
            c[num].append((temp_list[i], ip_dict[temp_list[i]]))
        extend(ip_dict, temp_list[i], temp_list[i + 1: len(temp_list)], c, minSup)
    return c


def loadSimpDat(dataa):
    data_dict_list2 = {}
    for i in range(len(dataa)):
        temp = dataa[i].split()
        for name in temp:
            name = frozenset([int(name)])
            if name in data_dict_list2.keys():
                if data_dict_list2[name][-1][1] + 1 == i:
                    data_dict_list2[name][-1][1] = i
                else:
                    data_dict_list2[name].append([i, i])
            else:
                data_dict_list2[name] = [[i, i]]
    return data_dict_list2


if __name__ == '__main__':
    fr = open(p + 'data_num' + args_l[0] + '.txt')
    data = fr.readlines()
    fr.close()
    data_dict = loadSimpDat(data)
    print('================START=================')
    start_time = time.time()
    ans = talky_g(data_dict, ms * len(data))
    print('Time: ', time.time() - start_time)
    print('================END=================')
    fw = open(path + 'Hash_gen_pickle.txt', 'wb')
    pickle.dump(ans, fw)
    fw.close()

    fr = open(p + 'data_all_list_pickle.txt', 'rb')
    data_all_list = pickle.load(fr)
    fr.close()

    fw = open(path + 'Hash_gen.txt', 'w+')
    for kk, vv in ans.items():
        for v in vv:
            for i in v[0]:
                print(data_all_list[int(i)], end=' ', file=fw)
            print('', file=fw)
    fw.close()
