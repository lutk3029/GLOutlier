import time
import pickle
import sys
import os

args = sys.argv#"python Charm.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc)
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


def replaceInItems(sc, ta, dic):
    temp_list = list(dic.keys())
    for k in temp_list:
        if k & sc == sc:
            dic[k | ta] = dic[k]
            del dic[k]


def isSubsumed(dic, num, y):
    if num not in dic.keys():
        return False
    else:
        list1 = dic[num]
        for l in list1:
            if l[1] == y:
                return True
        return False


def charmProp(xi, xj, y, minSup, nodes, newN, skip_set):
    if get_size_tid(y) >= minSup:
        if nodes[xi] == nodes[xj]:
            skip_set.add(xj)
            temp = xi | xj
            replaceInItems(xi, temp, newN)
            replaceInItems(xi, temp, nodes)
            return temp
        elif get_inter_tid(nodes[xi], nodes[xj]) == nodes[xi]:
            temp = xi | xj
            replaceInItems(xi, temp, newN)
            replaceInItems(xi, temp, nodes)
            return temp
        elif get_inter_tid(nodes[xi], nodes[xj]) == nodes[xj]:
            skip_set.add(xj)
            newN[xi | xj] = y
        elif nodes[xi] != nodes[xj]:
            newN[xi | xj] = y
    return xi


def charmExtended(nodes, c, minSup, skip_set):
    temp_list = list(nodes.keys())
    for i in range(len(temp_list)):
        fs_xi = temp_list[i]
        if fs_xi in skip_set:
            continue
        newN = {}
        # fs_x_prev = fs_xi
        x = frozenset([])
        for j in range(i + 1, len(temp_list)):
            fs_xj = temp_list[j]
            if fs_xj in skip_set:
                continue
            x = fs_xi | fs_xj
            y = get_inter_tid(nodes.get(fs_xi, []), nodes.get(fs_xj, []))
            fs_xi = charmProp(fs_xi, fs_xj, y, minSup, nodes, newN, skip_set)
        if len(newN) != 0:
            charmExtended(newN, c, minSup, skip_set)
        if nodes.get(fs_xi) and not isSubsumed(c, get_hash(nodes.get(fs_xi)), nodes.get(fs_xi)):
            num = get_hash(nodes.get(fs_xi))
            if num not in c.keys():
                c[num] = [(fs_xi, nodes.get(fs_xi))]
            else:
                c[num].append((fs_xi, nodes.get(fs_xi)))
        if nodes.get(x) and not isSubsumed(c, get_hash(nodes.get(x)), nodes.get(x)):
            num = get_hash(nodes.get(x))
            if num not in c.keys():
                c[num] = [(x, nodes.get(x))]
            else:
                c[num].append((x, nodes.get(x)))


def charm(ip_dict, minSup):
    temp_list = list(ip_dict.keys())
    for k in temp_list:
        if get_size_tid(ip_dict[k]) < minSup:
            del ip_dict[k]
    c = {}
    skip_set = set()
    charmExtended(ip_dict, c, minSup, skip_set)
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
    # for k, v in data_dict.items():
    #     print(k, v)
    print('================START=================')
    start_time = time.time()
    ans = charm(data_dict, ms * len(data))
    print('Time: ', time.time() - start_time)
    print('================END=================')
    fw = open(path + 'Hash_CFI_pickle.txt', 'wb')
    pickle.dump(ans, fw)
    fw.close()

    fr = open(p + 'data_all_list_pickle.txt', 'rb')
    data_all_list = pickle.load(fr)
    fr.close()

    fw = open(path + 'Hash_CFI.txt', 'w+')
    for kk, vv in ans.items():
        for v in vv:
            for i in v[0]:
                print(data_all_list[int(i)], end=' ', file=fw)
            print('', file=fw)
    fw.close()
