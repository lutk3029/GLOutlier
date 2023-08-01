import sys
import os

args = sys.argv

rulestxt = './file/' + args[1] + "/rules.txt"
rulestxt1 = './file/' + args[1] + "/rules1.txt"


def listlt(list1, list2):
    for i in range(len(list1)):
        if list1[i] > list2[i]:
            return False
    return True


def merge(data_list, left, mid, right):
    i = left
    while i <= mid:
        if data_list[i]:
            j = mid + 1
            while j <= right:
                if data_list[j]:
                    if data_list[i][0] & data_list[j][0] == data_list[i][0] and data_list[i][1] & data_list[j][1] == data_list[j][1] and listlt(data_list[i][2], data_list[j][2]):
                        data_list[j] = []
                    elif data_list[j][0] & data_list[i][0] == data_list[j][0] and data_list[j][1] & data_list[i][1] == data_list[i][1] and listlt(data_list[j][2], data_list[i][2]):
                        data_list[i] = []
                        break
                j += 1
        i += 1


def delMerge(data_list, left, right):
    if left < right:
        mid = (left + right) // 2
        delMerge(data_list, left, mid)
        delMerge(data_list, mid + 1, right)
        merge(data_list, left, mid, right)


fr = open(rulestxt)
dataa = fr.readlines()
fr.close()

tot = []
tol_dict = {}
for i in range(0, len(dataa)):
    temp = dataa[i].split()
    tset1 = set()
    tset2 = set()
    for j in range(0, temp.index("-->")):
        tset1.add(temp[j])
    for j in range(temp.index("-->") + 1, len(temp) - 4):
        tset2.add(temp[j])
    tol = list(map(int, temp[-1][temp[-1].index("=")+1:].split('-')))
    tot.append((tset1, tset2, tol))

delMerge(tot, 0, len(tot) - 1)

# del_list = []
# for i in range(len(tot)):
#     for j in range(len(tot)):
#         if i != j:
#             if tot[j][0] & tot[i][0] == tot[j][0] and tot[j][1] & tot[i][1] == tot[i][1] and listlt(tot[j][2], tot[i][2]):
#                 del_list.append(i)

foru = open(rulestxt1, "w+")
for i in range(len(tot)):
    if tot[i]:
        print(dataa[i], end='', file=foru)
foru.close()
