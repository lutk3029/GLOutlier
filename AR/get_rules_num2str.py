import sys
import pickle

args = sys.argv
dir = args[1]

p = './file/'
rulestxt = p + dir + "/rules1.txt"#1
rulestxt1 = p + dir + "/rules2.txt"#2

if __name__ == '__main__':
    fr = open(p + 'data_all_list_pickle.txt', 'rb')
    data_all_list = pickle.load(fr)
    fr.close()

    fr = open(rulestxt)
    dataa = fr.readlines()
    fr.close()

    foru = open(rulestxt1, "w+")
    for d in dataa:
        temp = d.split()
        for i in range(len(temp)):
            if i < len(temp) - 4 and temp[i] != '-->':
                print(data_all_list[int(temp[i])], end=' ', file=foru)
            else:
                print(temp[i], end=' ', file=foru)
        print('', file=foru)
    foru.close()
