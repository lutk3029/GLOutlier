import pickle
import sys

args = sys.argv
data_len = float(args[1])
print("data_len=%f\n",data_len)
path = './file/'

if __name__ == '__main__':
    fr = open(path + 'actuator_data_list_set_pickle.txt', 'rb')
    actuator_data_list_set = pickle.load(fr)
    fr.close()
    fr = open(path + 'sensor_value_data_list_set_pickle.txt', 'rb')
    sensor_value_data_list_set = pickle.load(fr)
    fr.close()
    fr = open(path + 'sensor_trend_data_list_set_pickle.txt', 'rb')
    sensor_trend_data_list_set = pickle.load(fr)
    fr.close()
    # fr = open(path +'sensor_data_list_set_pickle.txt', 'rb')
    # sensor_data_list_set = pickle.load(fr)
    # fr.close()

    data_all_set = set()
    dataList = []
    for i in range(len(actuator_data_list_set)):# 
        data_all_set = data_all_set | sensor_trend_data_list_set[i] | actuator_data_list_set[i] | \
                       sensor_value_data_list_set[i]
        dataList.append(list(sensor_trend_data_list_set[i] | actuator_data_list_set[i] | sensor_value_data_list_set[i]))
   

    data_all_list = sorted(list(data_all_set))
    fw = open(path + 'data_all_list_pickle.txt', 'wb')
    pickle.dump(data_all_list, fw)
    fw.close()
    fw = open(path + 'data_len_pickle' + args[1] + '.txt', 'wb')
    pickle.dump(len(dataList), fw)
    fw.close()
    # for i in range(len(data_all_list)):
    #     print(str(i) + ': ' + data_all_list[i])
    fo = open(path + 'data_num' + args[1] + '.txt', "w+")
    for d in dataList:
        for dd in d:
            print(data_all_list.index(dd), end=' ', file=fo)
        print('', file=fo)
    fo.close()
