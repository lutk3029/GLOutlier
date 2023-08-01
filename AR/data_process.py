import os
import time
import sys

data_s_c_dict = {}
min_s_c_dict = {}
# min_s_c_dict[0.5] = [0.6,0.7,0.8,0.9]
# min_s_c_dict[0.6] = [0.6,0.7,0.8,0.9]
# min_s_c_dict[0.7] = [0.7,0.8,0.9,1.0]#
# min_s_c_dict[0.8] = [0.8,0.9]
# min_s_c_dict[0.9] = [0.9]
# data_s_c_dict[0.6] = min_s_c_dict
min_s_c_dict[0.7] = [0.9]

data_s_c_dict[1.0] = min_s_c_dict
path = "./file/"
# args = sys.argv#"python Charm.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc)
# args_l = args[1]

for data_len, l in data_s_c_dict.items():
#     print("start data_string2num.py")
#     os.system("python D:/vscode_document/anomally_detection_code/AR/data_string2num.py " + str(data_len))
#     print("finish data_string2num.py")
    for min_ms, v in l.items():
        for min_mc in v:
            args = str(data_len) + '-' + str(min_ms) + '-' + str(min_mc)
            fr = open(path + args +'/det'+'.txt')
            #fr = open(path + args +'/det_without_tol'+'.txt')
            #fr = open('D:/vscode_document/anomally_detection_code/CNN/'+'det-50#20#2#32#100#128#1e-05#4'+'.txt')
            data = fr.readlines()
            fr.close()
            n = len(data)
            dataa = []
            for i in range(n):
                data[i] = data[i].rstrip("\n")
                dataa.append(data[i].split(' '))
            count = 0
            TP = FP = FN = TN = 0
            T_num = 0
            F_num = 0
            for i in range(n):
                if((dataa[i][1] != '0') & (dataa[i][0] != '0')):
                    TP = TP + 1
                    T_num= T_num+1
                if(dataa[i][0] == dataa[i][1]):
                    count = count + 1
                if((dataa[i][1] != '0') & (dataa[i][0] == '0')):
                    FP = FP + 1
                    F_num = F_num + 1
                if((dataa[i][1] == '0') & (dataa[i][0] != '0')):
                    FN = FN + 1
                    T_num= T_num+1
                if((dataa[i][1] == '0') & (dataa[i][0] == '0')):
                    TN = TN + 1
                    F_num = F_num + 1
            auc =(TP+TN)/(TP+TN+FP+FN)
            Precision = TP / (TP + FP)
            Recall = TP / (TP +FN)
            TK = TN/ (FP + TN)
            print(args)
            print(TP ,FP,TN,FN)
            print(F_num,T_num)
            print(auc,Precision,Recall,2*Precision*Recall/(Precision + Recall))