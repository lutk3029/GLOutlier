#AT和AT融合
import pickle
sup_list = [0.5,0.6,0.7,0.75,0.85] #sup of global method


def unify_global_and_local(LF):
    # for num in range(len(sup_list)):
    # index1 = sup + 6#index of local detection
    # index2 = sup + 11

    #get the continuous interval of local detection
    
    fw = open('./ATR/res_pickle_SWaT'+str(LF)+ '.txt', 'rb')
    det_list= pickle.load(fw)
    fw.close()

    n = len(det_list)#length of test dataset

    #get the label
    n = len(det_list)
    l = r = l1 =r1 = 0
    f_l = f_r = 0
    data_list = []
    data_list1 = []
    if(det_list[0][0]==1):
        fr = 1
    for i in range(len(det_list)):
        res = det_list[i]
        if (res[0] == 1):
            if(f_l):
                r = r+1
            else:
                f_l = 1
                l = r = i
        else:
            if(f_l):
                f_l = 0
                data_list.append([l,r])
        if (res[1] ==1):
            if(f_r):
                r1 = r1+1
            else:
                f_r = 1
                l1 = r1 = i
        else:
            if(f_r):
                f_r = 0
                data_list1.append([l1,r1])
    
    #get the result of global detection result
    # data_list3 = data_det_dict[0.7]
    sup_l = [0.7,0.75,0.8,0.85,0.9,0.95] #support of the global method
    for supp in sup_l:
        fr = open('./file/1.0-'+str(0.7)+'-' +str(supp)+'/det'+'.txt')
        data = fr.readlines()
        fr.close()
        data_list3 = []
        n = len(data)
        dataa = []
        for i in range(n):
            data[i] = data[i].rstrip("\n")
            dataa.append(data[i].split(' '))
        l = r = l1 =r1 = 0
        f_l = f_r = 0
        if(dataa[0][0]=='1'):
            fr = 1
        for i in range(len(dataa)):
            res = dataa[i]
            if (res[1] =='1'):
                if(f_r):
                    r1 = r1+1
                else:
                    f_r = 1
                    l1 = r1 = i
            else:
                if(f_r):
                    f_r = 0
                    data_list3.append([l1,r1])
                
    #结果结合
        fr = open('./file/data_attack_v0_pickle.txt', 'rb')
        dataDict = pickle.load(fr)
        fr.close()
        values = dataDict['values']
        name = dataDict['name']
        label = dataDict['label']
        real_det = []#the results
        for i in range(len(label)):
            real_det.append([str(label[i]),'0'])
        for i in range(len(data_list1)):
            if (data_list1[i][1] - data_list1[i][0] >= 10):
                for j in range(data_list1[i][0],data_list1[i][1]+1):
                        real_det[j][1] = '1'

        for i in range(len(data_list3)):
            if (data_list3[i][1] - data_list3[i][0] >= 10):
                for j in range(data_list3[i][0],data_list3[i][1]+1):
                        real_det[j][1] = '1'
        count = 0
        TP = FP = FN = TN = 0
        T_num = 0
        F_num = 0
        dataa = real_det
        # for i in range(n):
        #     if(dataa[i][1]=='1'):
        #         dataa[i][1]='0'
        #     if (dataa[i][1] == '2' or dataa1[i][1] == '1'):
        #         dataa[i][1]='1'
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

        if (TP + FP == 0):
            Precision =0
        else:
            Precision = TP / (TP + FP)
        Recall = TP / (TP +FN)
        if (Recall ==0):
            F1 = 0
        else:
            F1 = 2*Precision*Recall/(Precision + Recall)
        print(LF)
        print(TP ,FP,TN,FN)
        print(F_num,T_num)
        print(auc,Precision,Recall,F1)

if __name__ == '__main__':
    unify_global_and_local(0.5)