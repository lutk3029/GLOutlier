import os
import time
data_s_c_dict = {}
min_s_c_dict = {}
# min_s_c_dict[0.7] = [0.7]#
min_s_c_dict[0.7] = [0.75]
min_s_c_dict[0.7] = [0.8]
min_s_c_dict[0.7] = [0.85]
min_s_c_dict[0.7] = [0.9]
# min_s_c_dict[0.7] = [0.95]
# min_s_c_dict[0.8] = [0.9]
# min_s_c_dict[0.85] = [0.9]
# min_s_c_dict[0.9] = [0.9]
# min_s_c_dict[0.95] = [0.9]
# min_s_c_dict[0.7] = [0.95]#best
# min_s_c_dict[0.75] = [0.95]
# min_s_c_dict[0.8] = [0.95]
# min_s_c_dict[0.85] = [0.95]
# min_s_c_dict[0.9] = [0.95]
# min_s_c_dict[0.95] = [0.95]

data_s_c_dict[1.0] = min_s_c_dict


def run_global_detection(data_len,min_ms,min_mc):
    print("start data_string2num.py")
    os.system("python ./AR/data_string2num.py " + str(data_len))

    print("finish data_string2num.py")

    #Mining Frequent closed itemset
    print("start Charm.py")
    os.system("python ./AR/Charm.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc))
    print("finish Charm.py")

    # Mining Generator
    print("start TalkyG.py")
    os.system("python ./AR/TalkyG.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc))
    print("finish TalkyG.py")

    # Mining non-redundant rules
    print("start Cfi_Fg.py")
    os.system("python ./AR/Cfi_Fg.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc))
    print("finish Cfi_Fg.py")

    # Mining non-redundant rules with tolerances
    print("start del_rules.py")
    os.system("python ./AR/del_rules.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc))
    print("finish del_rules.py")

    # change num to str
    print("start get_rules_num2str.py")
    os.system("python ./AR/get_rules_num2str.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc))
    print("finish get_rules_num2str.py")

    # anomaly detection
    print("start detection.py")
    os.system("python ./AR/detection.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc))
    # os.system("python ./detection_without_tol.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc))
    print("finish detection.py")



if __name__ == '__main__':
    for data_len, l in data_s_c_dict.items():
        #将字符转为数字
        print("start data_string2num.py")
        os.system("python ./AR/data_string2num.py " + str(data_len))
        print("finish data_string2num.py")
        for min_ms, v in l.items():
            for min_mc in v:
                pass
                #=======================训练阶段=======================
                #Mining Frequent closed itemset
                print("start Charm.py")
                os.system("python ./AR/Charm.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc))
                print("finish Charm.py")

                # Mining Generator
                print("start TalkyG.py")
                os.system("python ./AR/TalkyG.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc))
                print("finish TalkyG.py")

                # Mining non-redundant rules
                print("start Cfi_Fg.py")
                os.system("python ./AR/Cfi_Fg.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc))
                print("finish Cfi_Fg.py")

                # Mining non-redundant rules with tolerances
                print("start del_rules.py")
                os.system("python ./AR/del_rules.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc))
                print("finish del_rules.py")

                # change num to str
                print("start get_rules_num2str.py")
                os.system("python ./AR/get_rules_num2str.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc))
                print("finish get_rules_num2str.py")

                # =======================testing phase======================
                # anomaly detection
                print("start detection.py")
                os.system("python ./AR/detection.py " + str(data_len) + '-' + str(min_ms) + '-' + str(min_mc))
                print("finish detection.py")