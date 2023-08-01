import sys
import os
from data_discretization import data_discretization
from data_partition import data_partition
from unidfying_detection_result import unify_global_and_local


if __name__ == '__main__':

    #prepare data first
    #os.system("python pre_data.py")
    

    #the actuator and sensor of SWaT
    actuator_p = ['P101', 'P201', 'P203', 'P205', 'P302', 'P402', 'P403', 'P501']#列出所有成对的，储存的时候成对的存在一起
    actuator_p_backup = ['P102', 'P202', 'P204', 'P206', 'P301', 'P401', 'P404', 'P502']
    sensor_value_name = ['FIT101', 'FIT201','DPIT301', 'FIT301', 'FIT601']# 'FIT101', 'FIT201','DPIT301', 'FIT301', 'FIT601'
# #sensor_trend_name = ['LIT101', 'LIT301', 'LIT401','AIT203','AIT501','AIT502','AIT402']
    sensor_trend_name = ['LIT101','LIT301', 'LIT401']# 
    # get_act_data()
    # get_cluster_center(sensor_value_name,sensor_trend_name)

    #data discretization first
    data_discretization(sensor_value_name,sensor_trend_name,actuator_p,actuator_p_backup)


    #perform global detection
    os.system("python ./AR/All.py")

    #preform local detection
    os.system("srun -p dell --gres=gpu:1 python3.6 ./ATR/main.py  --anormly_ratio 0.5 --num_epochs 3    --batch_size 256  --mode train --dataset SWaT0.5  --data_path ./SWaT_local_0.5 --input_c 17    --output_c 17")
    os.system("srun -p cpu python3.6 ./ATR/main.py --anormly_ratio 0.1  --num_epochs 10       --batch_size 256     --mode test    --dataset SWaT0.5   --data_path ./SWaT_local_0.5  --input_c 17    --output_c 17  --pretrained_model 10")

    #combine global and local detection method
    unify_global_and_local(0.5)