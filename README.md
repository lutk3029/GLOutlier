# Unifying Global and Local Anomaly Detection for Time Series

>

This repository contains code for the paper "Unifying Global and Local Anomaly Detection for Time Series" by Tongkai Lu, Shuai Ma and Zhongxi Zhang.

we propose an approach that unifies the global and local anomaly detection to achieve the good performance. To make the unification possible, we discretize time series data and partition the discrete data into global and local anomaly detection parts based on a concept of local factor. We design a rule-based global method using non-redundant association rules with tolerances to handle the difficulty of asynchronous changes of different attributes and the excessive number of rules. We develop a Transformer based on local method with LSH Attention to fit better for discrete data. We finally conduct an extensive experimental study to verify the advantage of our approach.


## Dataset
Download original SWaT and WADI datasets from  [iTrust](https://itrust.sutd.edu.sg/itrust-labs_datasets/).  
We also provide the [processed data](https://drive.google.com/drive/folders/1USOqY4xu4_kZTxM2_784TowQ2voUftQX?usp=sharing) and [pickle data] (https://drive.google.com/drive/folders/1ESdEykcOPRwwwy6lFV2_38TS_yqcTR51?usp=drive_link) for convience. The former file should be put in the `/data` directory and the latter should be put in the '/file'.


## Installation
We ran our code on Ubuntu 18.04 Linux, using Python 3.7.13, PyTorch 1.0.1.post2 and CUDA 9.0.

Use the following code to install python packages:


```sh
pip install -r requirements.txt
```

## Test our result
To directly test the result of our unified approach, please download the result of our local anomaly detection method in [here](https://drive.google.com/file/d/1KoO_CI8YU_IigcTLwA-EVJk9V1xq6r_K/view?usp=drive_link) and put it in the `/ATR` directory. 
Then run unidfying_detection_result.py to get the unification result

## Running our Unified approach (Method 1)
Before running our unified approach, make sure that you have downloaded our provided two kinds of data (both the processed and the pickle data).
Besides, you should also make sure the use of GPU and CPU device is the same as ours in 'run.py'.

```sh
python run.py
```


## Running our Unified approach (Method 2)
Step by step to run our approach
#### 1. Data discretization
```sh
python data_discretization.py
```

#### 2. Data partition
```sh
python data_partition.py
```

#### 3. Run global detection AR
```sh
cd ./AR
python All.py
```

#### 4. Train local detection ATR
```sh
cd ./ATR
bash train.sh
```

#### 5. Test local detection ATR
```sh
cd ./ATR
bash test.sh
```

#### 6. Get unfication results
```sh
cd ./ATR
python unidfying_detection_result.py
```

> **Note:** When executing the program, you have to execute data discretization and partition before detection. Here the default local factor is 0.5, if you want to change the local factor, you need to change it in 'data_partition.py' and the file name and input/output dim for running ATR should also be changed .

