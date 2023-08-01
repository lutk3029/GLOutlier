# Unifying Global and Local Anomaly Detection for Time Series

>

This repository contains code for the paper "Unifying Global and Local Anomaly Detection for Time Series" by Tongkai Lu, Shuai Ma and Zhongxi Zhang.

we propose an approach that unifies the global and local anomaly detection to achieve the good performance. To make the unification possible, we discretize time series data and partition the discrete data into global and local anomaly detection parts based on a concept of local factor. We design a rule-based global method using non-redundant association rules with tolerances to handle the difficulty of asynchronous changes of different attributes and the excessive number of rules. We develop a Transformer based on local method with LSH Attention to fit better for discrete data. We finally conduct an extensive experimental study to verify the advantage of our approach.





## Installation
We ran our code on Ubuntu 18.04 Linux, using Python 3.7.13, PyTorch 1.0.1.post2 and CUDA 9.0.

Use the following code to install python packages:


```sh
pip install -r requirements.txt
```

## Running our Unified approach (Method 1)
#### Run our unified approach

```sh
python run.py
```


## Running our Unified approach (Method 2)
Step by step to run our approach
#### Data discretization
```sh
python data_discretization.py
```

#### Data partition
```sh
python data_partition.py
```

#### Run global detection AR
```sh
cd ./AR
python All.py
```

#### Train local detection ATR
```sh
cd ./ATR
bash train.sh
```

#### Test local detection ATR
```sh
cd ./ATR
bash test.sh
```

#### Get unfication results
```sh
cd ./ATR
python unidfying_detection_result.py
```

> **Note:** When executing the program, you have to execute data discretization and partition before detection. Here the default local factor is 0.5, if you want to change the local factor, you need to change it in 'data_partition.py' and the file name and input/output dim for running ATR should also be changed .
