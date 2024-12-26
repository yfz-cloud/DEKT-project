
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split,KFold
import csv
data = pd.read_csv(
    '../../../data/anonymized_full_release_competition_dataset/processed_data.csv',
    usecols = ['startTime', 'timeTaken', 'studentId', 'skill', 'problemId','problemType', 'correct','BORED','CONCENTRATING','CONFUSED','FRUSTRATED']
).dropna(subset=['skill', 'problemId']).sort_values('startTime')
data.timeTaken = data.timeTaken.astype(int)


# # 将数据离散化为十类，间距为0.1，且0属于第一类，1属于最后一类
# # 且去除右边界

# bins = [i * (1/100) for i in range(101)]  # 生成区间列表 [0.0, 0.033, 0.066, ..., 1.0]
# labels = range(0, 100)  # 设置对应的类别标签 [1, 2, 3, ..., 30]
# data['FRU'] = pd.cut(data['FRUSTRATED'], bins=bins, labels=labels, include_lowest=True, right=False)
# data['CONF'] = pd.cut(data['CONFUSED'], bins=bins, labels=labels, include_lowest=True, right=False)
# data['CONC'] = pd.cut(data['CONCENTRATING'], bins=bins, labels=labels, include_lowest=True, right=False)
# data['BOR'] = pd.cut(data['BORED'], bins=bins, labels=labels, include_lowest=True, right=False)


# data.to_csv('../../../data/anonymized_full_release_competition_dataset/test.csv', index=False)



# # 定义映射函数
# def map_to_class(value):
#     return int(value * 100)

# data['FRU'] = data['FRUSTRATED'].apply(map_to_class)
# data['CONF'] = data['CONFUSED'].apply(map_to_class)
# data['CONC'] = data['CONCENTRATING'].apply(map_to_class)
# data['BOR'] = data['BORED'].apply(map_to_class)

# data.to_csv('../../../data/anonymized_full_release_competition_dataset/test.csv', index=False)

# 指定要操作的四列名称
columns_to_round = ['BORED','CONCENTRATING','CONFUSED','FRUSTRATED']

# 将指定列的值保留四位小数
data[columns_to_round] = data[columns_to_round].round(4)
data.to_csv('../../../data/anonymized_full_release_competition_dataset/test0.csv', index=False)

