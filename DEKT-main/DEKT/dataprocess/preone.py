
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import json
import time, datetime
from sklearn.model_selection import train_test_split

pd.set_option('display.float_format',lambda x : '%.2f' % x)
np.set_printoptions(suppress=True)


## 将最初的字段名改正##   
data =  pd.read_csv('../../data/anonymized_full_release_competition_dataset/anonymized_full_release_competition_dataset.csv', encoding = "ISO-8859-1", low_memory=False)
data['BORED'] = data['confidence(BORED)']
data['CONCENTRATING'] = data['confidence(CONCENTRATING)']
data['CONFUSED'] = data['confidence(CONFUSED)']
data['FRUSTRATED'] = data['confidence(FRUSTRATED)']

order = ['startTime', 'timeTaken', 'studentId', 'skill', 'problemId','problemType', 'correct','BORED','CONCENTRATING','CONFUSED','FRUSTRATED']
data = data[order]

data.to_csv('../../data/anonymized_full_release_competition_dataset/processed_data.csv', index=False)

# print(data.isnull().sum())
# print(data.info())








# 打印结果
print(data)



