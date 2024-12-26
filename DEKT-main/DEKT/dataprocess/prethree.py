import pandas as pd



data =  pd.read_csv('../../../data/anonymized_full_release_competition_dataset/processed_data.csv', encoding = "ISO-8859-1", low_memory=False)
# 'BORED','CONCENTRATING','CONFUSED','FRUSTRATED']
print(len(data))
columns_to_round = ['BORED','CONCENTRATING','CONFUSED','FRUSTRATED']
# 创建一个条件，选择需要删除的行
condition_to_delete = (data[columns_to_round] == 1.0).any(axis=1)
# 删除符合条件的行
data_cleaned = data[~condition_to_delete]
print(len(data_cleaned))
file_path = '../../../data/anonymized_full_release_competition_dataset/test.csv'  
data_cleaned.to_csv(file_path, index=False)




