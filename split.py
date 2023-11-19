import pandas as pd

import pandas as pd
import os

report_df = pd.read_csv('/jhcnas1/xinyi/zhongshan_380/zhongshan_report.csv')
before_df = pd.read_csv('/jhcnas1/xinyi/zhongshan_380/before_370.csv')
before_root = '/jhcnas1/xinyi/zhongshan_380/before_npy'
after_df = pd.read_csv('/jhcnas1/xinyi/zhongshan_380/after_370.csv')
after_root = '/jhcnas1/xinyi/zhongshan_380/after_npy'
new_df = pd.DataFrame()
new_csv_path = '/jhcnas1/xinyi/zhongshan_380/patient_370.csv'

# 添加空列
new_df['name'] = ''
new_df['num']=''
new_df['PCR'] = ''
new_df['before_T1_COR_W'] = ''
new_df['before_T1_TRA_W'] = ''
new_df['before_T2_TRA_W'] = ''
new_df['after_T1_COR_W'] = ''
new_df['after_T1_TRA_W'] = ''
new_df['after_T2_TRA_W'] = ''

m=0
for i, row in report_df.iterrows():
    if not (pd.isna(row).all()):
        for j in range(len(before_df)):
            num = str(before_df.at[j, 'name']).split('_')[1]
            if str(int(row['影像号'])) == num:
                found=False
                for k in range(len(after_df)):
                    try:
                        num1 = str(after_df.at[k, 'name']).split('_')[1]
                        if num == num1:
                            found = True
                            new_df.at[m, 'name'] = before_df.at[j, 'name']
                            new_df.at[m, 'PCR'] = row['PCR']
                            new_df.at[m,'num']=num
                            new_df.at[m, 'before_T1_COR_W'] = os.path.join(before_df.at[j, 'name'], 'T1_COR.npy')
                            new_df.at[m, 'before_T1_TRA_W'] = os.path.join(before_df.at[j, 'name'], 'T1_TRA.npy')
                            new_df.at[m, 'before_T2_TRA_W'] = os.path.join(before_df.at[j, 'name'], 'T2_TRA.npy')
                            new_df.at[m, 'after_T1_COR_W'] = os.path.join(after_df.at[k, 'name'], 'T1_COR.npy')
                            new_df.at[m, 'after_T1_TRA_W'] = os.path.join(after_df.at[k, 'name'], 'T1_TRA.npy')
                            new_df.at[m, 'after_T2_TRA_W'] = os.path.join(after_df.at[k, 'name'], 'T2_TRA.npy')
                            m+=1
                            break
                    except:
                        pass
                if not found:
                    name = str(before_df.at[j, 'name']).split('_')[0]

                    for k in range(len(after_df)):
                        name1 = str(after_df.at[k, 'name']).split('_')[0]
                        if name == name1:
                            found=True
                            new_df.at[m, 'name'] = before_df.at[j, 'name']
                            new_df.at[m, 'PCR'] = row['PCR']
                            new_df.at[m, 'num'] = num
                            new_df.at[m, 'before_T1_COR_W'] = os.path.join(before_df.at[j, 'name'], 'T1_COR.npy')
                            new_df.at[m, 'before_T1_TRA_W'] = os.path.join(before_df.at[j, 'name'], 'T1_TRA.npy')
                            new_df.at[m, 'before_T2_TRA_W'] = os.path.join(before_df.at[j, 'name'], 'T2_TRA.npy')
                            new_df.at[m, 'after_T1_COR_W'] = os.path.join(after_df.at[k, 'name'], 'T1_COR.npy')
                            new_df.at[m, 'after_T1_TRA_W'] = os.path.join(after_df.at[k, 'name'], 'T1_TRA.npy')
                            new_df.at[m, 'after_T2_TRA_W'] = os.path.join(after_df.at[k, 'name'], 'T2_TRA.npy')
                            m+=1
                            break

                break

df_new = new_df
new_df.to_csv(new_csv_path, index=False)


# 计算新数据量及划分比例
n_new = len(df_new)
ratio = [0.7, 0.1, 0.2]
n_split = [int(r * n_new) for r in ratio]

# 从新数据中进行采样
new_train = df_new.sample(n_split[0])
remainder = df_new.drop(new_train.index)
new_val = remainder.sample(n_split[1])
new_test = remainder.drop(new_val.index)

# 输出更新后的数据集
new_train.to_csv('/jhcnas1/xinyi/zhongshan_380/train_370.csv', index=False)
new_val.to_csv('/jhcnas1/xinyi/zhongshan_380/val_370.csv', index=False)
new_test.to_csv('/jhcnas1/xinyi/zhongshan_380/test_370.csv', index=False)