import pandas as pd

import pandas as pd
import os

report_df=pd.read_csv('/jhcnas1/xinyi/zhongliu_67/report.csv')
new_df = pd.DataFrame()
new_csv_path = '/jhcnas1/xinyi/zhongliu_67/patient.csv'
root='/jhcnas1/xinyi/zhongliu_67/resample'
# 添加空列

new_df['num']=''
new_df['PCR'] = ''
new_df['before_T1_COR'] = ''
new_df['before_T1_TRA'] =''
new_df['before_T2_TRA'] =''
new_df['after_T1_COR'] = ''
new_df['after_T1_TRA'] =''
new_df['after_T2_TRA'] =''

j=0
for i, row in report_df.iterrows():
    if not (pd.isna(row).all()):
        num=str(int(row['放射科号']))
        patient_path=os.path.join(root,num)
        before_path=os.path.join(patient_path,'Before')
        after_path=os.path.join(patient_path,'After')

        before_mol=os.listdir(before_path)
        after_mol = os.listdir(after_path)
        if 'T1_COR.npy' in before_mol and 'T1_TRA.npy' in before_mol and 'T2_TRA.npy' in before_mol and 'T1_COR.npy' in after_mol and 'T1_TRA.npy' in after_mol and 'T2_TRA.npy' in after_mol:
            new_df.at[j,'num']=num
            new_df.at[j,'PCR']=row['PCR']
            new_df.at[j,'before_T1_COR'] = os.path.join(num,'Before','T1_COR.npy')
            new_df.at[j,'before_T1_TRA'] = os.path.join(num,'Before','T1_TRA.npy')
            new_df.at[j,'before_T2_TRA'] = os.path.join(num,'Before','T2_TRA.npy')
            new_df.at[j,'after_T1_COR'] = os.path.join(num,'After','T1_COR.npy')
            new_df.at[j,'after_T1_TRA'] = os.path.join(num,'After','T1_TRA.npy')
            new_df.at[j,'after_T2_TRA'] = os.path.join(num,'After','T2_TRA.npy')
            j=j+1
        else:
            print(patient_path)

new_df.to_csv(new_csv_path, index=False)