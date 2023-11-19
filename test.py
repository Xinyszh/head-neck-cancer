import pandas as pd
import SimpleITK as sitk
import pandas as pd
import os
import numpy as np

# report_df = pd.read_csv('/jhcnas1/xinyi/zhongshan_380/zhongshan_report.csv')
# new_csv_path = '/jhcnas1/xinyi/zhongshan_380/after_370.csv'
# root = '/jhcnas1/xinyi/zhongshan_380/after_370'
# # 添加空列
# new_df = pd.DataFrame()
# new_df['name'] = ''
# new_df['num'] = ''
# new_df['PCR'] = ''
# new_df['before_T1_COR_W'] = ''
#
# new_df['before_T1_COR_size1'] = ''
# new_df['before_T1_COR_size2'] = ''
# new_df['before_T1_COR_size3'] = ''
# new_df['before_T1_TRA_W'] = ''
#
# new_df['before_T1_TRA_size1'] = ''
# new_df['before_T1_TRA_size2'] = ''
# new_df['before_T1_TRA_size3'] = ''
# new_df['before_T2_TRA_W'] = ''
#
# new_df['before_T2_TRA_size1'] = ''
# new_df['before_T2_TRA_size2'] = ''
# new_df['before_T2_TRA_size3'] = ''
#
# m = 0
# patients = os.listdir(root)
# patients.sort()
# for patient in patients:
#     num = str(patient).split('_')[1]
#     for i, row in report_df.iterrows():
#         if str(int(row['影像号'])) == num:
#             new_df.at[m, 'name'] = str(patient)
#             new_df.at[m, 'num'] = str(patient).split('_')[1]
#             new_df.at[m, 'PCR'] = row['PCR']
#             sequences = os.listdir(os.path.join(root, patient))
#             sequences.sort()
#             for sequence in sequences:
#                 if 'COR' in sequence or 'cor' in sequence:
#                     if 'DIXON' in sequence:
#                         if 'Vane' in sequence or '_W' in sequence:
#                             new_df.at[m, 'before_T1_COR_W'] = sequence
#                     else:
#                         new_df.at[m, 'before_T1_COR_W'] = sequence
#                 elif 'T2' in sequence or 'STIR' in sequence:
#                     if 'DIXON' in sequence:
#                         if 'Vane' in sequence or '_W' in sequence:
#                             new_df.at[m, 'before_T2_TRA_W'] = sequence
#                     else:
#                         new_df.at[m, 'before_T2_TRA_W'] = sequence
#                 else:
#                     if 'DIXON' in sequence:
#                         if 'Vane' in sequence or '_W' in sequence:
#                             new_df.at[m, 'before_T1_TRA_W'] = sequence
#                     else:
#                         new_df.at[m, 'before_T1_TRA_W'] = sequence
#
#             if not pd.isna(new_df.at[m, 'before_T1_COR_W']):
#                 reader = sitk.ImageSeriesReader()
#                 dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(root, patient, new_df.at[m, 'before_T1_COR_W']))
#                 reader.SetFileNames(dicom_names)
#                 image = reader.Execute()
#                 image_array = sitk.GetArrayFromImage(image)
#                 new_df.at[m, 'before_T1_COR_size1'] = image_array.shape[0]
#                 new_df.at[m, 'before_T1_COR_size2'] = image_array.shape[1]
#                 new_df.at[m, 'before_T1_COR_size3'] = image_array.shape[2]
#             if not pd.isna(new_df.at[m, 'before_T1_TRA_W']):
#                 reader = sitk.ImageSeriesReader()
#                 dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(root, patient, new_df.at[m, 'before_T1_TRA_W']))
#                 reader.SetFileNames(dicom_names)
#                 image = reader.Execute()
#                 image_array = sitk.GetArrayFromImage(image)
#                 new_df.at[m, 'before_T1_TRA_size1'] = image_array.shape[0]
#                 new_df.at[m, 'before_T1_TRA_size2'] = image_array.shape[1]
#                 new_df.at[m, 'before_T1_TRA_size3'] = image_array.shape[2]
#             if not pd.isna(new_df.at[m, 'before_T2_TRA_W']):
#                 reader = sitk.ImageSeriesReader()
#                 dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(root, patient, new_df.at[m, 'before_T2_TRA_W']))
#                 reader.SetFileNames(dicom_names)
#                 image = reader.Execute()
#                 image_array = sitk.GetArrayFromImage(image)
#                 new_df.at[m, 'before_T2_TRA_size1'] = image_array.shape[0]
#                 new_df.at[m, 'before_T2_TRA_size2'] = image_array.shape[1]
#                 new_df.at[m, 'before_T2_TRA_size3'] = image_array.shape[2]
#             m = m + 1
#             break
#
# df_new = new_df
# new_df.to_csv(new_csv_path, index=False)

report_df = pd.read_csv('/jhcnas1/xinyi/zhongshan_380/zhongshan_report.csv')
new_csv_path = '/jhcnas1/xinyi/zhongshan_380/roi_370.csv'
root = '/jhcnas1/xinyi/zhongshan_380/crop_370'
# 添加空列
new_df = pd.DataFrame()
new_df['name'] = ''
new_df['num'] = ''
new_df['PCR'] = ''
new_df['before_T1_COR'] = ''
new_df['before_T1_COR_size1'] = ''
new_df['before_T1_COR_size2'] = ''
new_df['before_T1_COR_size3'] = ''
new_df['before_T1_TRA'] = ''
new_df['before_T1_TRA_size1'] = ''
new_df['before_T1_TRA_size2'] = ''
new_df['before_T1_TRA_size3'] = ''
new_df['before_T2_TRA'] = ''
new_df['before_T2_TRA_size1'] = ''
new_df['before_T2_TRA_size2'] = ''
new_df['before_T2_TRA_size3'] = ''

m = 0
patients = os.listdir(root)
patients.sort()
for patient in patients:
    num = str(patient).split('_')[1]
    for i, row in report_df.iterrows():
        if str(int(row['影像号'])) == num:
            new_df.at[m, 'name'] = str(patient)
            new_df.at[m, 'num'] = str(patient).split('_')[1]
            new_df.at[m, 'PCR'] = row['PCR']
            sequences = os.listdir(os.path.join(root, patient))
            sequences.sort()
            if 'T1_COR.npy' in sequences:
                new_df.at[m, 'before_T1_COR'] = 'T1_COR.npy'
                data = np.load(os.path.join(root, patient, 'T1_COR.npy'))
                new_df.at[m, 'before_T1_COR_size1'] = data.shape[0]
                new_df.at[m, 'before_T1_COR_size2'] = data.shape[1]
                new_df.at[m, 'before_T1_COR_size3'] = data.shape[2]
            if 'T1_TRA.npy' in sequences:
                new_df.at[m, 'before_T1_TRA'] = 'T1_TRA.npy'
                data = np.load(os.path.join(root, patient, 'T1_TRA.npy'))
                new_df.at[m, 'before_T1_TRA_size1'] = data.shape[0]
                new_df.at[m, 'before_T1_TRA_size2'] = data.shape[1]
                new_df.at[m, 'before_T1_TRA_size3'] = data.shape[2]
            if 'T2_TRA.npy' in sequences:
                new_df.at[m, 'before_T2_TRA'] = 'T2_TRA.npy'
                data = np.load(os.path.join(root, patient, 'T2_TRA.npy'))
                new_df.at[m, 'before_T2_TRA_size1'] = data.shape[0]
                new_df.at[m, 'before_T2_TRA_size2'] = data.shape[1]
                new_df.at[m, 'before_T2_TRA_size3'] = data.shape[2]
            m = m + 1
            break

df_new = new_df
new_df.to_csv(new_csv_path, index=False)

# # 计算新数据量及划分比例
# n_new = len(df_new)
# ratio = [0.7, 0.1, 0.2]
# n_split = [int(r * n_new) for r in ratio]
#
# # 从新数据中进行采样
# new_train = df_new.sample(n_split[0])
# remainder = df_new.drop(new_train.index)
# new_val = remainder.sample(n_split[1])
# new_test = remainder.drop(new_val.index)
#
# # 输出更新后的数据集
# new_train.to_csv('/jhcnas1/xinyi/zhongshan_380/train.csv', index=False)
# new_val.to_csv('/jhcnas1/xinyi/zhongshan_380/val.csv', index=False)
# new_test.to_csv('/jhcnas1/xinyi/zhongshan_380/test.csv', index=False)