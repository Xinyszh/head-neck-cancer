import pandas as pd
import os
import SimpleITK as sitk
from scipy.ndimage import zoom
import numpy as np
before_root = '/jhcnas1/xinyi/zhongshan_380/before_370'
save_npy_root = '/jhcnas1/xinyi/zhongshan_380/before_npy'
before_df = pd.read_csv('/jhcnas1/xinyi/zhongshan_380/before_370.csv')

for i, row in before_df.iterrows():
    name = row['name']
    if (not pd.isna(row['before_T1_COR_W'])) and (not pd.isna(row['before_T1_TRA_W'])) and not (
            pd.isna(row['before_T2_TRA_W'])):

        if not os.path.isdir(os.path.join(save_npy_root, name)):
            os.makedirs(os.path.join(save_npy_root, name))

        if not os.path.isfile(os.path.join(save_npy_root, row['name'], 'T1_COR.npy')):
            T1_COR_path = os.path.join(before_root, row['name'], row['before_T1_COR_W'])
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(T1_COR_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            image_array = sitk.GetArrayFromImage(image)

            data_npy = zoom(image_array,
                            [45 / image_array.shape[0], 384 / image_array.shape[1], 384 / image_array.shape[2]])
            data_npy = (data_npy - data_npy.mean()) / data_npy.std()
            np.save(os.path.join(save_npy_root, row['name'], 'T1_COR.npy'), data_npy)

        if not os.path.isfile(os.path.join(save_npy_root, row['name'], 'T1_TRA.npy')):
            T1_TRA_path = os.path.join(before_root, row['name'], row['before_T1_TRA_W'])
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(T1_TRA_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            image_array = sitk.GetArrayFromImage(image)

            data_npy = zoom(image_array,
                            [70 / image_array.shape[0], 384 / image_array.shape[1], 384 / image_array.shape[2]])
            data_npy = (data_npy - data_npy.mean()) / data_npy.std()
            np.save(os.path.join(save_npy_root, row['name'], 'T1_TRA.npy'), data_npy)

        if not os.path.isfile(os.path.join(save_npy_root, row['name'], 'T2_TRA.npy')):
            T2_TRA_path = os.path.join(before_root, row['name'], row['before_T2_TRA_W'])
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(T2_TRA_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            image_array = sitk.GetArrayFromImage(image)

            data_npy = zoom(image_array,
                            [36 / image_array.shape[0], 384 / image_array.shape[1], 384 / image_array.shape[2]])
            data_npy = (data_npy - data_npy.mean()) / data_npy.std()
            np.save(os.path.join(save_npy_root, row['name'], 'T2_TRA.npy'), data_npy)