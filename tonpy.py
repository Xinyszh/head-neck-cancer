import pydicom
import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
import pandas as pd


def tonpy(root, save_root, csv_root):
    df = pd.read_csv(csv_root)
    for i, row in df.iterrows():
        if not pd.isna(row).all():
            patient = str(row['patient'])
            T1_COR = str(row['T1_COR'])
            T1_SAG = str(row['T1_SAG'])
            T1_TRA = str(row['T1_TRA'])
            T2_COR = str(row['T2_COR'])
            T2_TRA = str(row['T2_TRA'])

            if not os.path.isdir(os.path.join(save_root, patient)):
                os.makedirs(os.path.join(save_root, patient))

            T1COR_path = os.path.join(save_root, patient, 'T1_COR.npy')
            if not os.path.isfile(T1COR_path):
                if os.path.isdir(os.path.join(root, patient, T1_COR)):
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(
                        os.path.join(root, patient, T1_COR))
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    image_array = sitk.GetArrayFromImage(image)  # z, y, x
                    origin = image.GetOrigin()  # x, y, z
                    spacing = image.GetSpacing()  # x, y, z
                    print(image_array.shape)

                    image_array = zoom(image_array,
                                       (140 / image_array.shape[0], 384 / image_array.shape[1], 384 / image_array.shape[2]),
                                       order=1)
                    image_array = (image_array - image_array.mean()) / image_array.std()
                    np.save(T1COR_path, image_array)
                else:
                    print(patient,'T1_COR')

            T1SAG_path = os.path.join(save_root, patient, 'T1_SAG.npy')
            if not os.path.isfile(T1SAG_path):
                if os.path.isdir(os.path.join(root, patient, T1_SAG)):
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(
                        os.path.join(root, patient, T1_SAG))
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    image_array = sitk.GetArrayFromImage(image)  # z, y, x
                    print(image_array.shape)
                    image_array = zoom(image_array,
                                       (140 / image_array.shape[0], 384 / image_array.shape[1], 384 / image_array.shape[2]),
                                       order=1)
                    image_array = (image_array - image_array.mean()) / image_array.std()
                    np.save(T1SAG_path, image_array)
                else:
                    print(patient,'T1_SAG')

            T1TRA_path = os.path.join(save_root, patient, 'T1_TRA.npy')
            if not os.path.isfile(T1TRA_path):
                if os.path.isdir(os.path.join(root, patient, T1_TRA)):
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(
                        os.path.join(root, patient, T1_TRA))
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    image_array = sitk.GetArrayFromImage(image)  # z, y, x
                    print(image_array.shape)
                    image_array = zoom(image_array,
                                       (140 / image_array.shape[0], 384 / image_array.shape[1], 384 / image_array.shape[2]),
                                       order=1)
                    image_array = (image_array - image_array.mean()) / image_array.std()
                    np.save(T1TRA_path, image_array)
                else:
                    print(patient,'T1_TRA')

            T2COR_path = os.path.join(save_root, patient, 'T2_COR.npy')
            if not os.path.isfile(T2COR_path):
                if os.path.isdir(os.path.join(root, patient, T2_COR)):
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(
                        os.path.join(root, patient, T2_COR))
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    image_array = sitk.GetArrayFromImage(image)  # z, y, x
                    print(image_array.shape)
                    image_array = zoom(image_array,
                                       (140 / image_array.shape[0], 384 / image_array.shape[1], 384 / image_array.shape[2]),
                                       order=1)
                    image_array = (image_array - image_array.mean()) / image_array.std()
                    np.save(T2COR_path, image_array)
                else:
                    print(patient,'T2_COR')

            T2TRA_path = os.path.join(save_root, patient, 'T2_TRA.npy')
            if not os.path.isfile(T2TRA_path):
                if os.path.isdir(os.path.join(root, patient, T2_TRA)):
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(
                        os.path.join(root, patient, T2_TRA))
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    image_array = sitk.GetArrayFromImage(image)  # z, y, x
                    print(image_array.shape)
                    image_array = zoom(image_array,
                                       (140 / image_array.shape[0], 384 / image_array.shape[1], 384 / image_array.shape[2]),
                                       order=1)
                    image_array = (image_array - image_array.mean()) / image_array.std()
                    np.save(T2TRA_path, image_array)
                else:
                    print(patient,'T2_TRA')


root = '/jhcnas1/xinyi/zhongshan_380/before/before'
save_root = '/jhcnas1/xinyi/zhongshan_380/before/npy'
csv_root = '/jhcnas1/xinyi/zhongshan_380/before.csv'
tonpy(root, save_root, csv_root)
root = '/jhcnas1/xinyi/zhongshan_380/after/after'
save_root = '/jhcnas1/xinyi/zhongshan_380/after/npy'
csv_root = '/jhcnas1/xinyi/zhongshan_380/after.csv'
tonpy(root, save_root, csv_root)

