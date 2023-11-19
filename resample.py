import pydicom
import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
import pandas as pd
from PIL import Image


def resize_crop_inserter(img, size):
    w, h, d = img.shape
    ww, hh, dd = size
    output = img


    if w <= size[0]:
        # 小于size,做inserter
        tmp = np.zeros((ww, h, d)).astype(np.float32)
        x = np.random.randint(0, ww - w + 1)
        tmp[x:x + w, :, :] = output
        output = tmp
    else:
        x = np.random.randint(w - size[0])
        output = output[x:x + ww, :, :]


    if h <= size[1]:
        # 小于size,做inserter
        tmp = np.zeros((ww, hh, d)).astype(np.float32)
        y = np.random.randint(0, hh - h + 1)
        tmp[:, y:y + h, :] = output
        output = tmp
    else:
        y = np.random.randint(h - hh)
        output = output[:, y:y + hh, :]


    if d <= size[2]:
        # 小于size,做inserter
        tmp = np.zeros((ww, hh, dd)).astype(np.float32)
        z = np.random.randint(0, dd - d + 1)
        tmp[:, :, z:z + d] = output
        output = tmp
    else:
        z = np.random.randint(d - size[2])
        output = output[:, :,z: z + size[2]]

    return output


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
                    originSpacing = image.GetSpacing()  # x, y, z
                    originSize = image.GetSize()
                    newSpacing = [1, 1, 1]
                    newSize = [
                        int(np.round(originSize[0] * originSpacing[0] / newSpacing[0])),
                        int(np.round(originSize[1] * originSpacing[1] / newSpacing[1])),
                        int(np.round(originSize[2] * originSpacing[2] / newSpacing[2]))
                    ]
                    resample = sitk.ResampleImageFilter()
                    resample.SetOutputSpacing(newSpacing)
                    # 设置original
                    resample.SetOutputOrigin(image.GetOrigin())
                    # 设置方向
                    resample.SetOutputDirection(image.GetDirection())
                    resample.SetSize(newSize)
                    # 设置插值方式
                    resample.SetInterpolator(sitk.sitkNearestNeighbor)
                    # 设置transform
                    resample.SetTransform(sitk.Euler3DTransform())
                    # 默认像素值   resample.SetDefaultPixelValue(image.GetPixelIDValue())
                    new_image = resample.Execute(image)
                    image_array = sitk.GetArrayFromImage(new_image)
                    image_array = resize_crop_inserter(image_array, [150, 280, 280])
                    image_array = (image_array - image_array.mean()) / image_array.std()
                    print(image_array.shape)
                    np.save(T1COR_path, image_array)
                else:
                    print(patient, 'T1_COR')

            T1SAG_path = os.path.join(save_root, patient, 'T1_SAG.npy')
            if not os.path.isfile(T1SAG_path):
                if os.path.isdir(os.path.join(root, patient, T1_SAG)):
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(
                        os.path.join(root, patient, T1_SAG))
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    originSpacing = image.GetSpacing()  # x, y, z
                    originSize = image.GetSize()
                    newSpacing = [1, 1, 1]
                    newSize = [
                        int(np.round(originSize[0] * originSpacing[0] / newSpacing[0])),
                        int(np.round(originSize[1] * originSpacing[1] / newSpacing[1])),
                        int(np.round(originSize[2] * originSpacing[2] / newSpacing[2]))
                    ]
                    resample = sitk.ResampleImageFilter()
                    resample.SetOutputSpacing(newSpacing)
                    # 设置original
                    resample.SetOutputOrigin(image.GetOrigin())
                    # 设置方向
                    resample.SetOutputDirection(image.GetDirection())
                    resample.SetSize(newSize)
                    # 设置插值方式
                    resample.SetInterpolator(sitk.sitkNearestNeighbor)
                    # 设置transform
                    resample.SetTransform(sitk.Euler3DTransform())
                    # 默认像素值   resample.SetDefaultPixelValue(image.GetPixelIDValue())
                    new_image = resample.Execute(image)
                    image_array = sitk.GetArrayFromImage(new_image)
                    image_array = resize_crop_inserter(image_array, [150, 280, 280])
                    print(image_array.shape)
                    image_array = (image_array - image_array.mean()) / image_array.std()
                    np.save(T1SAG_path, image_array)
                else:
                    print(patient, 'T1_SAG')

            T1TRA_path = os.path.join(save_root, patient, 'T1_TRA.npy')
            if not os.path.isfile(T1TRA_path):
                if os.path.isdir(os.path.join(root, patient, T1_TRA)):
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(
                        os.path.join(root, patient, T1_TRA))
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    originSpacing = image.GetSpacing()  # x, y, z
                    originSize = image.GetSize()
                    newSpacing = [1, 1, 1]
                    newSize = [
                        int(np.round(originSize[0] * originSpacing[0] / newSpacing[0])),
                        int(np.round(originSize[1] * originSpacing[1] / newSpacing[1])),
                        int(np.round(originSize[2] * originSpacing[2] / newSpacing[2]))
                    ]
                    resample = sitk.ResampleImageFilter()
                    resample.SetOutputSpacing(newSpacing)
                    # 设置original
                    resample.SetOutputOrigin(image.GetOrigin())
                    # 设置方向
                    resample.SetOutputDirection(image.GetDirection())
                    resample.SetSize(newSize)
                    # 设置插值方式
                    resample.SetInterpolator(sitk.sitkNearestNeighbor)
                    # 设置transform
                    resample.SetTransform(sitk.Euler3DTransform())
                    # 默认像素值   resample.SetDefaultPixelValue(image.GetPixelIDValue())
                    new_image = resample.Execute(image)
                    image_array = sitk.GetArrayFromImage(new_image)
                    image_array = resize_crop_inserter(image_array, [250, 280, 280])
                    print(image_array.shape)
                    image_array = (image_array - image_array.mean()) / image_array.std()
                    np.save(T1TRA_path, image_array)
                else:
                    print(patient, 'T1_TRA')

            T2COR_path = os.path.join(save_root, patient, 'T2_COR.npy')
            if not os.path.isfile(T2COR_path):
                if os.path.isdir(os.path.join(root, patient, T2_COR)):
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(
                        os.path.join(root, patient, T2_COR))
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    originSpacing = image.GetSpacing()  # x, y, z
                    originSize = image.GetSize()
                    newSpacing = [1, 1, 1]
                    newSize = [
                        int(np.round(originSize[0] * originSpacing[0] / newSpacing[0])),
                        int(np.round(originSize[1] * originSpacing[1] / newSpacing[1])),
                        int(np.round(originSize[2] * originSpacing[2] / newSpacing[2]))
                    ]
                    resample = sitk.ResampleImageFilter()
                    resample.SetOutputSpacing(newSpacing)
                    # 设置original
                    resample.SetOutputOrigin(image.GetOrigin())
                    # 设置方向
                    resample.SetOutputDirection(image.GetDirection())
                    resample.SetSize(newSize)
                    # 设置插值方式
                    resample.SetInterpolator(sitk.sitkNearestNeighbor)
                    # 设置transform
                    resample.SetTransform(sitk.Euler3DTransform())
                    # 默认像素值   resample.SetDefaultPixelValue(image.GetPixelIDValue())
                    new_image = resample.Execute(image)
                    image_array = sitk.GetArrayFromImage(new_image)
                    image_array = resize_crop_inserter(image_array, [150, 280, 280])
                    print(image_array.shape)
                    image_array = (image_array - image_array.mean()) / image_array.std()
                    np.save(T2COR_path, image_array)
                else:
                    print(patient, 'T2_COR')

            T2TRA_path = os.path.join(save_root, patient, 'T2_TRA.npy')
            if not os.path.isfile(T2TRA_path):
                if os.path.isdir(os.path.join(root, patient, T2_TRA)):
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(
                        os.path.join(root, patient, T2_TRA))
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    originSpacing = image.GetSpacing()  # x, y, z
                    originSize = image.GetSize()
                    newSpacing = [1, 1, 1]
                    newSize = [
                        int(np.round(originSize[0] * originSpacing[0] / newSpacing[0])),
                        int(np.round(originSize[1] * originSpacing[1] / newSpacing[1])),
                        int(np.round(originSize[2] * originSpacing[2] / newSpacing[2]))
                    ]
                    resample = sitk.ResampleImageFilter()
                    resample.SetOutputSpacing(newSpacing)
                    # 设置original
                    resample.SetOutputOrigin(image.GetOrigin())
                    # 设置方向
                    resample.SetOutputDirection(image.GetDirection())
                    resample.SetSize(newSize)
                    # 设置插值方式
                    resample.SetInterpolator(sitk.sitkNearestNeighbor)
                    # 设置transform
                    resample.SetTransform(sitk.Euler3DTransform())
                    # 默认像素值   resample.SetDefaultPixelValue(image.GetPixelIDValue())
                    new_image = resample.Execute(image)
                    image_array = sitk.GetArrayFromImage(new_image)
                    image_array = resize_crop_inserter(image_array, [250, 280, 280])
                    print(image_array.shape)
                    image_array = (image_array - image_array.mean()) / image_array.std()
                    np.save(T2TRA_path, image_array)
                else:
                    print(patient, 'T2_TRA')


root = '/jhcnas1/xinyi/zhongshan_380/before/before'
save_root = '/jhcnas1/xinyi/zhongshan_380/before/new_resample'
csv_root = '/jhcnas1/xinyi/zhongshan_380/before.csv'
tonpy(root, save_root, csv_root)
root = '/jhcnas1/xinyi/zhongshan_380/after/after'
save_root = '/jhcnas1/xinyi/zhongshan_380/after/new_resample'
csv_root = '/jhcnas1/xinyi/zhongshan_380/after.csv'
tonpy(root, save_root, csv_root)

