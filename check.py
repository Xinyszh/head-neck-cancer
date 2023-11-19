import pydicom
import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
import pandas as pd


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
        output = output[:, :, z:z + size[2]]

    return output


def tonpy(root, str, save_root):
    patients = os.listdir(root)
    patients.sort()

    for i, patient in enumerate(patients):
        patient_path = os.path.join(root, patient)

        before_path = os.path.join(patient_path, str)
        modalities = os.listdir(before_path)

        for modality in modalities:
            reader = sitk.ImageSeriesReader()

            if not os.path.isdir(os.path.join(save_root, patient,str)):
                os.makedirs(os.path.join(save_root, patient,str))

            if 'T1' in modality and ('Cor' in modality or 'cor' in modality and 'CC' in modality):
                T1COR_path = os.path.join(save_root, patient, str, 'T1_COR.npy')
                if not os.path.isfile(T1COR_path):
                    dicom_names = reader.GetGDCMSeriesFileNames(
                        os.path.join(before_path, modality))
                    reader.SetFileNames(dicom_names)
                    try:
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
                        # print(modality, newSize)
                    except:
                        print(patient, modality, 'error')

            elif 'T1' in modality and 'A' in modality:
                T1TRA_path = os.path.join(save_root, patient, str, 'T1_TRA.npy')
                if not os.path.isfile(T1TRA_path):
                    dicom_names = reader.GetGDCMSeriesFileNames(
                        os.path.join(before_path, modality))
                    reader.SetFileNames(dicom_names)
                    try:
                        image = reader.Execute()
                        originSpacing = image.GetSpacing()  # x, y, z
                        originSize = image.GetSize()
                        newSpacing = [1, 1, 1]
                        newSize = [
                            int(np.round(originSize[0] * originSpacing[0] / newSpacing[0])),
                            int(np.round(originSize[1] * originSpacing[1] / newSpacing[1])),
                            int(np.round(originSize[2] * originSpacing[2] / newSpacing[2]))
                        ]
                        # print(modality, newSize)
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
                        image_array = (image_array - image_array.mean()) / image_array.std()
                        print(image_array.shape)
                        np.save(T1TRA_path, image_array)
                        # print(modality, newSize)
                    except:
                        print(patient, modality, 'error')


            elif 'T2' in modality and ('Cor' in modality or 'cor' in modality):
                T2COR_path = os.path.join(save_root, patient, str, 'T2_COR.npy')
                if not os.path.isfile(T2COR_path):
                    dicom_names = reader.GetGDCMSeriesFileNames(
                        os.path.join(before_path, modality))
                    reader.SetFileNames(dicom_names)
                    try:
                        image = reader.Execute()
                        originSpacing = image.GetSpacing()  # x, y, z
                        originSize = image.GetSize()
                        newSpacing = [1, 1, 1]
                        newSize = [
                            int(np.round(originSize[0] * originSpacing[0] / newSpacing[0])),
                            int(np.round(originSize[1] * originSpacing[1] / newSpacing[1])),
                            int(np.round(originSize[2] * originSpacing[2] / newSpacing[2]))
                        ]
                        # print(modality, newSize)
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
                        np.save(T2COR_path, image_array)
                    except:
                        print(patient, modality, 'error')


            elif 'T2' in modality and 'A' in modality:
                T2TRA_path = os.path.join(save_root, patient, str, 'T2_TRA.npy')
                if not os.path.isfile(T2TRA_path):
                    dicom_names = reader.GetGDCMSeriesFileNames(
                        os.path.join(before_path, modality))
                    reader.SetFileNames(dicom_names)
                    try:
                        image = reader.Execute()
                        originSpacing = image.GetSpacing()  # x, y, z
                        originSize = image.GetSize()
                        newSpacing = [1, 1, 1]
                        newSize = [
                            int(np.round(originSize[0] * originSpacing[0] / newSpacing[0])),
                            int(np.round(originSize[1] * originSpacing[1] / newSpacing[1])),
                            int(np.round(originSize[2] * originSpacing[2] / newSpacing[2]))
                        ]
                        # print(modality, newSize)
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
                        image_array = (image_array - image_array.mean()) / image_array.std()
                        print(image_array.shape)
                        np.save(T2TRA_path, image_array)
                    except:
                        print(patient, modality, 'error')


            # elif 'DWI' in modality:
            #     DWI_path = os.path.join(save_root, patient, str, 'T1_COR.npy')
            #     if not os.path.isfile(T1COR_path):
            #     dicom_names = reader.GetGDCMSeriesFileNames(
            #         os.path.join(before_path, modality))
            #     reader.SetFileNames(dicom_names)
            #     try:
            #         image = reader.Execute()
            #         originSpacing = image.GetSpacing()  # x, y, z
            #         originSize = image.GetSize()
            #         newSpacing = [1, 1, 1]
            #         newSize = [
            #             int(np.round(originSize[0] * originSpacing[0] / newSpacing[0])),
            #             int(np.round(originSize[1] * originSpacing[1] / newSpacing[1])),
            #             int(np.round(originSize[2] * originSpacing[2] / newSpacing[2]))
            #         ]
            #         # print(modality, newSize)
            #         df.at[i, 'DWI'] = modality
            #     except:
            #         print(patient, modality, 'error')
            #
            #
            # elif 'ADC' in modality:
            #     dicom_names = reader.GetGDCMSeriesFileNames(
            #         os.path.join(before_path, modality))
            #     reader.SetFileNames(dicom_names)
            #     try:
            #         image = reader.Execute()
            #         originSpacing = image.GetSpacing()  # x, y, z
            #         originSize = image.GetSize()
            #         newSpacing = [1, 1, 1]
            #         newSize = [
            #             int(np.round(originSize[0] * originSpacing[0] / newSpacing[0])),
            #             int(np.round(originSize[1] * originSpacing[1] / newSpacing[1])),
            #             int(np.round(originSize[2] * originSpacing[2] / newSpacing[2]))
            #         ]
            #         # print(modality, newSize)
            #         df.at[i, 'ADC'] = modality
            #     except:
            #         print(patient, modality, 'error')

            else:
                print(patient, modality)


root = '/jhcnas1/xinyi/zhongliu_67/dcm'
save_root = '/jhcnas1/xinyi/zhongliu_67/new_resample'
tonpy(root, 'Before', save_root)
tonpy(root, 'After', save_root)


