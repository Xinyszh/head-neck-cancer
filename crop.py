import pydicom
import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
import pandas as pd
from PIL import Image
import nibabel


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
        output = output[:, :, z: z + size[2]]

    return output


roi_root = '/jhcnas1/xinyi/zhongshan_380/ROI/ROI/ROI-zhongshan'
before_root = '/jhcnas1/xinyi/zhongshan_380/before_370'
save_crop_root = '/jhcnas1/xinyi/zhongshan_380/crop_370'
save_concate_root = '/jhcnas1/xinyi/zhongshan_380/concate_370'
save_npy_root = '/jhcnas1/xinyi/zhongshan_380/before_npy'
before_df = pd.read_csv('/jhcnas1/xinyi/zhongshan_380/before_370.csv')

for i, row in before_df.iterrows():
    name = row['name']
    if (not pd.isna(row['before_T1_COR_W'])) and (not pd.isna(row['before_T1_TRA_W'])) and not (
            pd.isna(row['before_T2_TRA_W'])):
        ROI_path = os.path.join(roi_root, row['name'])
        ROI_sequences = os.listdir(ROI_path)
        if not os.path.isdir(os.path.join(save_concate_root, name)):
            os.makedirs(os.path.join(save_concate_root, name))
        if not os.path.isdir(os.path.join(save_crop_root, name)):
            os.makedirs(os.path.join(save_crop_root, name))
        if not os.path.isdir(os.path.join(save_npy_root, name)):
            os.makedirs(os.path.join(save_npy_root, name))
        for ROI_sequence in ROI_sequences:
            if 'COR' in ROI_sequence:
                if (not os.path.isfile(os.path.join(save_concate_root, row['name'], 'T1_COR.npy'))) or (
                not os.path.isfile(os.path.join(save_crop_root, row['name'], 'T1_COR.npy'))) or (
                not os.path.isfile(os.path.join(save_npy_root, row['name'], 'T1_COR.npy'))):
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

                    T1_COR_ROI_path = os.path.join(ROI_path, ROI_sequence)
                    T1_COR_ROI = sitk.ReadImage(T1_COR_ROI_path)
                    T1_COR_ROI_image = sitk.GetArrayFromImage(T1_COR_ROI)

                    if image_array.shape[0] == T1_COR_ROI_image.shape[0] and image_array.shape[1] == \
                            T1_COR_ROI_image.shape[
                                1] and image_array.shape[2] == T1_COR_ROI_image.shape[2]:
                        data_concate = np.concatenate([[image_array], [T1_COR_ROI_image]], axis=0)
                        data_concate = zoom(data_concate, [1, 45 / image_array.shape[0], 384 / image_array.shape[1],
                                                           384 / image_array.shape[2]])
                        data_concate = (data_concate - data_concate.mean()) / data_concate.std()

                        ss = np.sum(T1_COR_ROI_image, axis=(1, 2))
                        nzero_idx_1 = np.argmax(ss != 0)
                        nzero_idx_2 = len(ss) - np.argmax(ss[::-1] != 0)
                        ss = np.sum(T1_COR_ROI_image, axis=(0, 2))
                        nzero_idx_3 = np.argmax(ss != 0)
                        nzero_idx_4 = len(ss) - np.argmax(ss[::-1] != 0)
                        ss = np.sum(T1_COR_ROI_image, axis=(0, 1))
                        nzero_idx_5 = np.argmax(ss != 0)
                        nzero_idx_6 = len(ss) - np.argmax(ss[::-1] != 0)
                        data_crop = image_array[nzero_idx_1:nzero_idx_2, nzero_idx_3:nzero_idx_4,
                                    nzero_idx_5:nzero_idx_6]
                        data_crop = zoom(data_crop, [10 / data_crop.shape[0], 60 / data_crop.shape[1],
                                                     60 / data_crop.shape[2]])
                        data_crop = (data_crop - data_crop.mean()) / data_crop.std()
                        np.save(os.path.join(save_concate_root, row['name'], 'T1_COR.npy'), data_concate)

                        np.save(os.path.join(save_crop_root, row['name'], 'T1_COR.npy'), data_crop)
                    elif image_array.shape[0] == T1_COR_ROI_image.shape[0]//2 and image_array.shape[1] == \
                            T1_COR_ROI_image.shape[
                                1] and image_array.shape[2] == T1_COR_ROI_image.shape[2]:
                        T1_COR_ROI_image=T1_COR_ROI_image[0:image_array.shape[0],:,:]
                        data_concate = np.concatenate([[image_array], [T1_COR_ROI_image]], axis=0)
                        data_concate = zoom(data_concate, [1, 45 / image_array.shape[0], 384 / image_array.shape[1],
                                                           384 / image_array.shape[2]])
                        data_concate = (data_concate - data_concate.mean()) / data_concate.std()

                        ss = np.sum(T1_COR_ROI_image, axis=(1, 2))
                        nzero_idx_1 = np.argmax(ss != 0)
                        nzero_idx_2 = len(ss) - np.argmax(ss[::-1] != 0)
                        ss = np.sum(T1_COR_ROI_image, axis=(0, 2))
                        nzero_idx_3 = np.argmax(ss != 0)
                        nzero_idx_4 = len(ss) - np.argmax(ss[::-1] != 0)
                        ss = np.sum(T1_COR_ROI_image, axis=(0, 1))
                        nzero_idx_5 = np.argmax(ss != 0)
                        nzero_idx_6 = len(ss) - np.argmax(ss[::-1] != 0)
                        data_crop = image_array[nzero_idx_1:nzero_idx_2, nzero_idx_3:nzero_idx_4,
                                    nzero_idx_5:nzero_idx_6]
                        data_crop = zoom(data_crop, [10 / data_crop.shape[0], 60 / data_crop.shape[1],
                                                     60 / data_crop.shape[2]])
                        data_crop = (data_crop - data_crop.mean()) / data_crop.std()
                        np.save(os.path.join(save_concate_root, row['name'], 'T1_COR.npy'), data_concate)

                        np.save(os.path.join(save_crop_root, row['name'], 'T1_COR.npy'), data_crop)
                    else:
                        print(row['name'], T1_COR_ROI_path, 'not same size')

            elif 'T1' in ROI_sequence:
                if (not os.path.isfile(os.path.join(save_concate_root, row['name'], 'T1_TRA.npy'))) or (
                        not os.path.isfile(os.path.join(save_crop_root, row['name'], 'T1_TRA.npy'))) or (
                        not os.path.isfile(os.path.join(save_npy_root, row['name'], 'T1_TRA.npy'))):
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

                    T1_TRA_ROI_path = os.path.join(ROI_path, ROI_sequence)
                    T1_TRA_ROI = sitk.ReadImage(T1_TRA_ROI_path)
                    T1_TRA_ROI_image = sitk.GetArrayFromImage(T1_TRA_ROI)

                    if image_array.shape[0] == T1_TRA_ROI_image.shape[0] and image_array.shape[1] == \
                            T1_TRA_ROI_image.shape[
                                1] and image_array.shape[2] == T1_TRA_ROI_image.shape[2]:
                        data_concate = np.concatenate([[image_array], [T1_TRA_ROI_image]], axis=0)
                        data_concate = zoom(data_concate, [1, 70 / image_array.shape[0], 384 / image_array.shape[1],
                                                           384 / image_array.shape[2]])
                        data_concate = (data_concate - data_concate.mean()) / data_concate.std()

                        ss = np.sum(T1_TRA_ROI_image, axis=(1, 2))
                        nzero_idx_1 = np.argmax(ss != 0)
                        nzero_idx_2 = len(ss) - np.argmax(ss[::-1] != 0)
                        ss = np.sum(T1_TRA_ROI_image, axis=(0, 2))
                        nzero_idx_3 = np.argmax(ss != 0)
                        nzero_idx_4 = len(ss) - np.argmax(ss[::-1] != 0)
                        ss = np.sum(T1_TRA_ROI_image, axis=(0, 1))
                        nzero_idx_5 = np.argmax(ss != 0)
                        nzero_idx_6 = len(ss) - np.argmax(ss[::-1] != 0)
                        data_crop = image_array[nzero_idx_1:nzero_idx_2, nzero_idx_3:nzero_idx_4,
                                    nzero_idx_5:nzero_idx_6]
                        data_crop = zoom(data_crop, [15 / data_crop.shape[0], 150 / data_crop.shape[1],
                                                     150 / data_crop.shape[2]])
                        data_crop = (data_crop - data_crop.mean()) / data_crop.std()
                        np.save(os.path.join(save_concate_root, row['name'], 'T1_TRA.npy'), data_concate)

                        np.save(os.path.join(save_crop_root, row['name'], 'T1_TRA.npy'), data_crop)
                    elif image_array.shape[0] == T1_TRA_ROI_image.shape[0]//2 and image_array.shape[1] == \
                                T1_TRA_ROI_image.shape[
                                    1] and image_array.shape[2] == T1_TRA_ROI_image.shape[2]:
                            T1_TRA_ROI_image=T1_TRA_ROI_image[0:image_array.shape[0],::]
                            data_concate = np.concatenate([[image_array], [T1_TRA_ROI_image]], axis=0)
                            data_concate = zoom(data_concate, [1, 70 / image_array.shape[0], 384 / image_array.shape[1],
                                                               384 / image_array.shape[2]])
                            data_concate = (data_concate - data_concate.mean()) / data_concate.std()

                            ss = np.sum(T1_TRA_ROI_image, axis=(1, 2))
                            nzero_idx_1 = np.argmax(ss != 0)
                            nzero_idx_2 = len(ss) - np.argmax(ss[::-1] != 0)
                            ss = np.sum(T1_TRA_ROI_image, axis=(0, 2))
                            nzero_idx_3 = np.argmax(ss != 0)
                            nzero_idx_4 = len(ss) - np.argmax(ss[::-1] != 0)
                            ss = np.sum(T1_TRA_ROI_image, axis=(0, 1))
                            nzero_idx_5 = np.argmax(ss != 0)
                            nzero_idx_6 = len(ss) - np.argmax(ss[::-1] != 0)
                            data_crop = image_array[nzero_idx_1:nzero_idx_2, nzero_idx_3:nzero_idx_4,
                                        nzero_idx_5:nzero_idx_6]
                            data_crop = zoom(data_crop, [15 / data_crop.shape[0], 150 / data_crop.shape[1],
                                                         150 / data_crop.shape[2]])
                            data_crop = (data_crop - data_crop.mean()) / data_crop.std()
                            np.save(os.path.join(save_concate_root, row['name'], 'T1_TRA.npy'), data_concate)

                            np.save(os.path.join(save_crop_root, row['name'], 'T1_TRA.npy'), data_crop)
                    else:
                        print(row['name'], T1_TRA_ROI_path, 'not same size')

            elif 'T2' in ROI_sequence:
                if (not os.path.isfile(os.path.join(save_concate_root, row['name'], 'T2_TRA.npy'))) or (
                        not os.path.isfile(os.path.join(save_crop_root, row['name'], 'T2_TRA.npy'))) or (
                        not os.path.isfile(os.path.join(save_npy_root, row['name'], 'T2_TRA.npy'))):
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

                    T2_TRA_ROI_path = os.path.join(ROI_path, ROI_sequence)
                    T2_TRA_ROI = sitk.ReadImage(T2_TRA_ROI_path)
                    T2_TRA_ROI_image = sitk.GetArrayFromImage(T2_TRA_ROI)

                    if image_array.shape[0] == T2_TRA_ROI_image.shape[0] and image_array.shape[1] == \
                            T2_TRA_ROI_image.shape[
                                1] and image_array.shape[2] == T2_TRA_ROI_image.shape[2]:
                        data_concate = np.concatenate([[image_array], [T2_TRA_ROI_image]], axis=0)
                        data_concate = zoom(data_concate, [1, 36 / image_array.shape[0], 384 / image_array.shape[1],
                                                           384 / image_array.shape[2]])
                        data_concate = (data_concate - data_concate.mean()) / data_concate.std()

                        ss = np.sum(T2_TRA_ROI_image, axis=(1, 2))
                        nzero_idx_1 = np.argmax(ss != 0)
                        nzero_idx_2 = len(ss) - np.argmax(ss[::-1] != 0)
                        ss = np.sum(T2_TRA_ROI_image, axis=(0, 2))
                        nzero_idx_3 = np.argmax(ss != 0)
                        nzero_idx_4 = len(ss) - np.argmax(ss[::-1] != 0)
                        ss = np.sum(T2_TRA_ROI_image, axis=(0, 1))
                        nzero_idx_5 = np.argmax(ss != 0)
                        nzero_idx_6 = len(ss) - np.argmax(ss[::-1] != 0)
                        data_crop = image_array[nzero_idx_1:nzero_idx_2, nzero_idx_3:nzero_idx_4,
                                    nzero_idx_5:nzero_idx_6]
                        data_crop = zoom(data_crop, [10 / data_crop.shape[0], 75 / data_crop.shape[1],
                                                     75 / data_crop.shape[2]])
                        data_crop = (data_crop - data_crop.mean()) / data_crop.std()
                        np.save(os.path.join(save_concate_root, row['name'], 'T2_TRA.npy'), data_concate)

                        np.save(os.path.join(save_crop_root, row['name'], 'T2_TRA.npy'), data_crop)
                    elif image_array.shape[0] == T2_TRA_ROI_image.shape[0]//2 and image_array.shape[1] == \
                            T2_TRA_ROI_image.shape[
                                1] and image_array.shape[2] == T2_TRA_ROI_image.shape[2]:
                        T2_TRA_ROI_image=T2_TRA_ROI_image[0:image_array.shape[0],:,:]
                        data_concate = np.concatenate([[image_array], [T2_TRA_ROI_image]], axis=0)
                        data_concate = zoom(data_concate, [1, 36 / image_array.shape[0], 384 / image_array.shape[1],
                                                           384 / image_array.shape[2]])
                        data_concate = (data_concate - data_concate.mean()) / data_concate.std()

                        ss = np.sum(T2_TRA_ROI_image, axis=(1, 2))
                        nzero_idx_1 = np.argmax(ss != 0)
                        nzero_idx_2 = len(ss) - np.argmax(ss[::-1] != 0)
                        ss = np.sum(T2_TRA_ROI_image, axis=(0, 2))
                        nzero_idx_3 = np.argmax(ss != 0)
                        nzero_idx_4 = len(ss) - np.argmax(ss[::-1] != 0)
                        ss = np.sum(T2_TRA_ROI_image, axis=(0, 1))
                        nzero_idx_5 = np.argmax(ss != 0)
                        nzero_idx_6 = len(ss) - np.argmax(ss[::-1] != 0)
                        data_crop = image_array[nzero_idx_1:nzero_idx_2, nzero_idx_3:nzero_idx_4,
                                    nzero_idx_5:nzero_idx_6]
                        data_crop = zoom(data_crop, [10 / data_crop.shape[0], 75 / data_crop.shape[1],
                                                     75 / data_crop.shape[2]])
                        data_crop = (data_crop - data_crop.mean()) / data_crop.std()
                        np.save(os.path.join(save_concate_root, row['name'], 'T2_TRA.npy'), data_concate)

                        np.save(os.path.join(save_crop_root, row['name'], 'T2_TRA.npy'), data_crop)
                    else:
                        print(row['name'], T2_TRA_ROI_path, 'not same size')

            else:
                print(ROI_path, ROI_sequence)




