import os
import SimpleITK as sitk
from shutil import copy

root="/jhcnas1/xinyi/zhongshan_380/after_370"
patients= os.listdir(root)
for patient in patients:
    sequences=os.listdir(os.path.join(root,patient))
    sequences.sort()
    for sequence in sequences:
        if 'DIXON' in sequence and 'Vane' not in sequence and 'in' not in sequence and '_W' not in sequence:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(
                os.path.join(root,patient,sequence))

            if 'COR' or 'cor' in sequence :
                if len(dicom_names)>50:
                    mid = len(dicom_names) // 2
                    sequence_front=sequence+'_W'
                    sequence_behind=sequence+'_in'

                    if not os.path.isdir(os.path.join(root,patient,sequence_front)):
                        os.makedirs(os.path.join(root,patient,sequence_front))

                    if not os.path.isdir(os.path.join(root,patient,sequence_behind)):
                        os.makedirs(os.path.join(root,patient,sequence_behind))
                    # 前半部分
                    for i in range(mid):
                        fname = dicom_names[i]
                        copy(os.path.join(root,patient, sequence, fname), os.path.join(root, patient,sequence_front,os.path.basename(fname)))

                    # 后半部分
                    for i in range(mid, len(dicom_names)):
                        fname = dicom_names[i]
                        copy(os.path.join(root,patient, sequence, fname), os.path.join(root, patient,sequence_behind,os.path.basename(fname)))
            elif 'T2' in sequence:
                if len(dicom_names)>50:
                    mid = len(dicom_names) // 2
                    sequence_front = sequence + '_W'
                    sequence_behind = sequence + '_in'

                    if not os.path.isdir(os.path.join(root, patient, sequence_front)):
                        os.makedirs(os.path.join(root, patient, sequence_front))

                    if not os.path.isdir(os.path.join(root, patient, sequence_behind)):
                        os.makedirs(os.path.join(root, patient, sequence_behind))
                    # 前半部分
                    for i in range(mid):
                        fname = dicom_names[i]
                        copy(os.path.join(root, patient, sequence, fname),
                             os.path.join(root, patient, sequence_front, os.path.basename(fname)))

                    # 后半部分
                    for i in range(mid, len(dicom_names)):
                        fname = dicom_names[i]
                        copy(os.path.join(root, patient, sequence, fname),
                             os.path.join(root, patient, sequence_behind, os.path.basename(fname)))
            else:
                if len(dicom_names)>100:
                    mid = len(dicom_names) // 2
                    sequence_front = sequence + '_W'
                    sequence_behind = sequence + '_in'

                    if not os.path.isdir(os.path.join(root, patient, sequence_front)):
                        os.makedirs(os.path.join(root, patient, sequence_front))

                    if not os.path.isdir(os.path.join(root, patient, sequence_behind)):
                        os.makedirs(os.path.join(root, patient, sequence_behind))
                    # 前半部分
                    for i in range(mid):
                        fname = dicom_names[i]
                        copy(os.path.join(root, patient, sequence, fname),
                             os.path.join(root, patient, sequence_front, os.path.basename(fname)))

                    # 后半部分
                    for i in range(mid, len(dicom_names)):
                        fname = dicom_names[i]
                        copy(os.path.join(root, patient, sequence, fname),
                             os.path.join(root, patient, sequence_behind, os.path.basename(fname)))
