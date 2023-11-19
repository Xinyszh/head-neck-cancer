from torchvision import transforms as T
from data.mri import T1COR_Dataset, T1COR_concate_Dataset, T1COR_zhongliu_Dataset, T1COR_before_Dataset,T1TRA_before_Dataset,T2TRA_before_Dataset,T1COR_after_Dataset,T1TRA_after_Dataset,T2TRA_after_Dataset,T1TRA_Dataset, T1TRA_concate_Dataset, T1TRA_zhongliu_Dataset, \
    T2TRA_Dataset, T2TRA_concate_Dataset,T2TRA_zhongliu_Dataset
from data.augmentation import Inserter, Flipper
from torch.utils.data import Dataset

transforms = {
    'ZhongShan': {
        "train": Flipper(),
        "val": None,
        "test": None,
    }

}


def get_dataset(modal, dataset_split, transform_split):
    before_root = '/jhcnas1/xinyi/zhongshan_380/before_npy'
    after_root = '/jhcnas1/xinyi/zhongshan_380/after_npy'
    root = '/jhcnas1/xinyi/zhongshan_380/'
    transform_t1 = transforms['ZhongShan'][transform_split]
    if modal == 'T1_COR':
        dataset = T1COR_Dataset(
            split=dataset_split,
            root=root,
            before_root=before_root,
            after_root=after_root,
            transform=transform_t1)
    elif modal == 'T1_COR_concate':
        dataset = T1COR_concate_Dataset(
            split=dataset_split,
            root=root,
            before_root=before_root,
            after_root=after_root,
            transform=transform_t1)
    elif modal == 'T1_COR_before':
        dataset = T1COR_before_Dataset(
            split=dataset_split,
            root=root,
            before_root=before_root,
            transform=transform_t1)
    elif modal == 'T1_COR_after':
        dataset = T1COR_after_Dataset(
            split=dataset_split,
            root=root,
            after_root=after_root,
            transform=transform_t1)
    elif modal == 'T1_TRA_after':
        dataset = T1TRA_after_Dataset(
            split=dataset_split,
            root=root,
            after_root=after_root,
            transform=transform_t1)
    elif modal == 'T2_TRA_after':
        dataset = T2TRA_after_Dataset(
            split=dataset_split,
            root=root,
            after_root=after_root,
            transform=transform_t1)
    elif modal == 'T1_COR_zhongliu':
        dataset = T1COR_zhongliu_Dataset(
            root='/jhcnas1/xinyi/zhongliu_67',
            csv_root='/jhcnas1/xinyi/zhongliu_67/patient.csv',
            transform=transform_t1)
    elif modal == 'T1_TRA':
        dataset = T1TRA_Dataset(
            split=dataset_split,
            root=root,
            before_root=before_root,
            after_root=after_root,
            transform=transform_t1)
    elif modal == 'T1_TRA_concate':
        dataset = T1TRA_concate_Dataset(
            split=dataset_split,
            root=root,
            before_root=before_root,
            after_root=after_root,
            transform=transform_t1)
    elif modal == 'T1_TRA_before':
        dataset = T1TRA_before_Dataset(
            split=dataset_split,
            root=root,
            before_root=after_root,
            transform=transform_t1)
    elif modal == 'T1_TRA_zhongliu':
        dataset = T1TRA_zhongliu_Dataset(
            root='/jhcnas1/xinyi/zhongliu_67',
            csv_root='/jhcnas1/xinyi/zhongliu_67/patient.csv',
            transform=transform_t1)
    elif modal == 'T2_TRA':
        dataset = T2TRA_Dataset(
            split=dataset_split,
            root=root,
            before_root=before_root,
            after_root=after_root,
            transform=transform_t1)
    elif modal == 'T2_TRA_concate':
        dataset = T2TRA_concate_Dataset(
            split=dataset_split,
            root=root,
            before_root=before_root,
            after_root=after_root,
            transform=transform_t1)
    elif modal == 'T2_TRA_before':
        dataset = T2TRA_before_Dataset(
            split=dataset_split,
            root=root,
            before_root=before_root,
            transform=transform_t1)
    elif modal == 'T2_TRA_zhongliu':
        dataset = T2TRA_zhongliu_Dataset(
            root='/jhcnas1/xinyi/zhongliu_67',
            csv_root='/jhcnas1/xinyi/zhongliu_67/patient.csv',
            transform=transform_t1)

    return dataset
