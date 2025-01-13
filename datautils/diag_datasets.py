"""
Domainbed Datasets
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import numpy as np
from scipy.io import loadmat
import pandas as pd
from datautils.sig_process import sig_segmentation, dataset_transform

DATASETS = [
    'SichuanU',
    'SichuanU_sym02',
    'SichuanU_asym02',

    'CranfieldU',
    'CranfieldU_sym02',
    'CranfieldU_asym02',

]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        """
        __getitem__() is a magic method in Python, which when used in a class,
        allows its instances to use the [] (indexer) operators. Say x is an
        instance of this class, then x[i] is roughly equivalent to type(x).__getitem__(x, i).
        """
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class SichuanU(MultipleDomainDataset):
    ENVIRONMENTS = ['1200rpm', '1500rpm', '1800rpm']

    def __init__(self, root, device, test_env_ids, label_noise_type, label_noise_rate):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        # -----------------------------------------------------------
        self.filename = {'p1': None,      # part 1: class type
                         'p2': '_1_ZZH_', # part 2: 表示主箱体上方设定载荷为3N*M时采集的垂直向振动数据
                         'p3': None,      # part 3: speed
                         'p4': '_Ns290000Fs5K.mat'} # part 4: tail
        # -----------------------------------------------------------
        # -- sample points
        self.seg_len = 2000   # len of each sample
        self.len_total = 280000
        self.class_name_list = ['ZC', 'TD', 'XD']  # ['ZC','QL', 'TL', 'XL1']
        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = ['1200r', '1500r', '1800r']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []
        # -----------------------------------------------------------
        #-- label noise transition relations
        assert label_noise_rate <=0.5 # clean labels should be dominant
        # -- symmetric transition matrix
        self.tran_sym   = {0: {0: 1 - label_noise_rate,   1: label_noise_rate * 0.5, 2: label_noise_rate * 0.5},
                           1: {0: label_noise_rate * 0.5, 1: 1 - label_noise_rate,   2: label_noise_rate * 0.5},
                           2: {0: label_noise_rate * 0.5, 1: label_noise_rate * 0.5, 2: 1 - label_noise_rate}}
        #-- asymmetric noise transition matirx
        self.tran_asym  = {0: {0: 1-label_noise_rate,     1: label_noise_rate * 0.2, 2: label_noise_rate * 0.8},
                           1: {0: label_noise_rate * 0.8, 1: 1-label_noise_rate,     2: label_noise_rate * 0.2},
                           2: {0: label_noise_rate * 0.2, 1: label_noise_rate * 0.8, 2: 1-label_noise_rate}}

        if label_noise_type in ['sym', 'asym'] and label_noise_rate>0: # symmetric or asymmetric
            self.tran_matrix =  self.tran_sym if label_noise_type == 'sym' else self.tran_asym
        elif label_noise_type in ['nsfree'] and label_noise_rate == 0: # noise free
            self.tran_matrix = None
        else:
            raise ValueError('label_noise_rate and label_noise_type are conflict')

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            tran_matrix_temp = None if env_id in test_env_ids else self.tran_matrix
            data, labels = self.get_samples(root, env_name)
            self.datasets.append(dataset_transform(data, labels, self.input_shape, self.num_classes,
                                                   device,  tran_matrix_temp))

    def get_samples(self, root, env_name):
        data_seg_all = []
        label_seg_all = []
        for cls, lab in zip(self.class_name_list, self.class_list):
            self.filename['p1'] = cls  # fault condition
            self.filename['p3'] = env_name
            file_str = self.filename['p1'] + self.filename['p2'] + self.filename['p3'] + self.filename['p4']
            file_path = os.path.join(root, file_str)
            data = loadmat(file_path)['data'].squeeze()[:]
            data_temp, label_temp = sig_segmentation(data, label=lab, seg_len = self.seg_len,
                                                     start = 0, stop=self.len_total)
            data_seg_all.extend(data_temp)
            label_seg_all.extend(label_temp)

        return data_seg_all, label_seg_all


class SichuanU_sym02(SichuanU):
    ENVIRONMENTS = ['1200rpm', '1500rpm', '1800rpm']

    def __init__(self, root, device, test_env_ids, label_noise_type, label_noise_rate):
        label_noise_rate = 0.2
        label_noise_type = 'sym'
        super().__init__(root, device, test_env_ids, label_noise_type, label_noise_rate)

class SichuanU_asym02(SichuanU):
    ENVIRONMENTS = ['1200rpm', '1500rpm', '1800rpm']

    def __init__(self, root, device, test_env_ids, label_noise_type, label_noise_rate):
        label_noise_rate = 0.2
        label_noise_type = 'asym'
        super().__init__(root, device, test_env_ids, label_noise_type, label_noise_rate)


class CranfieldU(MultipleDomainDataset):
    ENVIRONMENTS = ['20kgf', '40kgf', '-40kgf']

    def __init__(self, root, device, test_env_ids, label_noise_type, label_noise_rate):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        # -----------------------------------------------------------
        # self.filename = {'p1': None,      # part 1: class type
        #                  'p2': ('sin', 'trap'), # part 2: motion profile
        #                  'p3': ('1st', '2nd'),  # part 3: severity for back and lub faults
        #                  'p4': ('3rd', '4th'), # part 4: severity for point fault
        #                  'p5': (str(i+1) for i in range(10))} # part 5: repeat
        # -----------------------------------------------------------
        # -- sample points
        self.seg_len = 1000   # len of each sample
        self.len_total = 4000
        self.class_name_dict = {'back':{'Backlash1.mat':'1st', 'Backlash2.mat':'2nd'},
                                'lub':{'LackLubrication1.mat':'1st', 'LackLubrication2.mat':'2nd'},
                                'point':{'Spalling3.mat':'3rd', 'Spalling4.mat':'4th'}}  # ['ZC','QL', 'TL', 'XL1']

        self.class_list = [i for i in range(len(self.class_name_dict))]
        self.environments = ['20kg', '40kg', 'neg40kg']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []
        # -----------------------------------------------------------
        #-- label noise transition relations
        assert label_noise_rate <=0.5 # clean labels should be dominant
        # -- symmetric transition matrix
        self.tran_sym   = {0: {0: 1 - label_noise_rate,   1: label_noise_rate * 0.5, 2: label_noise_rate * 0.5},
                           1: {0: label_noise_rate * 0.5, 1: 1 - label_noise_rate,   2: label_noise_rate * 0.5},
                           2: {0: label_noise_rate * 0.5, 1: label_noise_rate * 0.5, 2: 1 - label_noise_rate}}

        #-- asymmetric noise transition matirx
        self.tran_asym  = {0: {0: 1-label_noise_rate,     1: label_noise_rate * 0.2, 2: label_noise_rate * 0.8},
                           1: {0: label_noise_rate * 0.8, 1: 1-label_noise_rate,     2: label_noise_rate * 0.2},
                           2: {0: label_noise_rate * 0.2, 1: label_noise_rate * 0.8, 2: 1-label_noise_rate}}

        if label_noise_type in ['sym', 'asym'] and label_noise_rate>0: # symmetric or asymmetric
            self.tran_matrix =  self.tran_sym if label_noise_type == 'sym' else self.tran_asym
        elif label_noise_type in ['nsfree'] and label_noise_rate == 0: # noise free
            self.tran_matrix = None
        else:
            raise ValueError('label_noise_rate and label_noise_type are conflict')

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            tran_matrix_temp = None if env_id in test_env_ids else self.tran_matrix
            data, labels = self.get_samples(root, env_name)
            self.datasets.append(dataset_transform(data, labels, self.input_shape, self.num_classes,
                                                   device,  tran_matrix_temp))

    def get_samples(self, root, env_name):
        data_seg_all = []
        label_seg_all = []
        file_idxs = [str(i+1) for i in range(10)]
        for cls, lab in zip(self.class_name_dict, self.class_list):
            for cls_file in self.class_name_dict[cls]:
                file_path = os.path.join(root, cls_file)
                data_dict = loadmat(file_path)
                for motion in ['sin', 'trap']:
                    severity = self.class_name_dict[cls][cls_file]
                    for file_id in file_idxs:
                        file_str = cls + motion+ severity + env_name + file_id
                        if file_str == 'backtrap1st40kg2': # cope with missing values
                            data_dict['backtrap1st40kg2'] = (data_dict['backtrap1st40kg1'] + data_dict['backtrap1st40kg3']) / 2
                        data = np.concatenate((data_dict[file_str][:,1], data_dict[file_str][:,2]), axis=0)
                        data_temp, label_temp = sig_segmentation(data, label=lab, seg_len = self.seg_len,
                                                                 start = 0, stop=self.len_total)
                        data_seg_all.extend(data_temp)
                        label_seg_all.extend(label_temp)
        return data_seg_all, label_seg_all


class CranfieldU_sym02(CranfieldU):
    ENVIRONMENTS = ['20kgf', '40kgf', '-40kgf']

    def __init__(self, root, device, test_env_ids, label_noise_type, label_noise_rate):
        label_noise_rate = 0.2
        label_noise_type = 'sym'
        super().__init__(root, device, test_env_ids, label_noise_type, label_noise_rate)

class CranfieldU_asym02(CranfieldU):
    ENVIRONMENTS = ['20kgf', '40kgf', '-40kgf']

    def __init__(self, root, device, test_env_ids, label_noise_type, label_noise_rate):
        label_noise_rate = 0.2
        label_noise_type = 'asym'
        super().__init__(root, device, test_env_ids, label_noise_type, label_noise_rate)



if __name__ == '__main__':

    root = r'..\datasets\CranfieldU'
    device = 'cuda'
    test_env_ids = [1]
    label_noise_type = 'asym'
    label_noise_rate = 0.2
    data_class = CranfieldU_asym02(root, device, test_env_ids, label_noise_type, label_noise_rate)
