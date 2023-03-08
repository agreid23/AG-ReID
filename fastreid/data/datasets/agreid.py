# encoding: utf-8

import glob
import os.path as osp
import re
import mat4py
import logging
import pandas as pd
import torch
from fastreid.data.datasets.bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY

__all__ = ['AGReID']


@DATASET_REGISTRY.register()
class AGReID(ImageDataset):

    dataset_dir = ''
    dataset_url = ''
    dataset_name = "agreid"

    def __init__(self, root='', **kwargs):
        self.logger = logging.getLogger('fastreid.' + __name__)

        self.root = 'datasets/AG-ReID'
        test_options = ['exp3_aerial_to_ground', 'exp6_ground_to_aerial']
        test_choice = test_options[0]

        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = self.dataset_dir
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')

        self.query_dir = osp.join(self.data_dir, test_choice + '/query')
        self.gallery_dir = osp.join(self.data_dir, test_choice + '/bounding_box_test')
        self.gallery_dir_add_on = osp.join(self.data_dir, 'gallery_add_on/' + test_choice)

        self.qut_attribute_path = osp.join(self.data_dir, 'qut_attribute_v8.mat')
        self.attribute_dict_all = self.generate_attribute_dict(self.qut_attribute_path, "qut_attribute")

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.gallery_dir_add_on,
            self.qut_attribute_path,
        ]

        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)

        query = self.process_dir(self.query_dir, is_train=False)

        gallery = self.process_dir(self.gallery_dir, is_train=False)
        gallery_dir_add_on = self.process_dir(self.gallery_dir_add_on, is_train=False)
        gallery = gallery + gallery_dir_add_on


        super(AGReID, self).__init__(train, query, gallery, **kwargs)

        # Required for create solution file later
        import pandas as pd
        query = {'query': query}
        gallery = {'gallery': gallery}
        query_df = pd.DataFrame(query)
        gallery_df = pd.DataFrame(gallery)
        # saving the data
        csv_dir = osp.join(self.data_dir)
        query_df.to_csv(csv_dir + '/' + 'query_' + test_choice + '.csv')
        gallery_df.to_csv(csv_dir + '/' + 'gallery_' + test_choice + '.csv')

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        data = []

        for img_path in img_paths:
            fname = osp.split(img_path)[-1]

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)

            camid, _ = pattern_camid.search(fname).groups()
            camid = int(camid)

            if is_train:
                pid = self.dataset_name + "_" + str(pid)
            data.append((img_path, pid, camid))

        return data

    def generate_attribute_dict(self, dir_path: str, dataset: str):

        mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
        mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)

        mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
        mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)

        mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
        mat_attribute = mat_attribute.drop(['image_index'], axis=1)

        self.key_attribute = list(mat_attribute.keys())

        h, w = mat_attribute.shape
        dict_attribute = dict()

        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[0:].astype(int)) * 2 - 3

        return dict_attribute

    def name_of_attribute(self):
        if self.key_attribute:
            print(self.key_attribute)
            return self.key_attribute
        else:
            assert False
