import os
import torch
import math
import numpy as np
from copy import deepcopy
from torch.utils.data import Sampler

# sampler used for meta-training
class meta_batchsampler(Sampler):
    
    def __init__(self,data_source,way,shots):

        self.way = way
        self.shots = shots

        class2id = {}

        for i,(image_path,class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id]=[]
            class2id[class_id].append(i)

        self.class2id = class2id


    def __iter__(self):

        temp_class2id = deepcopy(self.class2id)
        for class_id in temp_class2id:
            np.random.shuffle(temp_class2id[class_id])       

        while len(temp_class2id) >= self.way:

            id_list = []

            list_class_id = list(temp_class2id.keys())

            pcount = np.array([len(temp_class2id[class_id]) for class_id in list_class_id])

            batch_class_id = np.random.choice(list_class_id,size=self.way,replace=False,p=pcount/sum(pcount))

            for shot in self.shots:
                for class_id in batch_class_id:
                    for _ in range(shot):
                        id_list.append(temp_class2id[class_id].pop())

            for class_id in batch_class_id:
                if len(temp_class2id[class_id])<sum(self.shots):
                    temp_class2id.pop(class_id)

            yield id_list


class meta_batch_sampler(Sampler):
    def __init__(self, data_source, epoch_size, way, support_shot, query_shot, seed):
        self.data_source = data_source
        self.epoch_size = epoch_size
        self.way = way
        self.support_shot = support_shot
        self.query_shot = query_shot
        self.total_shot = self.support_shot + self.query_shot
        self.class2id = self.get_class_id()

        self.seed = seed
        self.rnd = np.random

    def get_class_id(self):
        class2id = {}
        for i, (image_path, class_id) in enumerate(self.data_source.imgs):
            if class_id not in class2id:
                class2id[class_id] = []
            class2id[class_id].append(i)

        return class2id

    def __iter__(self):
        if self.seed is not None:
            self.rnd = np.random.RandomState(self.seed)

        for i in range(self.epoch_size):
            support_index_list = []
            query_index_list = []
            selected_class = self.rnd.choice(list(self.class2id.keys()), size=self.way, replace=False)
            for class_index in selected_class:
                selected_index = self.rnd.choice(self.class2id[class_index], size=self.total_shot, replace=False)
                support_index_list.extend(selected_index[:self.support_shot])
                query_index_list.extend(selected_index[self.support_shot:self.total_shot])
            support_index_list.extend(query_index_list)

            yield support_index_list


# sampler used for meta-testing
class random_sampler(Sampler):

    def __init__(self,data_source,way,shot,query_shot=16,trial=1000):

        class2id = {}

        for i,(image_path,class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id]=[]
            class2id[class_id].append(i)

        self.class2id = class2id
        self.way = way
        self.shot = shot
        self.trial = trial
        self.query_shot = 16

    def __iter__(self):

        way = self.way
        shot = self.shot
        trial = self.trial
        query_shot = self.query_shot
        
        class2id = deepcopy(self.class2id)        
        list_class_id = list(class2id.keys())

        for i in range(trial):

            id_list = []
 
            np.random.shuffle(list_class_id)
            picked_class = list_class_id[:way]

            for cat in picked_class:
                np.random.shuffle(class2id[cat])
                
            for cat in picked_class:
                id_list.extend(class2id[cat][:shot])
            for cat in picked_class:
                id_list.extend(class2id[cat][shot:(shot+query_shot)])

            yield id_list