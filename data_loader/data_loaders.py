from base import  BaseSystemDataLoader
from data_loader.systemDataset import SystemDataset, System_info

import numpy as np
import os

class SystemDataLoader(BaseSystemDataLoader):
    """
    molecular frames System data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, drop_last=True):
        self._data_dir =  os.path.normpath(data_dir) #./train
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._drop_last = drop_last
        self.__init_dataset(self._data_dir)

    def __init_dataset(self,data_dir):
        system_info = System_info(data_dir)
        self.subsystems = system_info.subsystems
        self.system_set_size = system_info.system_set_size
        self.system_paths = system_info.system_paths
        self.system_prop_list = system_info.get_system_prop()
        self.system_set_prop_dict_list = system_info.get_system_set_prop()
        
    def step(self,selected_system_index=None,selected_set_index=None):
        #Choose a system set at random
        if selected_system_index==None and selected_set_index==None:
            self.selected_system = np.random.choice(self.subsystems, size=1, replace=True, p=self.system_prop_list)[0]
            self.selected_set = np.random.choice(list(self.system_set_size[self.selected_system].keys()),size=1,p=self.system_set_prop_dict_list[self.selected_system])[0]
        elif selected_system_index != None and selected_set_index != None:
            if selected_system_index not in self.subsystems:
                raise ValueError(f"selected_set_index not in systems index")
            if selected_set_index not in self.system_set_size[selected_system_index].keys():
                raise ValueError(f"selected_set_index  not in systems_set index")
            self.selected_system = selected_system_index
            self.selected_set = selected_set_index
        else:
            raise(f"selected_system_index selected_set_index must be set at the same time")
        dataset = SystemDataset(self.system_paths[self.selected_system],self.selected_set,self.system_set_size[self.selected_system][self.selected_set])
        self.n_samples = len(dataset)
        dataloader = BaseSystemDataLoader(dataset=dataset,batch_size=self._batch_size,shuffle=self._shuffle, num_workers=self._num_workers, drop_last=self._drop_last)
        self.num_batches = len(dataloader)
        return dataloader

    def get_num_batches(self):
        return self.num_batches
    
    @property
    def batch_size(self):
        return self._batch_size

    def get_selected_system_set(self):
        system_set_path = os.path.join(self._data_dir,self.selected_system,self.selected_set)
        return system_set_path
    
    def change_system_prop(self, new_system_prop_dict: dict):
        assert self.system_prop_dict.keys() == new_system_prop_dict.keys(), "dict.keys mismatching"
        self.system_prop_dict = new_system_prop_dict
        self.system_prop_list = [self.system_prop_dict[i_system] for i_system in self.subsystems]
