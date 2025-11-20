import os
import sys
import collections

import torch
from torch.utils.data import Dataset

import numpy as np

from utils.chemical_symbols import to_atomic_numbers
from utils import convert_path

DTYPE = torch.float32

class SystemDataset(Dataset):
    def __init__(self, system_path, selected_set, system_set_size):
        super(SystemDataset, self).__init__()
        self.system_path = system_path  # ./train/data_01
        self.selected_set = selected_set # 'set.000'
        self.selected_data_set_path = os.path.join(self.system_path, self.selected_set)
        self.system_set_size =  system_set_size
        self.__load()


    def __getitem__(self, index):
        coord = self.coord_data[index]
        number = self.number_data[index]
        force = self.force_data[index]
        energy = self.energy_data[index]
        cell = self.cell_data[index]
        return (cell, number, coord, force, energy)

    def __len__(self):
        return self.system_set_size
    
    def __load(self):
        """
        load system set data
        """
        type_path = os.path.join(self.system_path, 'type.raw')
        
        type_exists = os.path.exists(type_path)
        number_exists = os.path.exists(os.path.join(self.selected_data_set_path, 'number.npy'))
        if type_exists and number_exists:
            print("Error: Both 'type.raw' and 'number.npy' exist. Please check your data setup.")
            sys.exit(1)
        elif type_exists:
            zi_list = np.loadtxt(type_path)
            number = torch.tensor(zi_list,dtype=torch.long)
            self.number_data = number.expand(self.system_set_size, number.shape[0])
        elif number_exists:
            self.number_data =  torch.from_numpy(np.load(os.path.join(self.selected_data_set_path, 'number.npy'))).to(torch.long)
        else:
            print("Error: 'type.raw' or 'number.npy' not found. Please check your data setup.")
            sys.exit(1)

        self.coord_data = torch.from_numpy(np.load(os.path.join(self.selected_data_set_path, 'coord.npy'))).to(DTYPE)
        self.force_data = torch.from_numpy(np.load(os.path.join(self.selected_data_set_path, 'force.npy'))).to(DTYPE)
        self.energy_data = torch.from_numpy(np.load(os.path.join(self.selected_data_set_path, 'energy.npy'))).unsqueeze(-1).to(DTYPE)

        if os.path.exists(os.path.join(self.selected_data_set_path, 'box.npy')):
            self.cell_data = torch.from_numpy(np.load(os.path.join(self.selected_data_set_path, 'box.npy'))).to(DTYPE)
        else:
            self.cell_data = torch.full((self.coord_data.shape[0], 9),float('nan')).to(DTYPE)

class System_info():

    def __init__(self, data_root):
        self.data_root = convert_path(data_root)  # ./train
        self.subsystems = list(sorted(os.listdir(self.data_root)))
        system_set_size = collections.OrderedDict()
        system_size = collections.OrderedDict()
        system_paths = {}
        total_system_size = 0
        for i_system in self.subsystems:  # i_system: data_01
            set_names = [
                i for i in os.listdir(os.path.join(self.data_root, i_system))
                if 'set' in i
            ]
            set_names = sorted(set_names, key=lambda x: int(x.split('.')[1]))
            system_set_size[i_system] = collections.OrderedDict()
            system_size[i_system] = 0
            system_paths[i_system] = os.path.join(self.data_root, i_system)
            for i_set in set_names:
                coor_path = os.path.join(self.data_root, i_system, i_set,
                                         'coord.npy')
                coor_data = np.load(coor_path)
                coor_data_size = coor_data.shape[0]
                total_system_size += coor_data_size
                system_set_size[i_system][i_set] = coor_data_size
                system_size[i_system] += coor_data_size

        self.system_set_size = system_set_size
        self.system_size = system_size
        self.system_paths = system_paths
        self.total_system_size = total_system_size

    def get_system_prop(self):
        system_prop_list = []
        for i_system, size in self.system_size.items():
            system_prop_list.append(size / self.total_system_size)
        return system_prop_list
    def get_system_set_prop(self):
        system_set_prop_dict_list = {}
        for i_system, sets in self.system_set_size.items():
            system_set_prop_dict_list[i_system] = []
            for set_key,set_size in sets.items():
                system_set_prop_dict_list[i_system].append(set_size / self.system_size[i_system])
        return system_set_prop_dict_list
