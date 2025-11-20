from typing import Any
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

DTYPE = torch.float32

class mse_loss(nn.Module):
    def __init__(self,start_pref_e, limit_pref_e, start_pref_f, limit_pref_f, decay_step_size, epochs, device):
        super(mse_loss,self).__init__()
        self.loss_function = nn.MSELoss(reduction="mean")
        self.decay_step_size = decay_step_size
        n = epochs / decay_step_size
        self.e_alpha = torch.tensor(pow((limit_pref_e/start_pref_e),(1/n)),dtype=DTYPE,device=device)
        self.f_alpha = torch.tensor(pow((limit_pref_f/start_pref_f),(1/n)),dtype=DTYPE,device=device)
        self.e_pref = torch.tensor(start_pref_e,dtype=DTYPE,device=device)
        self.f_pref = torch.tensor(start_pref_f,dtype=DTYPE,device=device)
    
    def forward(self, predict_energy, energy, predict_force, force):
        energy_loss = self.loss_function(predict_energy, energy)
        force_loss = self.loss_function(predict_force, force)
        loss = self.e_pref * energy_loss + self.f_pref * force_loss
        return loss
    
    def step(self,epoch):
        """
        Update energy_prefactor,force_prefactor according the current epoch index
        """
        if epoch % self.decay_step_size == 0:
            self.e_pref = self.__compute_new_pref(self.e_pref, self.e_alpha)
            self.f_pref = self.__compute_new_pref(self.f_pref, self.f_alpha)
    
    def state_dict(self):
        state_dict = {"decay_step_size":self.decay_step_size,"pref_e":self.e_pref,"pref_f":self.f_pref,"e_alpha":self.e_alpha,"f_alpha":self.f_alpha}
        return state_dict
    
    def from_restart(self, pref_data:dict):
        self.decay_step_size = pref_data["decay_step_size"]
        self.e_pref  = pref_data["pref_e"]
        self.f_pref  = pref_data["pref_f"]
        self.e_alpha = pref_data["e_alpha"]
        self.f_alpha = pref_data["f_alpha"]
    
    def __compute_new_pref(self,pref,alpha):
        return pref * alpha