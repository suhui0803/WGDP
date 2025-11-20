import numpy as np
import torch
import torch.nn.functional as F
from base import BaseTrainer
from utils import MetricTracker
import os
import time

DTYPE = torch.float32

class Trainer(BaseTrainer):
    """
    DPTrainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, lr_scheduler, config, device,
                 train_data_loader, valid_data_loader=None):
        super().__init__(model, criterion, metric_ftns, optimizer, lr_scheduler, config, device)
        self.config = config
        self.device = device
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.train_log_step =  int(np.sqrt(train_data_loader.batch_size))
        self.valid_log_step =  int(np.sqrt(valid_data_loader.batch_size))

        self.train_metrics = MetricTracker(
            'loss_trn', *[m.__name__+'_e_trn' for m in self.metric_ftns], 
            *[m.__name__+'_f_trn' for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss_val', *[m.__name__+'_e_val' for m in self.metric_ftns], 
            *[m.__name__+'_f_val' for m in self.metric_ftns], writer=self.writer)

        self.select_system_update_interval = 1

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        training_start_time = time.time()
        
        self.model.train()
        
        self.train_metrics.reset()
        if epoch % self.select_system_update_interval == 0:
            train_loader = self.train_data_loader.step()
        train_dataset_path = self.train_data_loader.get_selected_system_set()
        train_batches = self.train_data_loader.get_num_batches()

        for batch_idx, data in enumerate(train_loader):
            boxs, numbers, coords, force, energy = data
            boxs = boxs.to(self.device)
            numbers = numbers.to(self.device)
            coords = coords.to(self.device)
            force = force.to(self.device)
            energy = energy.to(self.device)
            boxs.requires_grad = False
            numbers.requires_grad = False
            coords.requires_grad = True
            self.optimizer.zero_grad()
            predict_energy = self.model(boxs, numbers, coords)
            
            dE_dxyz = torch.autograd.grad(
                predict_energy,
                coords,
                grad_outputs=torch.ones_like(predict_energy,dtype=DTYPE,device=self.device),
                retain_graph=True,
                create_graph=True,
            )[0]

            predict_force = -dE_dxyz

            loss = self.criterion(predict_energy, energy, predict_force, force)

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * train_batches + batch_idx)
            self.train_metrics.update('loss_trn', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__+'_e_trn', met(predict_energy, energy))
                self.train_metrics.update(met.__name__+'_f_trn', met(predict_force, force))

            if batch_idx % self.train_log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._train_progress(batch_idx,train_batches),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == train_batches:
                break
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
                
        log = self.train_metrics.result()
        training_end_time = time.time()
        training_time = training_end_time - training_start_time

        log.update({"lr":self.lr_scheduler.get_last_lr()[0]})
        log.update({"train_dataset":train_dataset_path})
        log.update({"training_time":training_time})

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{k: v for k, v in val_log.items()})
        
        self.criterion.step(epoch)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        validating_start_time = time.time()
        if epoch % self.select_system_update_interval == 0:
            valid_loader = self.valid_data_loader.step()
        valid_dataset_path = self.valid_data_loader.get_selected_system_set()
        valid_batches = self.valid_data_loader.get_num_batches()

        self.model.eval()
        self.valid_metrics.reset()

        for batch_idx, data in enumerate(valid_loader):
            boxs, numbers, coords, force, energy = data
            boxs = boxs.to(self.device)
            numbers = numbers.to(self.device)
            coords = coords.to(self.device)
            force = force.to(self.device)
            energy = energy.to(self.device)
            boxs.requires_grad = False
            numbers.requires_grad = False
            coords.requires_grad = True
            self.model.zero_grad()
            predict_energy = self.model(boxs, numbers, coords)
            
            dE_dxyz = torch.autograd.grad(
                predict_energy,
                coords,
                grad_outputs=torch.ones_like(predict_energy,dtype=DTYPE,device=self.device),
                retain_graph=True,
                create_graph=True,
            )[0]
            predict_force = -dE_dxyz
            
            loss = self.criterion(predict_energy, energy, predict_force, force)

            self.writer.set_step((epoch - 1) * valid_batches + batch_idx, 'valid')
            self.valid_metrics.update('loss_val', loss.item())
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__+'_e_val', met(predict_energy, energy))
                self.valid_metrics.update(met.__name__+'_f_val', met(predict_force, force))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx % self.valid_log_step == 0:
                self.logger.debug('Valid Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._valid_progress(batch_idx,valid_batches),
                    loss.item()))
        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        validating_end_time = time.time()
        validating_time = validating_end_time - validating_start_time
        log = self.valid_metrics.result()
        log.update({"valid_dataset":valid_dataset_path})
        log.update({"validating_time":validating_time})

        return log

    def _train_progress(self, batch_idx, train_batches):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = train_batches
        return base.format(current, total, 100.0 * current / total)
    
    def _valid_progress(self, batch_idx, valid_batches):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.valid_data_loader, 'n_samples'):
            current = batch_idx * self.valid_data_loader.batch_size
            total = self.valid_data_loader.n_samples
        else:
            current = batch_idx
            total = valid_batches
        return base.format(current, total, 100.0 * current / total)
    
    
