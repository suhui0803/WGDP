import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import os

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def convert_path(path: str) -> str:
    return path.replace(r'\/'.replace(os.sep, ''), os.sep)

class Lcurve_log():
    def __init__(self,lcurve_file_path):
        if os.path.exists(lcurve_file_path):
            os.remove(lcurve_file_path)
        self.lcurve_file_path = lcurve_file_path
        self.lcurve_file = open(self.lcurve_file_path, mode='a')
        self.lcurve_file.writelines(
            '#  step      rmse_val    rmse_trn    rmse_e_val  rmse_e_trn    rmse_f_val  rmse_f_trn          lr\n')
        self.lcurve_file.flush()
        self.close()

    def write_logdata(self,log:dict):
        with open(self.lcurve_file_path,  mode='a') as fp:
            line = "{:>7d}{:>14.2e}{:>12.2e}{:>14.2e}{:>12.2e}{:>14.2e}{:>12.2e}{:>12.2e}\n".format(
                    log["epoch"],log["loss_val"],log["loss_trn"],log["rmse_e_val"],log["rmse_e_trn"],
                    log["rmse_f_val"],log["rmse_f_trn"],log["lr"])
            fp.writelines(line)
            fp.flush()
        self.close()
    
    def close(self):
        self.lcurve_file.close()