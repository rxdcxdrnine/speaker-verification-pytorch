import random
import numpy as np
import torch
import torchaudio
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from models.se_resnet import *
from models.convnet import *

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, list_filepath, labels, nb_samp, args, mode, pre_emp=True):
        self.list_filepath = list_filepath
        self.labels = labels
        self.nb_samp = int(np.mean(nb_samp) * args["scaling"])
        self.mode = mode
        self.div = args["div"]
        self.pre_emp = pre_emp

        
    def __len__(self):
        return len(self.list_filepath)
    
    def _cut(self, X):        
        if self.mode == "train": # train
            if X.size(0) < self.nb_samp:
                nb_dup = int(np.ceil(self.nb_samp / X.size(0)))
                X = X.repeat(nb_dup)
            
            margin = X.size(0) - self.nb_samp
            st_idx = random.randint(0, margin)
            X = X[st_idx:(st_idx + self.nb_samp)]

            return X

        else: # eval / enroll
            if X.size(0) <= self.nb_samp:
                nb_dup = int(np.ceil(self.nb_samp / X.size(0)))
                X = X.repeat(nb_dup)

            list_X = []
            if self.div == 1:
                margin = X.size(0) - self.nb_samp
                st_idx = random.randint(0, margin)
                list_X.append(X[st_idx:(st_idx + self.nb_samp)])
                
            else:
                step = int((X.size(0) - self.nb_samp) / (self.div - 1))
                for i in range(self.div):
                    if i == 0:
                        list_X.append(X[:self.nb_samp])
                    elif i < self.div - 1:
                        list_X.append(X[i*step : i*step + self.nb_samp])
                    else:
                        list_X.append(X[-self.nb_samp:])

            X = torch.stack(list_X)
            return X
        
    def _pre_emphasis(self, X):
        return X[1:] - 0.97 * X[:-1]
    
    def __getitem__(self, index):
        filepath = self.list_filepath[index]
        waveform, _ = torchaudio.load(filepath, normalization=True) # load_wav
        
        X = waveform.squeeze(0)
        if self.pre_emp: X = self._pre_emphasis(X)
        X = self._cut(X)

        if self.mode == "train" or self.mode == "enroll":
            y = self.labels[filepath]    
            return X, y
                
        elif self.mode == "eval":
            return X
    
    
    
class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.se_resnet = se_resnet18()
        self.convnet = convnet()

        self.fc_in = torch.nn.Linear(256, 256)
        self.fc_out = torch.nn.Linear(256, 199)
        
    def forward(self, X, train=True):
        # the number of channel = 1, rank of the channel should be intergrated into batch rank.
        X = X.unsqueeze(1)
        out = self.convnet(X)
        out = self.fc_in(out)
        
        if train:
            out = self.fc_out(out)
        return out
     
    
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def EER(y, y_pred):
    
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    EER = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return EER