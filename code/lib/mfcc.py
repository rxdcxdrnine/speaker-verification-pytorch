import numpy as np
import torch
import torchaudio


class MFCC(torch.nn.Module):
    def __init__(self, args):
        super(MFCC, self).__init__()
        self.n_mfcc = args["n_mfcc"]
        self.delta = args["delta"]
        self.utt_regularization = args["utt_regularization"]

        self.sample_rate = args["sample_rate"]
        self.win_length = args["win_length"]
        self.hop_length = args["hop_length"]
        self.n_fft = args["n_fft"]
        self.n_mels = args["n_mels"]
        
    def forward(self, X):
        X = self._mfcc(X)
        if self.delta : X = self._delta(X)
        if self.utt_regularization : X = self._utt_regularization(X)

        return X
        
    def _utt_regularization(self, X):
        _, channel, _ = X.size()
        X_mean = X.mean(axis=-1).reshape(-1, channel, 1)
        X_std = X.std(axis=-1).reshape(-1, channel, 1)
        X = (X - X_mean) / X_std  
        return X
    
    def _mfcc(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        MFCC = torchaudio.transforms.MFCC(
            n_mfcc=self.n_mfcc,
            melkwargs={
                "win_length" : int(self.sample_rate * .001 * self.win_length),
                "hop_length" : int(self.sample_rate * .001 * self.hop_length),
                "n_fft" : self.n_fft,
                "n_mels" : self.n_mels}).to(device)
        X = MFCC(X)
        return X
            
            
    def _delta(self, X):
        compute_delta = torchaudio.transforms.ComputeDeltas()
        X_d1 = compute_delta(X)
        X_d2 = compute_delta(X_d1)
        X = torch.cat([X, X_d1, X_d2], dim=1)
        return X