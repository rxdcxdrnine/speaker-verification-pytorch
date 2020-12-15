import numpy as np
import torch
import torchaudio


class MelSpectrogram(torch.nn.Module):
    def __init__(self, args, local_cep_mvn=False, log_mel=True):
        super(MelSpectrogram, self).__init__()
        self.sample_rate = args["sample_rate"]
        self.win_length = args["win_length"]
        self.hop_length = args["hop_length"]
        self.n_fft = args["n_fft"]
        self.n_mels = args["n_mels"]

        self.local_cep_mvn = local_cep_mvn
        self.log_mel = log_mel
        self.epsilon = 1e-6
        
    def forward(self, X):
        X = self._mel_spectrogram(X)
        if self.log_mel : X = torch.log(X + self.epsilon)
        if self.local_cep_mvn : X = self._local_cep_mvn(X)

        return X

    def _mel_spectrogram(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            win_length=int(self.sample_rate * .001 * self.win_length),
            hop_length=int(self.sample_rate * .001 * self.hop_length),
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            window_fn=torch.hamming_window,
            ).to(device)
        X = mel_spectrogram(X)
        
        X = X.transpose(1, 2)
        return X

    def _local_cep_mvn(self, X):
        X_mean = X.mean(dim=-1, keepdim=True)
        X_std = X.std(dim=-1, keepdim=True)
        X_std[X_std < .001] = .001
        result = (X - X_mean) / X_std
        
        return result
    