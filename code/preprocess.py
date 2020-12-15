import os
import glob
import random
import pickle
from tqdm import tqdm
import wave           

import numpy as np
import torch
import torchaudio


class DatConvertFile(torch.utils.data.Dataset):
    def __init__(self, path):
        self.filepaths = list(np.sort(glob.glob(path)))
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, index):
        dat = self.filepaths[index]
        f = open(dat, 'rb')
        length = os.path.getsize(dat) / 2
        
        wav_dst = dat.replace('/dat/', '/wav/')
        wav_dst = wav_dst.replace('.dat', '.wav')
        
        wav_dir = os.path.dirname(wav_dst)
        if not os.path.isdir(wav_dir):
            os.makedirs(wav_dir)

        wave_output = wave.open(wav_dst, 'w')
        wave_output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))

        wave_output.writeframes(f.read(int(2 * length)))
        wave_output.close()
        
        if os.path.getsize(wav_dst) < 100:
            os.remove(wav_dst)
        
        return 0


class WavFiles:
    speaker_labels = []
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.filepath_labels = {}
        self.nb_samples = []

    def collect_label(self):
        for filepath in tqdm(self.filepaths, ncols=60):
            filename = os.path.splitext(os.path.basename(filepath))[0]
            speaker_name = filename[4:9]
            
            if speaker_name in WavFiles.speaker_labels:
                self.filepath_labels[filepath] = WavFiles.speaker_labels.index(speaker_name)
            else:
                WavFiles.speaker_labels.append(speaker_name)
                self.filepath_labels[filepath] = WavFiles.speaker_labels.index(speaker_name)

    def get_nb_samples(self):
        for filepath in tqdm(self.filepaths, ncols=60):
            waveform, _ = torchaudio.load(filepath, normalization=True)
            self.nb_samples.append(waveform.size(1))
            
            
if __name__ == "__main__":    
    
    # # converting dat to wav
    # dat_convertset = DatConvertFile(path="/data/project_1/dat/*/*/*/*/*/*.dat")
    # dat_convertloader = torch.utils.data.DataLoader(dat_convertset, batch_size=1, num_workers=1)
    # for _ in tqdm(dat_convertloader, ncols=60): pass 
    
    
    # train/enroll/eval split
    np.random.seed(42)
    train_filepaths = []
    glob_paths = ["/data/project_1/wav/month/*/*/*/*/*.wav",
                  "/data/project_1/wav/week/*/*/*/*/*.wav"]
    for glob_path in glob_paths:
        train_filepaths.extend(glob.glob(glob_path))
    train_speaker_names = list(np.sort(list(set([filepath.split('/')[-5] for filepath in train_filepaths]))))
    train_filepaths = list(np.sort(train_filepaths))
    
    enroll_filepaths = []
    candid_filepaths = glob.glob("/data/project_1/wav/season/*/*/*/Digit4/*.wav")
    candid_speaker_names = list(np.sort(list(set([filepath.split('/')[-5] for filepath in candid_filepaths]))))
    for name in candid_speaker_names:
        for period in range(1, 5):
            if name not in train_speaker_names:
                candid_filepaths = glob.glob("/data/project_1/wav/season/%s/R%s/*/Digit4/*.wav" % (name, period))
                enroll_filepaths.append(random.choice(candid_filepaths))
    enroll_filepaths = list(np.sort(enroll_filepaths))
    
    eval_filepaths = []
    candid_filepaths = glob.glob("/data/project_1/wav/season/*/*/*/*/*.wav")
    for filepath in candid_filepaths:
        speaker_name = filepath.split('/')[-5]
        if (speaker_name not in train_speaker_names) and (filepath not in enroll_filepaths):
            eval_filepaths.append(filepath)
    eval_filepaths = list(np.sort(eval_filepaths))  
     
     
    # train/enroll/eval WavFiles       
    train_convert = WavFiles(filepaths=train_filepaths)
    enroll_convert = WavFiles(filepaths=enroll_filepaths)
    eval_convert = WavFiles(filepaths=eval_filepaths)
    
    train_convert.collect_label()
    enroll_convert.collect_label()
    eval_convert.collect_label()
    speaker_labels = WavFiles.speaker_labels
 
    train_convert.get_nb_samples()

    with open("/result/speaker_labels.txt", 'w') as f:
        for i in range(len(speaker_labels)):
            f.write(f"{speaker_labels[i]} {i}\n")

    # # save
    # with open("/result/convert.pkl", "wb") as file:
    #     pickle.dump(train_convert, file)
    #     pickle.dump(speaker_labels, file)