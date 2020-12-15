import os
import glob
import random
import pickle
from tqdm import tqdm

import numpy as np
import torch
import torchaudio


class WavFiles:
    speaker_labels = []
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.nb_samples = []
        self.filepath_labels = {}

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
        for filepath in self.filepaths:
            waveform, sample_rate = torchaudio.load(filepath, normalization=True)
            self.nb_samples.append(waveform.size(1))
                        

if __name__ == "__main__":

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
    
    
    # trial / speaker_model
    np.random.seed(42)
    with open("/result/trial.txt", 'w') as trial:
        for enroll_label in list(set(enroll_convert.filepath_labels.values())):
            enroll_speaker = speaker_labels[enroll_label]       
            for i, eval_label in enumerate(eval_convert.filepath_labels.values()):
                if enroll_label == eval_label:
                    start = i
                    break
            
            for i, eval_label in enumerate(eval_convert.filepath_labels.values()):
                if i > start and enroll_label != eval_label:
                    stop = i
                    break
                else:
                    stop = i + 1
                    
            filepaths = eval_convert.filepaths[start:stop]
            filepaths = np.random.choice(filepaths, size=1500)
            for filepath in filepaths:
                trial.write(f"{enroll_speaker} {filepath} 1\n")
                            
            filepaths = eval_convert.filepaths[:start] + eval_convert.filepaths[stop:]            
            filepaths = np.random.choice(filepaths, size=1500)
            for filepath in filepaths:
                trial.write(f"{enroll_speaker} {filepath} 0\n")
                
    with open("/result/speaker_model.txt", 'w') as speaker_model:
        for filepath in enroll_convert.filepaths:
            filename = os.path.splitext(os.path.basename(filepath))[0]
            speaker_name = filename[4:9]
            
            speaker_model.write(f"{speaker_name} {filepath}\n")