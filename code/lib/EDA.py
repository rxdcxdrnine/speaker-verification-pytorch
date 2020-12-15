import glob
import numpy as np
import torchaudio
import torch
import os


glob_paths = ["/data/project_1/dat/month/*/*/*/*/*.dat",
              "/data/project_1/dat/week/*/*/*/*/*.dat"]
train_filepaths = []
for glob_path in glob_paths:
    train_filepaths.extend(glob.glob(glob_path))
test_filepaths = glob.glob("/data/project_1/dat/season/*/*/*/*/*.dat")
print("number of .dat train filepaths :", len(train_filepaths))
print("number of .dat test filepaths :", len(test_filepaths), "\n")


glob_paths = ["/data/project_1/wav/month/*/*/*/*/*.wav",
              "/data/project_1/wav/week/*/*/*/*/*.wav"]
train_filepaths = []
for glob_path in glob_paths:
    train_filepaths.extend(glob.glob(glob_path))
test_filepaths = glob.glob("/data/project_1/wav/season/*/*/*/*/*.wav")
print("number of .wav train filepaths :", len(train_filepaths))
print("number of .wav train filepaths :", len(test_filepaths), "\n")

def nb_speakers(filepaths):
    speaker_names = []
    for filepath in filepaths:    
        file_ID = os.path.splitext(os.path.basename(filepath))[0]
        speaker_name = file_ID[4:9]
        if speaker_name not in speaker_names:
            speaker_names.append(speaker_name)
    return len(speaker_names)
print("number of train speakers :", nb_speakers(train_filepaths))
print("number of test speakers :", nb_speakers(test_filepaths))

"""
dat/Month : 99 + JJH00 = 100
dat/Week : 98 + JJH00 + HSI00 = 100
dat/Season : 49 + HSI00 = 50

All = dat/Month + dat/Week + dat/Season = 246 + JJH00 + HSI00 = 248
Train = dat/Month + dat/Week = 197 + JJH00 + HSI00 = 199
Test = dat/Season = 49 (EXCEPT HSI00)
"""

