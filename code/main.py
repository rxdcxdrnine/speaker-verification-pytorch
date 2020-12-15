import os
import pickle
from tqdm import tqdm
import datetime

import numpy as np
import torch
import torchaudio
import neptune

from lib.mfcc import *
from lib.mel_spectrogram import *
from preprocess import *
from loader import *


def timestamp(filename):
    mtime = os.path.getmtime(filename)
    KST = datetime.timezone(datetime.timedelta(hours=9))
    time = datetime.datetime.fromtimestamp(mtime, tz=KST)
    
    return str(time)


def main(args):

    ## load data
    seed_num = 42
    random.seed(42)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
            
    with open("/result/convert.pkl", "rb") as file:
        train_convert = pickle.load(file)
        speaker_labels = pickle.load(file)

    utt_num = 4
    enroll_filepaths = []
    enroll_labels = {}
    with open("/result/speaker_model.txt", 'r') as speaker_model:
        lines = speaker_model.readlines()
        for line in lines:
            filepath = line.split(' ')[1].strip()
            enroll_filepaths.append(filepath)

            name = line.split(' ')[0].strip()
            enroll_labels[filepath] = speaker_labels.index(name)

    trial_eval = []
    trial_enroll = []
    trial_corrects = []
    with open("/result/trial.txt", 'r') as trial:
        lines = trial.readlines()
        for line in lines:
            name = line.split(' ')[0].strip()
            enroll_label = speaker_labels.index(name)
            trial_enroll.append(enroll_label)

            filepath = line.split(' ')[1].strip()
            trial_eval.append(filepath)
            
            correct = line.split(' ')[2].strip()
            trial_corrects.append(float(correct))

    train_dataset = CustomDataset(list_filepath=train_convert.filepaths, 
                                labels=train_convert.filepath_labels,
                                nb_samp=train_convert.nb_samples,
                                args=args,
                                mode="train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=13) # batch_size 

    enroll_dataset = CustomDataset(list_filepath=enroll_filepaths,
                                labels=enroll_labels,
                                nb_samp=train_convert.nb_samples,
                                args=args,
                                mode="enroll")
    enroll_dataloader = torch.utils.data.DataLoader(enroll_dataset, batch_size=utt_num, shuffle=False, num_workers=13)

    eval_dataset = CustomDataset(list_filepath=trial_eval,
                                labels=None,
                                nb_samp=train_convert.nb_samples,
                                args=args,
                                mode="eval")
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=13)

    device = torch.device(f"cuda:{args['GPU_NUM']}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print("Current cuda device :", torch.cuda.current_device(), "\n")

    if device == "cuda":
        torch.cuda.manual_seed(seed_num)

    mel_spectrogram = MelSpectrogram(args).to(device)
    model = Model(args).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if args["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args["learning_rate"],
                                    weight_decay = args["weight_decay"],
                                    momentum=0.95,
                                    nesterov=True)
                                            
    elif args["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args["learning_rate"],
                                    weight_decay = args["weight_decay"],
                                    amsgrad=True)

    if args["neptune"]:
        neptune.init("changgu/ETRI",
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiN2JhMTk0Y2QtNTc0Ni00ZGVmLTliMjItMmViY2M5YzQ1NzA5In0=")
        neptune.create_experiment(name="ETRI DB",
                                params={"scaling" : args["scaling"],
                                        "div" : args["div"],
                                        "batch_size" : args["batch_size"],

                                        "n_mfcc" : args["n_mfcc"],
                                        ""

                                        "n_fft" : args["n_fft"],
                                        "n_mels" : args["n_mels"],
                                        "win_length" : args["win_length"],
                                        "hop_length" : args["hop_length"],
                                        
                                        "learning_rate" : args["learning_rate"],
                                        "weight_decay" : args["weight_decay"],
                                        "optimizer" : args["optimizer"]})


    # ## global mean variance
    # if os.path.isfile("/result/global.npz"):
    #     global_load = np.load("/result/global.npz")
    #     global_mean = global_load["mean"]
    #     global_std = global_load["std"]
    
    # else:
    #     global_spec = []
    #     for X_train, _ in tqdm(train_dataloader, ncols=60):
    #         X_train = X_train.to(device)
    #         spec_train = mel_spectrogram(X_train)
            
    #         spec_train = spec_train.reshape(-1, args["n_mels"])  ## mfcc; n_mfcc
    #         global_spec.append(spec_train.cpu().detach().numpy())
            
    #     global_spec = np.vstack(global_spec)
    #     global_mean = np.mean(global_spec, axis=0, keepdims=True)
    #     global_std = np.std(global_spec, axis=0, keepdims=True)
    #     np.savez("/result/global.npz", mean=global_mean, std=global_std)
        
    # global_mean = torch.from_numpy(global_mean).to(device)
    # global_std = torch.from_numpy(global_std).to(device)


    epochs = 300
    for epoch in range(epochs):

        model.train()
        loss_per_batch = 0
        total_batch = len(train_dataloader)
        
        for X_train, y_train in tqdm(train_dataloader, ncols=60):
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            spec_train = mel_spectrogram(X_train)
            # spec_train = (spec_train - global_mean) / global_std  # global_mvn
            prob_train = model(spec_train)
            
            model.zero_grad()
            loss = criterion(prob_train, y_train)
            loss.backward()
            optimizer.step()

            loss_per_batch += loss.item() / total_batch

        model.eval()
        with torch.set_grad_enabled(False):

            # speaker_models        
            speaker_models = {}
            for X_enroll, y_enroll in enroll_dataloader:
                X_enroll = X_enroll.view(utt_num * args["div"], -1).to(device)
                
                spec_enroll = mel_spectrogram(X_enroll)
                # spec_enroll = (spec_enroll - global_mean) / global_std  # global_mvn
                prob_enroll = model(spec_enroll, train=False)

                speaker_embedding = torch.mean(prob_enroll, axis=0).detach().cpu().numpy()
                speaker_label = int(torch.unique(y_enroll))
                speaker_models[speaker_label] = speaker_embedding

            # enroll for trial
            enroll_trial = []
            for enroll_label in trial_enroll:
                enroll_trial.append(speaker_models[enroll_label])
            enroll_trial = np.stack(enroll_trial)

            # eval for trial
            eval_trial = []
            for X_eval in tqdm(eval_dataloader, ncols=60):
                X_eval = X_eval.view(args["batch_size"] * args["div"], -1).to(device)
                
                spec_eval = mel_spectrogram(X_eval)
                # spec_eval = (spec_eval - global_mean) / global_std  # global_mvn
                prob_eval = model(spec_eval, train=False)

                prob_eval = prob_eval.view(args["batch_size"], args["div"], -1)
                eval_embedding = torch.mean(prob_eval, axis=1).detach().cpu().numpy()
                eval_trial.append(eval_embedding)

            eval_trial = np.vstack(eval_trial)

            # caculate EER
            trial_scores = []
            for i in range(len(trial_corrects)):
                score = cos_sim(enroll_trial[i,:], eval_trial[i,:]) 
                trial_scores.append(float(score))
                    
            eer = EER(trial_corrects, trial_scores)
            print(f"epoch : {epoch + 1}  loss_per_batch : {loss_per_batch:.6f}  eer : {eer:.6f}\n")

        if args["neptune"]:
            neptune.log_metric("loss", loss_per_batch)
            neptune.log_metric("eer", eer)

    print("training_finished")


if __name__ == "__main__":

    print("\nloader   : " + timestamp("loader.py"))
    print("convnet  : " + timestamp("models/convnet.py"))
    print("main     : " + timestamp("main.py") + "\n")

    args = {
        # setting arguments
        "GPU_NUM" : 0,
        "neptune" : False,

        # enroll/eval arguments
        "scaling" : 1.5,
        "div" : 4,
        "batch_size" : 100,

        # mfcc arguments
        "n_mfcc" : 13,
        "delta" : True,
        "utt_regularization" : True,

        # melspectrogram arguments
        "sample_rate" : 16000,
        "win_length" : 25,
        "hop_length" : 10,
        "n_fft" : 1024,
        "n_mels" : 60,

        #optimizatino argument,
        "learning_rate" : 1e-2,
        "weight_decay" : 1e-4,
        "optimizer" : "SGD",
    }

    main(args)