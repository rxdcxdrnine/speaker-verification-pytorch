# speaker-verification-pytorch
Speaker Verification with Pytorch Implementation.

## 1. Preprocessing
- melspectrogram
- mfcc
- local cepstral mean/variance
- time augmentation

## 2. DNN
### 2.1 Models
> - MLP (Linear + BatchNorm + ReLU)
> - CNN (Conv2d + BatchNorm + ReLU)
> - ResNet
> - SeResNet (Squeeze-and-Excitation Network + ResNet)

### 2.2 Loss functions
> - crossEntropy
> - amsoftmax
> - prototypical

### 2.3 Optimizers
> - SGD
> - Adam

## 3. Eval
> - EER
