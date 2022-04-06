
#This can be run if the audio files are stored a zipped folder called "data.zip", this folder should contain 2 folders: "train" and "test"
#GPU runtime should be active

#setup and imports
import torch
import os
import librosa
import re
import numpy as np
import sklearn
import json

device = torch.device('cuda:0')
print("Device: {}".format(device))

#If running on colab
from google.colab import drive
drive.mount('/content/drive')

path = input('Input path of files:')

#Unzip data file, path may need to be changed
!unzip 'drive/My Drive/data.zip'
print('Files have been unzipped!')

#Training labels are imported
with open(path+'train.json', "r") as f:
    data_train = json.load(f)

X_files = list(data_train.keys())
y = list(data_train.values())

#Test lables are imported
with open(path+'test.json', "r") as f:
    data_test = json.load(f)

X_files_test = list(data_test.keys())

#Function to obtain Mel Frequency Cepstral Coefficients (MFCCs) from the .wav audio files
#The function also normalizes the length of all audio signals: shorter signals are padded from the center, longer signals are truncated from the center

def to_mfcc(filename, in_path, sample_rate, seconds, mfcc_channels):
  signal, sr = librosa.load(in_path+filename, sr = sample_rate)
  length = len(signal)
  length2 = length / 2
  size = seconds * sample_rate
  halfsize = size/ 2
  n_fft = 2048
  hop_length = 512

  if length < size:
    signal = librosa.util.pad_center(signal, size=size, mode='constant')
  else:
    signal = signal[int(length2) - int(halfsize) : int(length2) + int(halfsize)]

  mfcc = librosa.feature.mfcc(signal, n_fft = n_fft, hop_length=hop_length, n_mfcc = mfcc_channels).T

  return mfcc

#Create a 3-d numpy array to store the mfccs for all training files and broadcast the mfccs
in_path_train = '/content/data/train/'
mfcc_train = to_mfcc(filename = X_files[0], in_path = in_path_train, sample_rate = 22050, seconds = 4, mfcc_channels=18)
dim_train = mfcc_train.shape

X = np.zeros((len(X_files), dim_train[0], dim_train[1]))

for i, file in enumerate(X_files):
  mfcc = to_mfcc(filename = file, in_path = in_path_train, sample_rate = 22050, seconds = 4, mfcc_channels=18)
  X[i,:,:] = mfcc
  print(i)

#Save mfcc array
np.save(path+'X_final_code.npy', X)

#Create a 3-d numpy array to store the mfccs for all test files and broadcast the mfccs
in_path_test = '/content/data/test/'
mfcc_test = to_mfcc(filename = X_files_test[0], in_path = in_path_test, sample_rate = 22050, seconds = 4, mfcc_channels=18)
dim_test = mfcc.shape

X_test = np.zeros((len(X_files_test), dim_test[0], dim_test[1]))

for i, file in enumerate(X_files_test):
  X_test[i, :, :] = to_mfcc(filename=file, in_path=in_path_test, sample_rate=22050, seconds=4, mfcc_channels=18)
  print(i)

#Save the mfcc array
np.save(path+'X_test_final_code.npy', X_test)

#Loading the mfcc array
X = np.load(path+'X_final_code.npy')
X_test = np.load(path+'X_test_final_code.npy')

#A small subset of the training data allows to keep track of the performance, not intended for fine tuning
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, test_size = 0.05)

# Our vectorized labels
X_train_dim = X_train.shape
X_val_dim = X_val.shape
X_test_dim = X_test.shape

X_train = np.asarray(X_train).astype('float32')
X_train = X_train[..., np.newaxis].reshape(X_train_dim[0], 1, X_train_dim[1], X_train_dim[2])
X_train = torch.tensor(X_train)
y_train = np.asarray(y_train).astype('float32') - 1 #bc index should start at 0
y_train = torch.tensor(y_train).type(torch.LongTensor)

X_val = np.asarray(X_val).astype('float32')
X_val = X_val[..., np.newaxis].reshape(X_val_dim[0], 1, X_val_dim[1], X_val_dim[2])
X_val = torch.tensor(X_val)
y_val = np.asarray(y_val).astype('float32') - 1 #bc index should start at 0
y_val = torch.tensor(y_val).type(torch.LongTensor)

X_test = np.asarray(X_test).astype('float32')
X_test = X_test[..., np.newaxis].reshape(X_test_dim[0], 1, X_test_dim[1], X_test_dim[2])
X_test = torch.tensor(X_test)

in_shape = X_train[0].shape

#Run to move data to GPU
X_train, y_train, X_val, y_val, X_test = X_train.to(device), y_train.to(device), X_val.to(device), y_val.to(device), X_test.to(device)

from torch import nn, optim

model_1 = nn.Sequential()

# Add convolutional and pooling layers
model_1.add_module('Conv_1', nn.Conv2d(in_channels=1, out_channels=68, stride = (2,2), kernel_size=(5,5), padding = (1,1)))
model_1.add_module('Relu_1', nn.ReLU())
model_1.add_module('MaxPool_1', nn.MaxPool2d(kernel_size=2, stride = (3,2), padding = (1,1)))
model_1.add_module('BachNorm_1', nn.BatchNorm2d(68)) 

model_1.add_module('Conv_2', nn.Conv2d(in_channels=68, out_channels=128, kernel_size=(3,3),padding = (1,1)))
model_1.add_module('Relu_2', nn.ReLU())
model_1.add_module('MaxPool_2', nn.MaxPool2d(kernel_size=2, stride = (2,2)))
model_1.add_module('BachNorm_2', nn.BatchNorm2d(128))


model_1.add_module('Conv_3', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3),padding = (1,1)))
model_1.add_module('Relu_3', nn.ReLU())
model_1.add_module('BachNorm_3', nn.BatchNorm2d(256))


model_1.add_module('Conv_4', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(2,2),padding = (1,1)))
model_1.add_module('Relu_4', nn.ReLU())
model_1.add_module('MaxPool_4', nn.MaxPool2d(kernel_size=2)) 
model_1.add_module('BachNorm_4', nn.BatchNorm2d(512))  

# Add a Flatten layer to the model
model_1.add_module('Flatten', nn.Flatten())

# Add a Linear layer with 64 units and relu activation
model_1.add_module('Linear_1', nn.Linear(in_features=512*7*1, out_features=800, bias=True))
model_1.add_module('Relu_L_1', nn.ReLU())
model_1.add_module('Dropout_1', nn.Dropout(p=0.3))

# Add the last Linear layer.
model_1.add_module('Linear_2', nn.Linear(in_features=800, out_features=183, bias=True))
model_1.add_module('Out_activation', nn.Softmax(-1))

model_1 = model_1.to(device)

from torchsummary import summary
summary(model_1, input_size=(in_shape))

from torch import optim
from random import shuffle

#The train_model() function is based on the Deep Learning practicals (MLP, CNN practicals)
def train_model(model, x_train_, y_train_, x_valid_, y_valid_, batch_size, num_epochs):
    print("Training...")
    
    # Create the loss function
    categorical_cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    acc_history = []
    val_acc_history = []
    loss_history = []
    val_loss_history = []
    
    batch_indices = list(range(len(x_train_)//batch_size))

    shuffle(batch_indices)

    for epoch in range(num_epochs):
      #Create optimizer 
        if epoch <= 20:
          optimizer = optim.Adam(model_1.parameters(), lr=0.0001)
        elif epoch <= 40:
          optimizer = optim.RMSprop(model_1.parameters(), lr=0.0001, weight_decay=0.0001)
        elif epoch <= 55:
          optimizer = optim.RMSprop(model_1.parameters(), lr=0.00005, weight_decay=0.00001)
        elif epoch <= 75: 
          optimizer = optim.RMSprop(model_1.parameters(), lr=0.00005)
        else: 
          optimizer = optim.RMSprop(model_1.parameters(), lr=0.00001)

        print('Currently training epoch {} of {}'.format(epoch, num_epochs))
        model.train()
        running_train_acc = 0 
        running_train_loss = 0
        for b_idx in batch_indices:
            optimizer.zero_grad()
            
            # Create batch
            batch_start = b_idx * batch_size
            batch_end = (b_idx + 1) * batch_size
            data_batch = x_train_[batch_start : batch_end]
            y_batch = y_train_[batch_start : batch_end]
            
            # Forward pass
            predictions = model(data_batch).squeeze()

            # Calculate loss
            loss = categorical_cross_entropy_loss(predictions, y_batch)

            running_train_loss += float(loss)

            # Backward pass and update
            loss.backward()
            optimizer.step()

            # Calculate perfrmance metric (accuracy)
            accuracy = (torch.max(predictions, dim=-1, keepdim=True)[1].flatten() == y_batch).sum() / len(y_batch)
            running_train_acc += accuracy

        train_acc = running_train_acc / len(batch_indices)
        train_loss = running_train_loss / len(batch_indices)

        model.eval()
        predictions = model(x_valid_).squeeze()
        val_acc = (torch.max(predictions, dim=-1, keepdim=True)[1].flatten() == y_valid_).sum() / len(y_valid_)
        val_loss = float(categorical_cross_entropy_loss(predictions, y_valid_))

        acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print('Accuracy is {} on train set and {} on validation set'.format(train_acc, val_acc))

    print('Training done.')
    predictions = model(x_valid_).squeeze()
    accuracy = (torch.max(predictions, dim=-1, keepdim=True)[1].flatten() == y_valid_).sum() / len(y_valid_)

    print("Accuracy on test set: {}".format(accuracy))
    
    return acc_history, val_acc_history, loss_history, val_loss_history

#Training model
acc_history, val_acc_history, loss_history, val_loss_history = train_model(model_1, X_train, y_train, X_val, y_val, batch_size=10, num_epochs = 100)

#Predict test data
model_1.eval()
test_predictions = model_1(X_test).squeeze()
test_predictions_class = torch.max(test_predictions, dim=-1, keepdim=True)[1].flatten()

#In training 1 was substracted to start at 0, we need to add it back
test_predictions_class = test_predictions_class + 1
test_predictions_class = test_predictions_class.tolist()

#Generate prediction file
for i, file in enumerate(data_test):
  data_test[file] = str(test_predictions_class[i])

with open(path+'test_submission_group1_last_of_last.json', 'w') as f:
    json.dump(data_test, f)

