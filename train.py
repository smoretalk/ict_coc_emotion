import os
import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
import torchvision
from torchvision import datasets, transforms, models
import torch.optim as optim
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import random
from torch.utils.data import DataLoader
from pathlib import Path

torch.set_num_threads(32)

#data_path
data_path = '/home/data_storage/emotion_speech/emotion_wav/'
wav_path = data_path + 'wavs/'
directory_path = '/home/hschung/speech/Speech-Emotion-Recognition-ROS/'


#read csv file 
df1 = pd.read_csv(data_path + 'male_script.csv')
df2 = pd.read_csv(data_path + 'female_script.csv')

#merge two tables
frames = [df1, df2]
df = pd.concat(frames, ignore_index=True)


#delete unused columns
df = df.drop(['성우', '연령', '성별', '상황키워드', '감정_소분류', 'Unnamed: 8'], axis=1)

#labels 
labels = df['감정_대분류'].unique()
labels = labels.tolist()
print(labels)

#add suffix to wav files
df['NO.'] = wav_path + df['NO.'] + '.wav'
df['NO.'] = df['NO.'].astype('string')
print(df)

#create dataset 
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = df['NO.']
        self.labels = df['감정_대분류']
        self.dataframe = df
        self.frame_length = 0.025
        self.frame_stride = 0.010

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]

        X, sr = librosa.load(data, res_type='kaiser_fast',sr=16000,offset=0.5)
        sample_rate = 16000
        input_nfft = int(round(sample_rate*self.frame_length))
        input_stride = int(round(sample_rate*self.frame_stride))

        S = librosa.feature.melspectrogram(y=X, n_mels=64, n_fft=input_nfft, hop_length=input_stride)
        P = librosa.power_to_db(S, ref=np.max)

        ## get label
        if df['감정_대분류'][index] == '상처':
            label = 0
        elif df['감정_대분류'][index] == '기쁨':
            label = 1
        elif df['감정_대분류'][index] == '불안':
            label = 2
        elif df['감정_대분류'][index] == '당황':
            label = 3
        elif df['감정_대분류'][index] == '분노':
            label = 4
        elif df['감정_대분류'][index] == '슬픔':
            label = 5
        else:
            label = 6
        return P, label

    def __len__(self):
        return len(self.data)

#dataloader
dataset = EmotionDataset()

train_size = int(0.9 * len(dataset))
val_size = int(len(dataset) - train_size)

train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], torch.Generator().manual_seed(42))


# spec2img
def getimg(dataset,feature):
        img_path=[]
        labels=[]
        for i in range(len(dataset)):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                p = librosa.display.specshow(dataset[i][0],ax=ax, sr=16000, hop_length=int(round(16000*0.025)), x_axis='time',y_axis='linear')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

                fig.savefig('./images_aug/%s_%d_%s.jpg' % (dataset[i][1],i,feature), bbox_inches=extent)
                img_path.append('./images_aug/%s_%d_%s.jpg' % (dataset[i][1],i,feature))
                labels.append(dataset[i][1])
                plt.ioff()
                plt.close()
        misc = {"img_path": img_path,
                "labels": labels}
        torch.save(misc, directory_path + "image_labels_augmentation_train.pt")
        return img_path , labels


# train_path , train_labels = getimg(train_set,'train')
# val_path , val_labels = getimg(val_set,'val')

#train without aug
train_files = torch.load(directory_path + "image_labels_train.pt")
train_path = train_files['img_path']
train_labels = train_files['labels']

val_files = torch.load(directory_path + "image_labels_val.pt")
val_path = val_files['img_path']
val_labels = val_files['labels']


#transformations 
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
                                                           

class img2tensor():
    def __init__(self,data_path,labels,transforms):
        self.data_path = data_path
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.data_path)

        
    def __getitem__(self, index):
        img_path = self.data_path[index]
        image = Image.open(img_path)
        I = train_transforms(image)
        label = self.labels[index]

        return I, label
        

# set batch_size
batch_size = 256
# dataloader
train_dataloader = torch.utils.data.DataLoader(img2tensor(train_path,train_labels,train_transforms), batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(img2tensor(val_path,val_labels,test_transforms), batch_size=batch_size, shuffle=False)
dataloaders_dict ={'train':train_dataloader, 'val': val_dataloader}
# test
batch_iterator = iter(dataloaders_dict['train'])
inputs, labels = next(batch_iterator)
print(inputs.size())
print(labels)

#model 
model = torchvision.models.resnet50(pretrained=True)
model.to(torch.device('cuda'))
model.fc = nn.Linear(in_features=2048, out_features=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters() ,lr=0.00001, weight_decay=1e-6, momentum=0.9)
model.train()


#training loop 
def train(net, dataloader, criterion, optimizer, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    torch.backends.cudnn.benchmark = True

    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('-------------------------------')

        for phase in ['train','val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and(phase == 'train'):
                continue
            for inputs, labels in tqdm(dataloader[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs,1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() *inputs.size(0)

                    epoch_corrects += torch.sum(preds == labels.data)

                    epoch_loss = epoch_loss / len(dataloader[phase].dataset)
                    epoch_acc = epoch_corrects.double() / len(dataloader[phase].dataset)
        

            print('{} Loss: {:.4f} ACC {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
                
        torch.save(net.state_dict(), directory_path + "new_model.pt")

#training
train(model, dataloaders_dict, criterion,optimizer, num_epochs=200)