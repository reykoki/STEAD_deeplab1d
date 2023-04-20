import os
from SeismoDataset import SeismoDataset
import numpy as np
import sys
import json
import shelve
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from functools import partial
from typing import Any, List, Optional
from decoder.deeplabv3.model import DeepLabV3Plus
from torchvision import transforms

torch.manual_seed(2022)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, flush=True)

def get_accuracy(time_diffs):
    time_diffs = np.array(list(zip(*time_diffs)))
    p_time_diff = time_diffs[0]*(1/100)
    s_time_diff = time_diffs[1]*(1/100)
    num_events = len(p_time_diff)
    p_num_lt_1 = len([i for i in p_time_diff if abs(i)<.1])
    s_num_lt_1 = len([i for i in s_time_diff if abs(i)<.1])
    print('fraction of p-waves within .1 sec: {}'.format(p_num_lt_1/num_events), flush=True)
    print('fraction of s-waves within .1 sec: {}'.format(s_num_lt_1/num_events), flush=True)
    p_num_lt_5 = len([i for i in p_time_diff if abs(i)<.5])
    s_num_lt_5 = len([i for i in s_time_diff if abs(i)<.5])
    print('fraction of p-waves within .5 sec: {}'.format(p_num_lt_5/num_events), flush=True)
    print('fraction of s-waves within .5 sec: {}'.format(s_num_lt_5/num_events), flush=True)

def histogram_data(y_true, y_pred):
    analyst_picks = torch.argmax(y_true, dim=-1)
    alg_picks = torch.argmax(y_pred, dim=-1)
    time_diffs = analyst_picks-alg_picks
    return time_diffs.tolist()



#train_set = SeismoDataset('subset.hdf5')
train_set = SeismoDataset('../subset_1k.hdf5')
#train_set = SeismoDataset('/scratch/alpine/mecr8410/STEAD/stead/wf_ds.hdf5')
print(len(train_set))



def earth_mover_distance(y_true, y_pred):
    emd = torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)))
    return emd

def test_model(dataloader, model, exp_num):
    history = {'batch_data': [], 'batch_labels': [], 'predictions': [], 'losses': [], 'time_diffs': []}
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0
    for data in dataloader:
        batch_data, batch_labels = data
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        predicted = model(batch_data)
        loss = earth_mover_distance(batch_labels, predicted).to(device)
        test_loss = loss.item()
        total_loss += test_loss
        time_diffs = histogram_data(batch_labels, predicted)
        history['batch_data'].extend(batch_data.tolist())
        history['predictions'].extend(predicted.tolist())
        history['batch_labels'].extend(batch_labels.tolist())
        history['losses'].append(loss.item())
        history['time_diffs'].extend(time_diffs)
    final_loss = total_loss/len(dataloader)
    print("Testing Loss: {}".format(round(final_loss,8)), flush=True)

    save_path = "/projects/mecr8410/history/spectro/exp{}/".format(exp_num)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ds = shelve.open(save_path + "test_results.shlv")
    ds.update(history)
    ds.close()
    get_accuracy(history['time_diffs'])

    return history



def val_model(dataloader, model):
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0
    all_time_diffs = []
    for data in dataloader:
        batch_data, batch_labels = data
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        predicted = model(batch_data)
        loss = earth_mover_distance(batch_labels, predicted).to(device)
        test_loss = loss.item()
        total_loss += test_loss
        time_diffs = histogram_data(batch_labels, predicted)
        all_time_diffs.extend(time_diffs)

    final_loss = total_loss/len(dataloader)
    print("Validation Loss: {}".format(round(final_loss,8)), flush=True)
    get_accuracy(all_time_diffs)
    return final_loss


def train_model(train_dataloader, val_dataloader, model, exp_num):
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-5)
    lambda1 = lambda epoch: 0.99 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    history = dict(train=[], val=[])
    best_loss = 100000.0
    num_epochs = 100


    for epoch in range(num_epochs):
        total_loss = 0.0
        print('--------------\nStarting Epoch: {}'.format(epoch), flush=True)
        model.train()
        torch.set_grad_enabled(True)
        #for batch_data, batch_labels in train_dataloader:
        for data in train_dataloader:
            batch_data, batch_labels = data
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad() # zero the parameter gradients
            predicted = model(batch_data)
            loss = earth_mover_distance(batch_labels, predicted).to(device)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            total_loss += train_loss
        epoch_loss = total_loss/len(train_dataloader)
        print("Training Loss:   {0}".format(round(epoch_loss,8), epoch+1), flush=True)
        val_loss = val_model(val_dataloader, model)
        history['val'].append(val_loss)
        history['train'].append(epoch_loss)
        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            save_path = 'models/exp{}/'.format(exp_num)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model, save_path+'best_model.pth')

    save_path = "/projects/mecr8410/history/spectro/exp{}/".format(exp_num)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ds = shelve.open(save_path + "history.shlv")
    ds.update(history)
    ds.close()

    print(history, flush=True)
    return model



exp_num = 'deeplab'


BATCH_SIZE = 128
#BATCH_SIZE = 256
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)


test_mode = False
if test_mode:
    model = torch.load('models/exp{}/best_model.pth'.format(exp_num))
    model = model.to(device)
else:
    model = DeepLabV3Plus(
            #encoder_name="resnext50_32x4d",
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=2
            )
    model = model.to(device)
    model = train_model(train_loader, val_loader, model, exp_num)

#test_model(test_loader, model, exp_num, hyperparams)

