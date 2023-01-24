import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import segmentation_models_pytorch as smp

from SeismoDataset import SeismoDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



train_set = SeismoDataset('subset.hdf5')


print('there are {} train samples in this dataset'.format(len(train_set)))

def val_model(dataloader, model, BCE_loss):
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0
    iou_dict= {'high': {'int': 0, 'union':0}, 'medium': {'int': 0, 'union':0}, 'low': {'int': 0, 'union':0}}
    for data in dataloader:
        batch_data, batch_labels = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        preds = model(batch_data)

        high_loss = BCE_loss(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
        med_loss = BCE_loss(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
        low_loss = BCE_loss(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
        loss = 3*high_loss + 2*med_loss + low_loss
        #loss = high_loss + med_loss + low_loss
        test_loss = loss.item()
        total_loss += test_loss
        iou_dict= compute_iou(preds[:,0,:,:], batch_labels[:,0,:,:], 'high', iou_dict)
        iou_dict= compute_iou(preds[:,1,:,:], batch_labels[:,1,:,:], 'medium', iou_dict)
        iou_dict= compute_iou(preds[:,2,:,:], batch_labels[:,2,:,:], 'low', iou_dict)
    display_iou(iou_dict)


    final_loss = total_loss/len(dataloader)
    print("Validation Loss: {}".format(round(final_loss,8)), flush=True)
    return final_loss

def train_model(train_dataloader, val_dataloader, model, n_epochs):
    lr = 1e-5
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    history = dict(train=[], val=[])
    best_loss = 10000.0
    BCE_loss = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        total_loss = 0.0
        print('--------------\nStarting Epoch: {}'.format(epoch), flush=True)
        model.train()
        torch.set_grad_enabled(True)
        #for batch_data, batch_labels in train_dataloader:
        for data in train_dataloader:
            batch_data, batch_labels = data
            batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
            print(batch_data.shape)
            print()
            #print(torch.isnan(batch_data).any())
            optimizer.zero_grad() # zero the parameter gradients
            preds = model(batch_data)
            high_loss = BCE_loss(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
            med_loss = BCE_loss(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
            low_loss = BCE_loss(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
            loss = 3*high_loss + 2*med_loss + low_loss
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            total_loss += train_loss
        epoch_loss = total_loss/len(train_dataloader)
        print("Training Loss:   {0}".format(round(epoch_loss,8), epoch+1), flush=True)
        #val_loss = val_model(val_dataloader, model, BCE_loss)
        #history['val'].append(val_loss)
        history['train'].append(epoch_loss)

        #if val_loss < best_loss:
        #    best_loss = val_loss
        #    torch.save(model, 'models/best_model.pth')
    print(history)
    return model, history




BATCH_SIZE = 8
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

n_epochs = 100
model = smp.DeepLabV3Plus(
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
)

model = model.to(device)
train_model(train_loader, val_loader, model, n_epochs)
