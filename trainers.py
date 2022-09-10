# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 15:01:28 2022

@author: Behrouz Eslami
"""


import torch
import torch.nn as nn


class ClassifierTrainer():
    
    def __init__(self, model, optimizer, train_loader, test_loader, init_model_pars=True):
        
        self.model = model
        if init_model_pars:
            self.model.init_model_params()
        self.CUDA = torch.cuda.is_available()
        if self.CUDA:
            self.model = model.cuda()
        self.loss_fn = nn.CrossEntropyLoss() 
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_losses = None
        self.test_losses = None
        self.train_accuracies = None
        self.test_accuracies = None
        
    def fit(self, num_epochs):

        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

        train_batch_num = len(self.train_loader)
        test_batch_num = len(self.test_loader)

        for epoch in range(num_epochs):

            epoch_loss = 0
            epoch_accuracy = 0

            self.model.train()

            for (inputs, labels) in self.train_loader:

                if self.CUDA:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = self.model(inputs)
                _, predictions = torch.max(outputs, 1)
                loss = self.loss_fn(outputs, labels)
                accuracy = (predictions==labels).sum()/len(labels)
                epoch_loss += loss/train_batch_num
                epoch_accuracy += accuracy/train_batch_num

                self.optimizer.zero_grad()
                loss.backward() 
                self.optimizer.step()

            self.train_losses.append(epoch_loss.item())
            self.train_accuracies.append(epoch_accuracy.item())

            self.model.eval()

            with torch.no_grad():

                epoch_loss = 0
                epoch_accuracy = 0

                for (inputs, labels) in self.test_loader:

                    if self.CUDA:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    outputs = self.model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = self.loss_fn(outputs, labels)
                    accuracy = (predictions==labels).sum()/len(labels)
                    epoch_loss += loss/test_batch_num
                    epoch_accuracy += accuracy/test_batch_num

                self.test_losses.append(epoch_loss.item())
                self.test_accuracies.append(epoch_accuracy.item())

            print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}'
                   .format(epoch+1, num_epochs, self.train_losses[-1], self.train_accuracies[-1], 
                     self.test_losses[-1], self.test_accuracies[-1]))
            
    def predict(self, dataset, idx):
        
        with torch.no_grad():
            x, y = dataset[idx]
            output = self.model.cpu()(x.unsqueeze(0))
            _, prediction = torch.max(output, 1)
        
        return prediction.item()
    
    def save_model(self, file):
        
        torch.save(self.model.state_dict(), file)
    
    def load_model(self, file):
        
        self.model.load_state_dict(torch.load(file))