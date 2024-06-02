#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 17:44:18 2024

@author: rabbiyayounas
"""
# Imports the TimeSeriesTransformer configuration and model for time series prediction from the Hugging Face transformers library.
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
#Imports the PyTorch library.
import torch
#Imports the DataLoader and Dataset classes from PyTorch, which are used for handling and batching data.
from torch.utils.data import DataLoader, Dataset


#Defines a custom dataset class for cryptocurrency data by inheriting from PyTorch's Dataset class.
class CryptoDataset(Dataset):
    
    #Initializes the dataset with the data and sequence length.
    def __init__(self, data, seq_length):
        
        #self.data: Stores the data.
        self.data = data
        
        #self.seq_length: Stores the sequence length.
        self.seq_length = seq_length


    def __len__(self):
       
  #len method: Returns the number of samples in the dataset.
  #The length is the total number of data points minus the sequence length.
        return len(self.data) - self.seq_length

  #getitem method: Retrieves a sequence and its next value.
  #idx: The starting index of the sequence.
  #Returns a tuple of (sequence, next value).

    def __getitem__(self, idx):
        return (self.data[idx:idx+self.seq_length], self.data[idx+self.seq_length])

#Dataset and DataLoader
#seq_length = 30: Sets the sequence length to 30.
#dataset = CryptoDataset(...): Creates an instance of the CryptoDataset with 'Close' values from the data DataFrame.
#dataloader = DataLoader(...): Creates a DataLoader to handle batching and shuffling of the dataset.
#batch_size=32: Sets the batch size to 32.
#shuffle=True: Shuffles the data at each epoch.

seq_length = 30
dataset = CryptoDataset(data['Close'].values, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


#Model Configuration

#config = TimeSeriesTransformerConfig(...): Creates a configuration for the TimeSeriesTransformer model.
#prediction_length=1: Sets the prediction length to 1 (predicting one step ahead).
#context_length=seq_length: Sets the context length to the sequence length (30).
#input_size=1: Sets the input size to 1 (univariate time series).
#num_encoder_layers=3: Sets the number of encoder layers to 3.
#num_decoder_layers=3: Sets the number of decoder layers to 3.
#d_model=64: Sets the dimension of the model to 64.
#num_heads=4: Sets the number of attention heads to 4.


config = TimeSeriesTransformerConfig(
    prediction_length=1,
    context_length=seq_length,
    input_size=1,
    num_encoder_layers=3,
    num_decoder_layers=3,
    d_model=64,
    num_heads=4
)

#Model Initialization
#model = TimeSeriesTransformerForPrediction(config): Initializes the TimeSeriesTransformer model with the configuration.
#model.train(): Sets the model to training mode.


model = TimeSeriesTransformerForPrediction(config)
model.train()


#Optimizer and Loss Function

#optimizer = torch.optim.Adam(...): Initializes the Adam optimizer with the model parameters and a learning rate of 0.001.
#criterion = torch.nn.MSELoss(): Initializes the Mean Squared Error (MSE) loss function.


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()


#for epoch in range(10): Iterates over 10 epochs.
#for x, y in dataloader: Iterates over batches in the DataLoader.
#x: Batch of input sequences.
#y: Batch of target values.
#optimizer.zero_grad(): Clears the gradients of all optimized parameters.
#x = x.unsqueeze(-1).float(): Adds a feature dimension to x and converts it to float.
#y = y.float().unsqueeze(-1): Adds a feature dimension to y and converts it to float.
#output = model(inputs_embeds=x, labels=y): Passes the input sequences and labels through the model to get the output.
#loss = criterion(output.logits, y): Computes the loss between the model's predictions and the true values.
#loss.backward(): Backpropagates the loss.
#optimizer.step(): Updates the model parameters based on the gradients.
#print(f'Epoch {epoch+1}, Loss: {loss.item()}'): Prints the loss for each epoch.


for epoch in range(10):
    for x, y in dataloader:
        optimizer.zero_grad()
        x = x.unsqueeze(-1).float()  # Add feature dimension
        y = y.float().unsqueeze(-1)  # Add feature dimension
        output = model(inputs_embeds=x, labels=y)
        loss = criterion(output.logits, y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')