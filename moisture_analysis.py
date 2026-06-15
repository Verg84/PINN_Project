import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
import pandas as pd
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset, DataLoader 


# load the data
xrds=xr.open_dataset('moisture.nc')
print(xrds)

df=xrds.to_dataframe()
df.to_csv('moisture.csv')

moisture=pd.read_csv('moisture.csv')
print(moisture.head())

# get column names with missing data
col_names_missing_data=[col for col in moisture.columns if moisture[col].isnull().any()]
# print(col_names_missing_data)
# drop columns
moisture_reduced=moisture.drop(['time'],axis='columns')
print(moisture_reduced.head())

# Impute missing values
for col in col_names_missing_data:
    moisture_reduced[col +'_was_missing']=moisture_reduced[col].isnull()

mImputer=SimpleImputer()
imputed_moisture=pd.DataFrame(mImputer.fit_transform(moisture_reduced))
imputed_moisture.columns=moisture_reduced.columns

print(imputed_moisture)

X_list=imputed_moisture.sm.values
X_np=np.array(X_list,dtype=np.float32).reshape(-1,1)
X=torch.from_numpy(X_list)
print(f'shape {X.shape}:\n{X}')

y_list=imputed_moisture.nobs.values
y_np=np.array(y_list,dtype=np.float32).reshape(-1,1)
y=torch.from_numpy(y_np)
print(f'shape {y.shape}:\n{y}')


# DATASET
class LRDataset(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.y=y
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]

# DATALOADER
loader=DataLoader(dataset=LRDataset(X_np,y_np),batch_size=2)

# MODEL CLASS
class LRModel(nn.Module):
    def __init__(self,input_size,output_size):
        super(LRModel,self).__init__()
        self.linear=nn.Linear(input_size,output_size)
    def forward(self,x):
        return self.linear(x)
model=LRModel(input_size=1,output_size=1)
model.train()

# Training Parameters
loss_fn=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.02)

# TRAINING
losses=[]
number_of_epochs=1000

for epoch in range(number_of_epochs):
    for j,(X,y) in enumerate(loader):
        # Optimize
        optimizer.zero_grad()
        # Forward Pass
        y_pred=model(X)
        # Compute loss
        loss=loss_fn(y_pred,y)
        losses.append(loss)
        # Backpropagation
        loss.backward()
        # Update weights
        optimizer.step()
        # Store the loss
        losses.append(float(loss.data))
        # Print
        if(epoch%100==0):
            print(f'EPOCH{epoch} - LOSS:{loss}')



