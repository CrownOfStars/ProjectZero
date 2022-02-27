import torch

import torch.nn as nn

import numpy as np

def DataLoader():
    mni = np.load("src\\mnist.npz")
    
    trainY = None
    testX = None
    testY = None
    for data in mni.files:
        if mni[data].shape == (60000,784):
            trainX = mni[data]
        elif mni[data].shape == (60000,10):
            trainY = mni[data]
        elif mni[data].shape == (10000,784):
            testX = mni[data]
        else:
            testY = mni[data]

    return (trainX,trainY),(testX,testY)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Net = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )

    def forward(self,x):
        y_pred = self.Net(x)
        return y_pred

train,test = DataLoader()

loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-3

EPOCH = 20

model = Model()

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

X = train[0]
Y = train[1]

X_v = test[0]
Y_v = test[1]

tLoss = []
vLoss = []

for i in range(EPOCH):
    
    np.random.seed(i)
    np.random.shuffle(X)
    np.random.seed(i)
    np.random.shuffle(Y)
    np.random.seed(i)
    np.random.shuffle(X_v)
    np.random.seed(i)
    np.random.shuffle(Y_v)

    for u in range(len(X)//60):
        y_pred= model(torch.tensor(X[u*60:(u+1)*60]).float())
        trainLoss = loss_fn(y_pred, torch.tensor(Y[u*60:(u+1)*60]).float())
        
        optimizer.zero_grad()
        trainLoss.backward()
        optimizer.step()

        
        if u % 50 == 0:
            vy_pred= model(torch.tensor(X_v[u*10:u*10+500]).float())
            validationLoss = loss_fn(vy_pred, torch.tensor(Y_v[u*10:u*10+500]).float())
            #print(f"trainLoss={trainLoss}, validationLoss={validationLoss}")
            tLoss.append(trainLoss.item())
            vLoss.append(validationLoss.item())

def maxof(li):
    Max = li[0]
    id = 0
    for i in range(1,10):
        if Max < li[i]:
            Max = li[i]
            id = i
    return id

cnt = 0
for u in range(len(X_v)):
    y_pred= model(torch.tensor(X[u]).float())
    if maxof(y_pred.detach().numpy()) == maxof(Y[u]):
        cnt += 1

print(f"acc = {cnt*100/len(X_v)}%")