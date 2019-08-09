import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from LoadData import Data
from AFM import afm

def train(model, data, config): 
    lr = config.lr
    optimizer = optim.Adam(model.parameters(),lr)

    total_loss = 0
    loss = 0
    for batch_id, (group, label) in enumerate(data):
        if config.GPU_available:
            group = group.to(device)
            label = label.to(device)

        model.zero_grad()
        output = model(group,config)
        loss = nn.MSELoss()(output,label)
        total_loss += loss
        loss.backward()
        optimizer.step()
        print('batch_id : %d, loss : %f' %(batch_id,loss))
    total_loss /= batch_id
    print(total_loss)
        

if __name__ == '__main__':
    config = Config()
    device = torch.device('cuda')
    trainData = Data(config,'test')
    num_user = trainData.user_number()
    model = afm(config, num_user)

    if config.GPU_available:
        model = model.to(device)

    for epoch in range(config.epoch):
        model.train()
        print(epoch+1)
        train(model, trainData.dataLoader(), config)