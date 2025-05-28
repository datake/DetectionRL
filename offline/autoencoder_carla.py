import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch import autograd
import argparse
import arguments as args
from agents import PPO_Agent
from env_wrappers import *
from networks import nature_cnn
from collections import deque
from torch.utils.data import TensorDataset, DataLoader
import time

class ConvAutoencoder(nn.Module):
    def __init__(self,action_dim,channel=4):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(channel, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(4*21*21 + action_dim, 256)
        self.fc2 = nn.Linear(256, 4*21*21)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, channel, 2, stride=2)

    def forward(self, x, action):
        # print('---')
        # print('x:',x.shape)
        x = torch.transpose(x,1,3) # NHWC -> NCHW
        # print('after transpose:',x.shape)
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        # print('after conv1:',x.shape)
        x = self.pool(x)
        # print('after pool:',x.shape)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        # print('after conv2:',x.shape)
        x = self.pool(x)  # compressed representation
        # print('after pool:',x.shape)
        x = x.view(-1, 4*21*21)
        # print('after view:',x.shape)

        x = torch.cat((x, action), dim=1) # f(s_t, a_t)
        # print('after cat:',x.shape)
        x = F.relu(self.fc1(x))
        # print('after fc1:',x.shape)
        x = F.relu(self.fc2(x))
        # print('after fc2:',x.shape)
        x = x.view(-1, 4, 21, 21)
        # print('after view:',x.shape)

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # print('after t_conv1:',x.shape)
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv2(x))
        # print('after t_conv2:',x.shape)
        x = torch.transpose(x,1,3) # NCHW -> NHWC
        # print('after transpose:',x.shape)
        return x

class EnsembleModel(nn.Module):
    def __init__(self,action_dim,model_num):
        super(EnsembleModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("mps") # for mac
        # self.device = "cpu"
        self.model_num = model_num
        self.action_dim = action_dim
        self.models = [ConvAutoencoder(self.action_dim).to(self.device) for _ in range(model_num)]
        self.optimizer = [torch.optim.Adam(model.parameters(), lr=0.001) for model in self.models] # 0.001
        self.criterion = nn.MSELoss()
        self.mse = nn.MSELoss(reduction='none')

    def compute_mse(self, x, action, target):
        # s_t, a_t, s_t+1
        mse_list = [0 for _ in range(self.model_num)]
        for i, model in enumerate(self.models):
            outputs = model(x, action)
            mse = self.mse(outputs, target)  # [data_num*n_labels, 84, 84, 4]
            mse = torch.mean(mse, dim=(1, 2, 3))   # [data_num*n_labels]
            mse_list[i] = mse.cpu().detach().numpy()
        return np.array(mse_list) # [model_num, data_num*n_labels]

    def learn(self, x, action, target):
        # for training
        loss_list = np.zeros(self.model_num)
        for i, model in enumerate(self.models):
            self.optimizer[i].zero_grad()
            outputs = model(x, action)
            # print('outputs:',i, outputs[:3,:1,:1,:1])
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer[i].step()
            loss_list[i] = loss.item()
        return loss_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train autoencoder-based transition model: \hat{s}_{t=1} = f(s_t, a_t)')
    parser.add_argument('--game', type=str, default="town",choices=['town', 'lane'], help='carla env')  # action space: 4, 9, 6
    parser.add_argument('--epoch', type=int, default=20)
    opt = parser.parse_args()
    print(opt)
    if opt.game == 'town':
        data_path = 'model/new_carla_data_town/carla-town-v0_10_1000000_1000'
    elif opt.game == 'lane':
        data_path = 'model/new_carla_data_lane/carla-lane-v0_10_1000000_1000'
    else:
        print('Enviroment Name Error!')

    ########### initialize the environment and data collection ###########
    # agent = PPO_Agent(args.batch_env.action_space, args.batch_env.observation_space, nature_cnn)

    data = np.load(data_path + '.npz')
    state = data['state']  # [10000, 84, 84, 4]
    action = data['action']  # (10000,)
    reward = data['reward']  # (99999,)
    terminal = data['terminal']
    nextstate = np.concatenate([state[1:, :, :, :], np.expand_dims(state[-1, :, :, :], axis=0)], axis=0)  # [10000, 84, 84, 4]
    n_labels = 9


    ########## Modeling via Convolutional Autoencoder ##########
    model = EnsembleModel(action_dim=n_labels,model_num=5)
    print('Prepare the dataloader!....', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    X_state = torch.from_numpy(state).float().to(model.device)
    Y_nextstate = torch.from_numpy(nextstate).float().to(model.device)
    X_action = torch.from_numpy(action).to(model.device)
    X_action = F.one_hot(X_action, num_classes=n_labels).float().to(model.device) # [data_num*n_labels, n_labels]

    dataset = TensorDataset(X_state, X_action, Y_nextstate)
    batch_size = 32  # Define the size of each batch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('Device: ', model.device)
    print('Start training...', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(opt.epoch):  # loop over the dataset multiple timee
        for j, (x_state, x_action, y_nextstate) in enumerate(dataloader):
            loss = model.learn(x_state, x_action, y_nextstate)
        loss_ = [round(i, 4) for i in loss]
        print('epoch:', epoch, 'ave loss', np.array(loss).mean(), 'each loss:', loss_, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    torch.save(model.state_dict(), 'model/autoencoder_{}.pth'.format(opt.game))
    # model.load_state_dict(torch.load('model/autoencoder_{}.pth'.format(opt.game)))


