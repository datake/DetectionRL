import numpy as np
from sklearn.covariance import EmpiricalCovariance,LedoitWolf,MinCovDet
import os
from env_wrappers import *
import arguments as args
from agents import PPO_Agent
from networks import nature_cnn
from collections import deque
import scipy.stats as stats
import copy, torch
from copy import deepcopy
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import itertools
import torch.nn as nn
import torch
from sklearn.manifold import TSNE
import pandas as pd
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from autoencoder import EnsembleModel
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='offline detection')
parser.add_argument('--game', type=str, default="carla", choices=['carla','breakout', 'asterix',  'spaceinvader','fishingderby', 'enduro',  'tutankham']) # action space: 4, 9, 6
parser.add_argument('--pca', type=int, default=1, help="PCA or not")
parser.add_argument('--feature', type=int, default=50, help="PCA feature length")
parser.add_argument('--conformal', type=int, default=1, help="whether we use the conformal to determine the thresholding")
opt = parser.parse_args()
print(opt)

img_num = 10000
a = os.listdir('model/new_carla_data_town/images_town')
b = os.listdir('model/new_carla_data_town/images_town_clear_to_rainy')
c = os.listdir('model/new_carla_data_town/images_town_day_to_night')
d = set(a).intersection(set(b)).intersection(set(c))
d = list(d)
l_image = ['img_{}.png'.format(value) for value in range(img_num) if 'img_{}.png'.format(value) in d]

Experiments = {'game1':['BreakoutNoFrameskip-v4','AsterixNoFrameskip-v4','SpaceInvadersNoFrameskip-v4'],
               'game2': ['EnduroNoFrameskip-v4','FishingDerbyNoFrameskip-v4','TutankhamNoFrameskip-v4'],
               'game3': ['carla'],
               'carla_OOD': ['rain', 'night'],
               'contaminated_ratio':[0.0],
               'feature':[opt.feature]}
if opt.conformal == 1:
    Experiments['covariance'] = ['Entropy_conformal']
else:
    Experiments['covariance'] = ['Empirical', 'MinCovDet', 'Euclidean_I', 'Euclidean_Diag', 'TMD']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if opt.game == 'carla':
    ATARI_NAME_list = Experiments['carla_OOD']
elif opt.game in ['breakout', 'asterix', 'spaceinvader', 'BAS']:
    ATARI_NAME_list = Experiments['game1']
elif opt.game in ['fishingderby', 'enduro',  'tutankham', 'EFT']:
    ATARI_NAME_list = Experiments['game2']
else:
    ATARI_NAME_list = None
    AssertionError('The game name is wrong!')

if opt.game == 'breakout':
    ATARI_NAME_list_focus = ['BreakoutNoFrameskip-v4']
elif opt.game == 'asterix':
    ATARI_NAME_list_focus = ['AsterixNoFrameskip-v4']
elif opt.game == 'spaceinvader':
    ATARI_NAME_list_focus = ['SpaceInvadersNoFrameskip-v4']
elif opt.game == 'enduro':
    ATARI_NAME_list_focus = ['EnduroNoFrameskip-v4']
elif opt.game == 'fishingderby':
    ATARI_NAME_list_focus = ['FishingDerbyNoFrameskip-v4']
elif opt.game == 'tutankham':
    ATARI_NAME_list_focus = ['TutankhamNoFrameskip-v4']
elif opt.game == 'carla':
    ATARI_NAME_list_focus = ['carla']
else:
    ATARI_NAME_list_focus = ATARI_NAME_list # loop other env in ATARI_NAME_list later
use_PCA = False if opt.pca == 0 else 1
if use_PCA:
    PCA_length = Experiments['feature']
else:
    PCA_length = [512]
loop = 0
Flag_Random = True
Flag_ADV = True
Flag_OOD = True
Flag_Write = True
if Flag_Write:
    Final_result = pd.DataFrame()

# define PCA function
def ApplyPCA(clean_feature_dict, PCA_length, data_num, n_labels):
    Clean_feature_dict = deepcopy(clean_feature_dict) # dict is immutable !!!!!!!!!!!!
    all_feature = deque(maxlen=data_num * n_labels)
    for i in range(n_labels): # labels 4
        all_feature.extend(clean_feature_dict[i])
    pca = PCA(n_components=PCA_length) # choose the number of components after PCA
    samples = np.array(all_feature).squeeze()
    pca.fit(samples)
    print('Conducting PCA {}, Variance Ratio is {}:'.format(feature_length, np.sum(pca.explained_variance_ratio_)))
    for i in range(n_labels):
        samples_clean = np.array(Clean_feature_dict[i]) # for evaluation
        Clean_feature_dict[i] = pca.transform(samples_clean)
    return pca, Clean_feature_dict

def MeanVarEstimate(Feature_dict, n_labels, covariance_estimator='Empirical'):
    model_list = []

    if covariance_estimator == 'TMD':
        SAMPLES = np.mat(Feature_dict[0]) # to compute all

    # fit the mean and covariance for each label
    for i in range(n_labels):
        samples = np.mat(Feature_dict[i])
        if i >= 1 and covariance_estimator == 'TMD':
            SAMPLES = np.concatenate([SAMPLES, samples], axis=0)
        # print(samples.shape)
        if covariance_estimator in ['Empirical', 'MD_conformal', 'Entropy_conformal', 'Envmodel_conformal']:
            model_list.append(EmpiricalCovariance())
        elif covariance_estimator in ['MinCovDet', 'RMD_conformal']:
            model_list.append(MinCovDet())
        elif covariance_estimator in ['TMD', 'Euclidean_I', 'Euclidean_Diag']:
            model_list.append(EmpiricalCovariance())
        else:
            print('covariance_estimator is wrong')
        model_list[i].fit(samples)

    if covariance_estimator == 'TMD':
        TMD = EmpiricalCovariance()
        TMD.fit(SAMPLES)
        for i in range(n_labels):
            model_list[i].precision_ = TMD.precision_ # to share the covariance(precision matrix)
    if covariance_estimator in ['Euclidean_I', 'Euclidean_Diag']:
        p = model_list[0].precision_.shape[0]
        for i in range(n_labels):
            if covariance_estimator == 'Euclidean_I':
                model_list[i].precision_ = np.eye(p) # simple euclidean
            else:
                model_list[i].precision_ = np.diag(1.0/np.diag(model_list[i].covariance_)) # simple euclidean
    return model_list


def ComputeEnvmodel(X_state, Y_nextstate, X_action, autoencoder):
    batch_size = 50
    mse_aggre = np.zeros((autoencoder.model_num, 1))
    # print('state dimension: ', X_state.shape)
    for i in range(int(X_state.shape[0]/batch_size)):
        X = X_state[i*batch_size:(i+1)*batch_size]
        act = X_action[i*batch_size:(i+1)*batch_size]
        Y = Y_nextstate[i*batch_size:(i+1)*batch_size]
        mse = autoencoder.compute_mse(X, act, Y) # list: [data_num ] * model_num
        mse_aggre = np.concatenate([mse_aggre, mse], axis=1)

    mse_aggre = mse_aggre[:, 1:] # [model_num, data_num*n_labels]
    # mse_aggre = np.concatenate(list_mse, axis=0) # [model_num, data_num*n_labels]
    mse_aggre = np.amin(mse_aggre, axis=0) # [model_num, data_num*n_labels] -> [data_num*n_labels]
    score = np.quantile(mse_aggre, 0.95)
    print('Score: ', score)
    return score

def ComputeEnvmodel_conformal(score, X_noisystate, Y_nextstate, X_action, autoencoder, Clean=False):
    Predicted_Negative_list = []
    fasle_positive, true_negative = 0.0, 0.0
    batch_size = 50
    mse_aggre = np.zeros((autoencoder.model_num, 1)) # initialize one column
    # print('noisy state dimension: ', X_noisystate.shape)
    for i in range(int(X_noisystate.shape[0]/batch_size)):
        X = X_noisystate[i * batch_size:(i + 1) * batch_size]
        act = X_action[i * batch_size:(i + 1) * batch_size]
        Y = Y_nextstate[i * batch_size:(i + 1) * batch_size]
        mse = autoencoder.compute_mse(X, act, Y)  # [model_num, batch_size]
        # try:
        #     mse = autoencoder.compute_mse(X, act, Y) # [model_num, batch_size]
        # except:
        #     print('X: ', X.shape, 'act: ', act.shape, 'Y: ', Y.shape)
        #     break
        mse_aggre = np.concatenate([mse_aggre, mse], axis=1)
        # print('mse_aggre: ', mse_aggre.shape)  # [model_num, batch_size]

    mse_aggre = mse_aggre[:, 1:]  # [model_num, data_num*n_labels]
    # print('mse_aggre:', mse_aggre.shape)
    assert mse_aggre.shape[1] == X_noisystate.shape[0]
    # mse = autoencoder.compute_mse(X_noisystate, X_action, Y_nextstate) # compare noisy next state and clean next state
    mse_aggre = np.amin(mse_aggre, axis=0)  # [model_num, data_num*n_labels] -> [data_num*n_labels]
    print(f'mean: {np.mean(mse_aggre)}, std: {np.std(mse_aggre)}, 25% quantile: {np.quantile(mse_aggre, 0.25)}, 75% quantile: {np.quantile(mse_aggre, 0.75)}')
    fasle_positive += np.sum(mse_aggre <= score)
    true_negative += np.sum(mse_aggre > score)
    # evaluate the accuracy: for clean, we wish all within the threshold, while for noisy, we wish all out of the threshold.
    if Clean:
        Predicted_Negative_list.append(fasle_positive / (true_negative + fasle_positive))
    else:
        Predicted_Negative_list.append(true_negative / (true_negative + fasle_positive))
    return Predicted_Negative_list

def ComputeEntropy(prob_list, n_labels, feature_length):
    dis_list_all = []
    scores = np.zeros(n_labels)
    for i in range(n_labels):  # for each feature to compute MD from the center of each action distribution (model_list)
        p = list(prob_list[i])
        p = torch.tensor(p)
        ent = -torch.sum(p * torch.log(p), dim=1) # [700, 9]
        dis_list_all.append(ent)
        try:
            scores[i] = np.quantile(ent, 0.95)  # may be empty for a certain set
        except:
            scores[i] = stats.chi2.ppf([0.95], df=feature_length)
    return dis_list_all, scores

def ComputeEntropy_conformal(Dist, scores, Clean=False):
    Predicted_Negative_list = []
    fasle_positive, true_negative = 0.0, 0.0
    for i in range(n_labels):
        threshold = scores[i]
        dis_temp = Dist[i]
        fasle_positive += sum(dis_temp <= threshold)
        true_negative += sum(dis_temp > threshold)
    # evaluate the accuracy: for clean, we wish all within the threshold, while for noisy, we wish all out of the threshold.
    if Clean:
        Predicted_Negative_list.append(fasle_positive / (true_negative + fasle_positive))
    else:
        Predicted_Negative_list.append(true_negative / (true_negative + fasle_positive))
    return Predicted_Negative_list # 95% if clean

def ComputeMD(Feature_dict, n_labels, model_list):

    dis_list_all = np.array([])
    for i in range(n_labels): # for each feature to compute MD from the center of each action distribution (model_list)
        dis_list = []
        for j in range(n_labels):
            value = np.mat(Feature_dict[i]) - model_list[j].location_ # broadcast
            x = np.matmul(np.matmul(value, model_list[j].precision_), value.T)
            diag = np.diag(x)
            dis_list.append(diag)
        #### bug: dist_list shoudl be cleared
        dis = np.min(dis_list, axis=0) # each feature i to the closest class [5120] * actions -> [5120]
        dis_list_all = np.concatenate([dis_list_all, dis])
    return dis_list_all

def ComputeComformal(Feature_dict, n_labels, model_list, feature_length, calibration=True):
    # [calibration] Feature_dict: [num_actions] -> [data_nums, dims]
    # step 1: compute MD and collect distances with actions
    dis_list_all = np.array([])
    dis_list_all_action = np.array([])
    for i in range(n_labels):  # for each feature to compute MD from the center of each action distribution (model_list)
        dis_list = []
        for j in range(n_labels):
            value = np.mat(Feature_dict[i]) - model_list[j].location_  # broadcast
            x = np.matmul(np.matmul(value, model_list[j].precision_), value.T)
            diag = np.diag(x)
            dis_list.append(diag)
        dis = np.min(dis_list, axis=0)  # each feature i to the closest class [5120] * actions -> [5120]
        dis_action = np.argmin(dis_list, axis=0) # [nums]
        dis_list_all = np.concatenate([dis_list_all, dis])
        dis_list_all_action = np.concatenate([dis_list_all_action, dis_action])
    if calibration:
        # step 2: compute quantile for MD distance
        scores = np.zeros(n_labels)
        for i in range(n_labels):
            dis_temp = dis_list_all[dis_list_all_action == i]
            try:
                scores[i] = np.quantile(dis_temp, 0.95) # may be empty for a certian set
            except:
                scores[i] = stats.chi2.ppf([0.95], df=feature_length)
    else:
        scores = None
    return dis_list_all, dis_list_all_action, scores

def ComputeAccuracy_conformal(Dist, Dist_action, scores, Clean=False):
    Predicted_Negative_list = []
    fasle_positive, true_negative = 0.0, 0.0
    for i in range(n_labels):
        threshold = scores[i]
        dis_temp = Dist[Dist_action == i]
        fasle_positive += sum(dis_temp <= threshold)
        true_negative += sum(dis_temp > threshold)
    if Clean:
        Predicted_Negative_list.append(fasle_positive / (true_negative + fasle_positive))
    else:
        Predicted_Negative_list.append(true_negative / (true_negative + fasle_positive))
    return Predicted_Negative_list

def ComputeAccuracy(Dis_list_all, p_star_list, feature_length, Clean=False):
    Predicted_Negative_list = []
    for p_star in p_star_list:
        prob = 1 - p_star
        threshold = stats.chi2.ppf([prob], df=feature_length)
        fasle_positive = sum(Dis_list_all <= threshold)
        true_negative = sum(Dis_list_all > threshold)
        if Clean:
            Predicted_Negative_list.append(fasle_positive / (true_negative + fasle_positive))
        else:
            Predicted_Negative_list.append(true_negative / (true_negative + fasle_positive))
    return Predicted_Negative_list

def save_plot(p_star_list, data, curve_label, xlabel, ylabel, title, filename, subfigure):
    # p_star_list = np.linspace(0.01,0.1,10)
    names = []
    for name in p_star_list:
        names.append(str(np.round(name, 2)))
    x = p_star_list
    # fig = plt.figure(figsize=(int(50 * len(p_star_list) / 100), 5))
    # ax = fig.add_subplot(111)
    plt.subplot(subfigure)
    plt.ylim(0, 1.1)
    for i in range(len(data)):
        plt.semilogx(x, data[i], marker='.', markersize=0.1, mec='r', mfc='w', label=curve_label[i], linewidth=3)
    plt.legend()
    # plt.xticks(x, names, rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)

    plt.xlabel(xlabel, size=15)
    if subfigure == 121:
        plt.ylabel(ylabel, size=15)
    plt.title(title, size=15)
    plt.savefig('log/' + path + '/' + filename + covariance_estimator + '.png', bbox_inches='tight')
    # plt.close()

def compute_perturbation(states, adv_epsilon, adv_stepsize, agent):
    state0 = torch.from_numpy(states).to(device).detach()
    random_noise = np.random.uniform(-adv_epsilon, adv_epsilon, size=(84, 84, 4))
    state_pgd = deepcopy(states) + random_noise
    state_pgd = torch.from_numpy(state_pgd).unsqueeze(0).to(device).float() # [1, 84, 84, 4]
    state_pgd.requires_grad = True
    with torch.enable_grad():
        policy_head, values, feature_vectors = agent.net(state_pgd)
        y = policy_head.probs.argmin(1).long().to(device).detach()
        loss0 = nn.CrossEntropyLoss()(policy_head.probs, y)  # [1,4], [1]
    loss0.backward()
    eta = adv_stepsize * state_pgd.grad.data.sign()
    state_pgd = state_pgd.data + eta
    eta = torch.clamp(state_pgd.data - state0.data, -adv_epsilon, adv_epsilon)
    state_pgd = state0.data + eta
    state_pgd.requires_grad = True

    # for _ in range(3):
    #     with torch.enable_grad():
    #         policy_head, values, feature_vectors = agent.net(state_pgd)
    #         y = policy_head.probs.argmin(1).long().to(device).detach()
    #         loss0 = nn.CrossEntropyLoss()(policy_head.probs, y)  # [1,4], [1]
    #     loss0.backward()
    #     eta = adv_stepsize * state_pgd.grad.data.sign()
    #     state_pgd = state_pgd.data + eta
    #     eta = torch.clamp(state_pgd.data - state0.data, -adv_epsilon, adv_epsilon)
    #     state_pgd = state0.data + eta
    #     state_pgd.requires_grad = True
    states = state_pgd.detach()
    return states.cpu().numpy()

def t_SNE(data, labels, n_labels):
    def plot_embedding(result, label, title, filename):  # [1083,2] [1083]
        x_min, x_max = np.min(result, 0), np.max(result, 0)
        data = (result - x_min) / (x_max - x_min)  # [0-1] scale
        plt.figure()
        for i in range(data.shape[0]):  # 1083
            plt.scatter(data[i, 0], data[i, 1], marker='o', color=plt.cm.Set1(label[i] / (1.0*n_labels)))
        plt.title(title)
        plt.savefig('log/' +filename +'.png', bbox_inches='tight')
        plt.close()

    tsne = TSNE(n_components=2, init='pca', random_state=0)  # n_components: 64 -> 2；
    result = tsne.fit_transform(data)  # [1083,64]-->[1083,2]
    # result_MD = tsne.fit_transform(data_MD)  # [1083,64]-->[1083,2]
    plot_embedding(result, labels, 't-SNE Visualization of Feature Vectors', 'tSNE')
    # plot_embedding(result_MD, labels, 't-SNE Visualization of Feature Vectors after MD', 'tSNE_MD')


def OOD_read(ood='rain'):
    img_num = 10000
    img_list = []
    if ood=='rain':
        path_ = 'images_town_clear_to_rainy'
        # path_ = 'images_lane_clear_to_rainy'
    else: # night
        path_ = 'images_town_day_to_night'
        # path_ = 'images_lane_day_to_night'
    print('working on the OOD: ', path_)
    for i in range(img_num):
        # img = cv2.imread('model/new_carla_data_lane/'+path_+'/img_' + str(i) + '.png')  # change the path here for ood images
        img = cv2.imread('model/new_carla_data_town/'+path_+'/img_' + str(i) + '.png')  # change the path here for ood images
        img = img[:, :, ::-1]  # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # RGB to gray scale
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)  # resize to 84x84
        img = np.expand_dims(img, -1)  # add channel dimension
        img_list.append(img)

    # frame stacking
    state_list = []
    stackedobs = np.zeros((84, 84, 4), dtype=np.uint8)
    for i in range(img_num):
        stackedobs = np.roll(stackedobs, shift=-1, axis=-1)
        stackedobs[..., -img_list[i].shape[-1]:] = img_list[i]
        state_list.append(copy.deepcopy(stackedobs))
        if i > 0 and terminal[i - 1]:
            stackedobs = np.zeros((84, 84, 4), dtype=np.uint8)  # reset the stackedobs

    state_list = np.array(state_list)
    state_list = state_list.astype(np.float32) / 255.0  # normalize to [0,1]
    return state_list # (10000, 84, 84, 4)

for feature_length in PCA_length:
    for contaminated_ratio in Experiments['contaminated_ratio']:
        # establish the file
        path = ''
        for item in ATARI_NAME_list:
            path += item[0]

        if use_PCA:
            path += '_PCA'
            path += str(feature_length)

        path += '_ratio'
        path += str(contaminated_ratio)

        # if not os.path.exists('log/'+path):
        #     os.makedirs('log/'+path)

        # compute and plot in each file
        # ATARI_NAME_list_focus = ATARI_NAME_list
        All_loop = len(Experiments['feature']) * len(Experiments['contaminated_ratio']) * len(ATARI_NAME_list_focus)
        # for ATARI_NAME in ATARI_NAME_list:
        for ATARI_NAME in ATARI_NAME_list_focus: # focus on breakout ###########################
            loop += 1
            print('#########################################################################################')
            print('Loop {}/{}, Game: {}, PCA:{}, Contaminated Ratio: {} '.format(loop, All_loop, ATARI_NAME, feature_length, contaminated_ratio), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            if ATARI_NAME == 'carla':
                # option 1: use the atari games as the OOD
                noise_ATARI_NAME_list = Experiments['carla_OOD']
                # option 2: use the rainy and night as OOD

            else:
                noise_ATARI_NAME_list = copy.copy(ATARI_NAME_list)
                noise_ATARI_NAME_list.remove(ATARI_NAME) # other environment as OOD

            train = False
            render = False

            ######################## (1) load model, initialization
            print('Step 1: Load Model',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            # args.batch_env = Baselines_DummyVecEnv(env_id=ATARI_NAME,num_env=args.NUMBER_ENV) # batch size: args.NUMBER_ENV
            # agent = PPO_Agent(args.batch_env.action_space,args.batch_env.observation_space,nature_cnn)
            ACTION_SPACE = 9
            agent = PPO_Agent(action_space=ACTION_SPACE,state_space=[84, 84, 4],net=nature_cnn)
            # if torch.cuda.is_available():
            #     agent.load_model('model/'+ATARI_NAME+'_0.00025_1000.pth')
            # else:
            #     agent.load_model('model/'+ATARI_NAME+'_0.00025_1000.pth', GPU=False)
            setting = 4  # [4, 5]: based on rainy and night images on town or lane： 4 town is for the current setting

            if setting == 4:
                model_path = 'model/new_carla_data_town/carla-town-v0_10_1000000_1000'
                data_path = 'model/new_carla_data_town/carla-town-v0_10_1000000_1000'
                data_num = 700  # town
                game_type = 'town'
            else: # setting = 5
                model_path = 'model/new_carla_data_lane/carla-town-v0_10_1000000_1000'
                data_path = 'model/new_carla_data_lane/carla-lane-v0_10_1000000_1000'
                data_num = 700  # lane: 5 for the lane model (imbalanced action) > we thus use the town model to generate more diverse action
                game_type = 'lane'

            agent.load_model(model_path, GPU=False)

            n_labels = 9
            feature_dict = {}
            state_dict = {}
            nextstate_dict = {}  # for envmodel detection
            OOD_feature_dict_rain = {}
            OOD_feature_dict_night = {}
            OOD_state_dict_rain = {}
            OOD_state_dict_night = {}
            OOD_nextstate_dict_rain = {}
            OOD_nextstate_dict_night = {}
            OOD_prob_dict_rain = {}
            OOD_prob_dict_night = {}

            def current_num():
                num = [0]*n_labels
                for i in range(n_labels):
                    num[i] = len(feature_dict[i])
                return num

            ######################## (2) interact clean data in feature_dict, state_dict
            print('Step 2: Collect Clean Data ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            ################# for offline data in carla


            data = np.load(data_path + '.npz')
            state = data['state'] # [10000, 84, 84, 4]
            action = data['action'] # (10000,)
            reward = data['reward'] # (99999,)
            terminal = data['terminal']
            print('------- read the rain and night images -----------',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            state_rain = OOD_read('rain') # ood data 1 [10000, 84, 84, 4]
            state_night = OOD_read('night') # ood data 2 [10000, 84, 84, 4]

            # construct the action data via the town model based on the lane images
            if setting == 5:
                action, _ = agent.act(states=state, rewards=None, dones=terminal, info=None, train=False, current_step=None)
                l = [action[action==i].shape for i in range(n_labels)]
                data_num = int(np.min(l) - 1)
                print('data num: ', data_num)
            print('------- finish reading the rain and night images -----------',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            for i in range(n_labels):
                feature_dict[i] = deque(maxlen=data_num)
                state_dict[i] = deque(maxlen=data_num)
                OOD_feature_dict_rain[i] = deque(maxlen=data_num)
                OOD_feature_dict_night[i] = deque(maxlen=data_num)
                OOD_state_dict_rain[i] = deque(maxlen=data_num)
                OOD_state_dict_night[i] = deque(maxlen=data_num)
                OOD_prob_dict_rain[i] = deque(maxlen=data_num)
                OOD_prob_dict_night[i] = deque(maxlen=data_num)
                nextstate_dict[i] = deque(maxlen=data_num)
                OOD_nextstate_dict_rain[i] = deque(maxlen=data_num)
                OOD_nextstate_dict_night[i] = deque(maxlen=data_num)

            list_rain, list_night = [], []

            for i in range(n_labels):
                # keep the same index
                Index = np.random.choice(len(action[action==i]), size=data_num, replace=False) # random sample
                states = state[action==i][Index] # [700, 84, 84, 4]
                states_rain = state_rain[action==i][Index] # [700, 84, 84, 4]
                states_night = state_night[action==i][Index] # [700, 84, 84, 4]
                _, feature_vectors, _ = agent.act(states=states, rewards=None, dones=terminal, info=None, train=False, current_step=None)
                _, feature_vectors_rain, probs_rain = agent.act(states=states_rain, rewards=None, dones=terminal, info=None, train=False, current_step=None)
                _, feature_vectors_night, probs_night = agent.act(states=states_night, rewards=None, dones=terminal, info=None, train=False, current_step=None)
                list_rain.append(feature_vectors_rain)
                list_night.append(feature_vectors_night)
                for j in range(states.shape[0]): # 120
                    feature_dict[i].append(feature_vectors[j])  # feature [j] <- current features [i]
                    state_dict[i].append(states[j])
                    OOD_state_dict_rain[i].append(states_rain[j])
                    OOD_state_dict_night[i].append(states_night[j])
                    OOD_prob_dict_rain[i].append(probs_rain[j])
                    OOD_prob_dict_night[i].append(probs_night[j])
                    if j < states.shape[0] - 1:
                        nextstate_dict[i].append(states[j+1])
                        OOD_nextstate_dict_rain[i].append(states_rain[j+1])
                        OOD_nextstate_dict_night[i].append(states_night[j+1])
                    else:
                        nextstate_dict[i].append(states[j]) # the last state is the same as the last state
                        OOD_nextstate_dict_rain[i].append(states_rain[j])
                        OOD_nextstate_dict_night[i].append(states_night[j])

            temp_p_list = [0.05]
            p_star_list = np.array(temp_p_list) # basic setting
            ##### PCA on all the data
            if use_PCA:
                pca, clean_feature_dict = ApplyPCA(feature_dict, feature_length, data_num, n_labels)
            else:
                pca = None
                clean_feature_dict = deepcopy(feature_dict)


            for i in range(n_labels):
                if use_PCA:
                    feature_vectors_rain = pca.transform(list_rain[i])
                    feature_vectors_night = pca.transform(list_night[i])
                else:
                    feature_vectors_rain = list_rain[i]
                    feature_vectors_night = list_night[i]
                for j in range(data_num):
                    OOD_feature_dict_rain[i].append(feature_vectors_rain[j]) # (50,)
                    OOD_feature_dict_night[i].append(feature_vectors_night[j])

            if 'Envmodel_conformal' in Experiments['covariance']:
                # state_dict[i]: [5120*10, 84, 84, 4]
                # print('For each action class: the state dimension: ', torch.tensor(list(state_dict[0])).shape)

                model = EnsembleModel(action_dim=n_labels, model_num=5)
                if torch.cuda.is_available():
                    model.load_state_dict(torch.load('model/autoencoder_{}.pth'.format(game_type)))
                else:
                    model.load_state_dict(torch.load('model/autoencoder_{}.pth'.format(game_type), map_location=torch.device('cpu')))

                # X_action = torch.arange(n_labels).repeat_interleave(data_num)
                # X_action = F.one_hot(X_action, num_classes=n_labels).float().to(model.device)

                X_action = torch.from_numpy(action).to(model.device)
                X_action = F.one_hot(X_action, num_classes=n_labels).float().to(model.device)  # [data_num*n_labels, n_labels]

                l_state = [torch.tensor(list(state_dict[i])) for i in range(n_labels)]
                l_nextstate = [torch.tensor(list(nextstate_dict[i])) for i in range(n_labels)]
                X_state = torch.cat(l_state, dim=0).to(model.device)  # [data_num*n_labels, 84, 84, 4]
                Y_nextstate = torch.cat(l_nextstate, dim=0).to(model.device)

                score = ComputeEnvmodel(X_state, Y_nextstate, X_action, model)

            ######################## (3) random noise setting ################
            print('Step 3: Detecting random noise! ...', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            if not Flag_Random:
                noisy_std_list = []
            else:
                if setting == 4:
                    noisy_std_list = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]  # for new generated images
                else: # lane
                    noisy_std_list = [0.1,  0.2, 0.3,  0.4,  0.5]  # for new generated images


            p_star_Predicted_Negative_list_Empirical = []  # save all data
            p_star_Predicted_Positive_list_Empirical = []  # save all data
            p_star_Predicted_Negative_list_MCD = []  # save all data
            p_star_Predicted_Positive_list_MCD = []  # save all data
            p_star_Predicted_Negative_list_Euclidean_I = []  # save all data
            p_star_Predicted_Positive_list_Euclidean_I = []  # save all data
            p_star_Predicted_Negative_list_Euclidean_Diag = []  # save all data
            p_star_Predicted_Positive_list_Euclidean_Diag = []  # save all data
            p_star_Predicted_Negative_list_TMD = []  # save all data
            p_star_Predicted_Positive_list_TMD = []  # save all data
            p_star_Predicted_Negative_list_Empirical_conformal = []  # save all data
            p_star_Predicted_Positive_list_Empirical_conformal = []  # save all data
            p_star_Predicted_Negative_list_MCD_conformal = []  # save all data
            p_star_Predicted_Positive_list_MCD_conformal = []  # save all data
            p_star_Predicted_Negative_list_Entropy_conformal = []  # save all data
            p_star_Predicted_Positive_list_Entropy_conformal = []  # save all data
            p_star_Predicted_Negative_list_Envmodel_conformal = []  # save all data
            p_star_Predicted_Positive_list_Envmodel_conformal = []  # save all data


            for noisy_std in noisy_std_list:  # for each noise strength
                print('For noise strength {}'.format(noisy_std),time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
                # (3.1) construct all noisy data
                noisy_feature_dict = {}
                noisy_feature_dict_evaluation = {}
                noisy_state_dict = {}
                probs_dict = {}
                probs_dict_evaluation = {}
                for i in range(n_labels):
                    noisy_feature_dict[i] = deque(maxlen=data_num)
                    noisy_feature_dict_evaluation[i] = deque(maxlen=data_num)
                    noisy_state_dict[i] = deque(maxlen=data_num)
                    probs_dict[i] = deque(maxlen=data_num)
                    probs_dict_evaluation[i] = deque(maxlen=data_num)
                    for j in range(data_num):
                        noise = np.random.normal(loc=0, scale=noisy_std, size=(84, 84, 4))
                        noise_state_evaluation = noise + state_dict[i][j]
                        if np.random.rand() < contaminated_ratio:
                            noise_state = noise + state_dict[i][j] # [n_labeles/num_action=4, 5120*10, 84, 84, 4(channel)]
                        else:
                            noise_state = state_dict[i][j] # [n_labeles/num_action=4, 5120*10, 84, 84, 4(channel)]
                        a, f, probs = agent.act([noise_state], None, None, None, train=train, current_step=0)  # to query the feature vectors
                        a_, f_, probs_ = agent.act([noise_state_evaluation], None, None, None, train=train, current_step=0)  # to query the feature vectors
                        probs_dict[i].append(probs.squeeze())
                        probs_dict_evaluation[i].append(probs_.squeeze())
                        noisy_state_dict[i].append(noise_state_evaluation.squeeze())  # for envmodel detection

                        if use_PCA:
                            noisy_feature_dict[i].append(pca.transform(f).squeeze())
                            noisy_feature_dict_evaluation[i].append(pca.transform(f_).squeeze())
                        else:
                            noisy_feature_dict[i].append(f.squeeze())
                            noisy_feature_dict_evaluation[i].append(f_.squeeze())
                # (3.2) PCA transformation on the contaminated data
                # for i in range(n_labels):
                #     noisy_feature_dict[i] = pca.transform(np.array(noisy_feature_dict[i]))
                #     noisy_feature_dict_evaluation[i] = pca.transform(np.array(noisy_feature_dict_evaluation[i]))
                for covariance_estimator in Experiments['covariance']:
                    # (3.3) estimate the mean and variance via two methods
                    model_list = MeanVarEstimate(noisy_feature_dict, n_labels, covariance_estimator)

                    ############ Evaluation on balanced data
                    # (3.4) 50% clean data to compute the MD again
                    if covariance_estimator == 'Envmodel_conformal':
                        Predicted_Positive_list = ComputeEnvmodel_conformal(score, X_state, Y_nextstate, X_action,model, Clean=True)
                    elif covariance_estimator == 'Entropy_conformal':
                        Dist, scores = ComputeEntropy(probs_dict, n_labels, feature_length)
                        Predicted_Positive_list = ComputeEntropy_conformal(Dist, scores, Clean=True) # 95%
                    elif covariance_estimator in ['MD_conformal', 'RMD_conformal']:
                        Dist, Dist_action, scores = ComputeComformal(clean_feature_dict, n_labels, model_list, feature_length, calibration=True)
                        Predicted_Positive_list = ComputeAccuracy_conformal(Dist, Dist_action, scores, Clean=True)
                    else:
                        Dist_list_all = ComputeMD(clean_feature_dict, n_labels, model_list)
                        Predicted_Positive_list = ComputeAccuracy(Dist_list_all, p_star_list, feature_length, Clean=True) # < threshold is true
                    if covariance_estimator == 'Empirical':
                        p_star_Predicted_Positive_list_Empirical.append(Predicted_Positive_list)
                    elif covariance_estimator == 'MinCovDet':
                        p_star_Predicted_Positive_list_MCD.append(Predicted_Positive_list)
                    elif covariance_estimator == 'Euclidean_I':
                        p_star_Predicted_Positive_list_Euclidean_I.append(Predicted_Positive_list)
                    elif covariance_estimator == 'Euclidean_Diag':
                        p_star_Predicted_Positive_list_Euclidean_Diag.append(Predicted_Positive_list)
                    elif covariance_estimator == 'TMD':
                        p_star_Predicted_Positive_list_TMD.append(Predicted_Positive_list)
                    elif covariance_estimator == 'MD_conformal':
                        p_star_Predicted_Positive_list_Empirical_conformal.append(Predicted_Positive_list)
                    elif covariance_estimator == 'RMD_conformal':
                        p_star_Predicted_Positive_list_MCD_conformal.append(Predicted_Positive_list)
                    elif covariance_estimator == 'Entropy_conformal':
                        p_star_Predicted_Positive_list_Entropy_conformal.append(Predicted_Positive_list)
                    elif covariance_estimator == 'Envmodel_conformal':
                        p_star_Predicted_Positive_list_Envmodel_conformal.append(Predicted_Positive_list)
                    else:
                        pass
                    # (3.5) 50% new features to compute the MD again
                    if covariance_estimator == 'Envmodel_conformal':
                        l_noisystate = [torch.tensor(list(noisy_state_dict[i])) for i in range(n_labels)]
                        X_noisystate = torch.cat(l_noisystate, dim=0).float().to(model.device)  # [data_num*n_labels, 84, 84, 4]
                        Predicted_Negative_list = ComputeEnvmodel_conformal(score, X_noisystate, Y_nextstate, X_action, model, Clean=False)
                    elif covariance_estimator == 'Entropy_conformal':
                        Dist, scores = ComputeEntropy(probs_dict_evaluation, n_labels, feature_length)
                        Predicted_Negative_list = ComputeEntropy_conformal(Dist, scores, Clean=False)
                    elif covariance_estimator in ['MD_conformal', 'RMD_conformal']:
                        Dist, Dist_action, _ = ComputeComformal(noisy_feature_dict_evaluation, n_labels, model_list, feature_length, calibration=False)
                        Predicted_Negative_list = ComputeAccuracy_conformal(Dist, Dist_action, scores, Clean=False) # score is based on calibration
                    else:
                        Dist_list_all = ComputeMD(noisy_feature_dict_evaluation, n_labels, model_list)
                        Predicted_Negative_list = ComputeAccuracy(Dist_list_all, p_star_list, feature_length, Clean=False)
                    # for plot 1*2
                    if covariance_estimator == 'Empirical':
                        p_star_Predicted_Negative_list_Empirical.append(Predicted_Negative_list)
                    elif covariance_estimator == 'MinCovDet':
                        p_star_Predicted_Negative_list_MCD.append(Predicted_Negative_list)
                    elif covariance_estimator == 'Euclidean_I':
                        p_star_Predicted_Negative_list_Euclidean_I.append(Predicted_Negative_list)
                    elif covariance_estimator == 'Euclidean_Diag':
                        p_star_Predicted_Negative_list_Euclidean_Diag.append(Predicted_Negative_list)
                    elif covariance_estimator == 'TMD':
                        p_star_Predicted_Negative_list_TMD.append(Predicted_Negative_list)
                    elif covariance_estimator == 'MD_conformal':
                        p_star_Predicted_Negative_list_Empirical_conformal.append(Predicted_Negative_list)
                    elif covariance_estimator == 'RMD_conformal':
                        p_star_Predicted_Negative_list_MCD_conformal.append(Predicted_Negative_list)
                    elif covariance_estimator == 'Entropy_conformal':
                        p_star_Predicted_Negative_list_Entropy_conformal.append(Predicted_Negative_list)
                    elif covariance_estimator == 'Envmodel_conformal':
                        p_star_Predicted_Negative_list_Envmodel_conformal.append(Predicted_Negative_list)
                    else:
                        pass
            # (3.6) plot
            if Flag_Random:
                print('Plotting...........')
                xlabel = 'p*'
                curve_label = []
                for std in noisy_std_list:
                    curve_label.append('std = ' + str(std))
                title = ATARI_NAME + ' \n Random Noise '
                # (3.6.1) positive: Empirical and MCD
                Predicted_Positive_Empirical = np.array(p_star_Predicted_Positive_list_Empirical)
                Predicted_Positive_MCD = np.array(p_star_Predicted_Positive_list_MCD)
                ylabel = 'Clean/Positive Accuracy'

                filename = ATARI_NAME + '_Random_CleanData'

                # (3.6.2) Negative: Empirical and MCD
                Predicted_Negative_Empirical = np.array(p_star_Predicted_Negative_list_Empirical)
                Predicted_Negative_MCD = np.array(p_star_Predicted_Negative_list_MCD)
                ylabel = 'Noisy/Negative Accuracy'
                filename = ATARI_NAME + '_Random_Negative'

                # (3.6.3) Overall: Empirical and MCD
                Predicted_Empirical = (np.array(p_star_Predicted_Positive_list_Empirical) + np.array(p_star_Predicted_Negative_list_Empirical)) / 2
                Predicted_MCD = (np.array(p_star_Predicted_Positive_list_MCD) + np.array(p_star_Predicted_Negative_list_MCD)) / 2
                Predicted_Euclidean_I = (np.array(p_star_Predicted_Positive_list_Euclidean_I) + np.array(p_star_Predicted_Negative_list_Euclidean_I)) / 2
                Predicted_Euclidean_Diag = (np.array(p_star_Predicted_Positive_list_Euclidean_Diag) + np.array(p_star_Predicted_Negative_list_Euclidean_Diag)) / 2
                Predicted_TMD = (np.array(p_star_Predicted_Positive_list_TMD) + np.array(p_star_Predicted_Negative_list_TMD)) / 2
                Predicted_EmpiricalConformal = (np.array(p_star_Predicted_Positive_list_Empirical_conformal) + np.array(p_star_Predicted_Negative_list_Empirical_conformal)) / 2
                Predicted_MCDConformal = (np.array(p_star_Predicted_Positive_list_MCD_conformal) + np.array(p_star_Predicted_Negative_list_MCD_conformal)) / 2
                Predicted_EntropyConformal = (np.array(p_star_Predicted_Positive_list_Entropy_conformal) + np.array(p_star_Predicted_Negative_list_Entropy_conformal)) / 2
                Predicted_EnvmodelConformal = (np.array(p_star_Predicted_Positive_list_Envmodel_conformal) + np.array(p_star_Predicted_Negative_list_Envmodel_conformal)) / 2

                ylabel = 'Overall Detection Accuracy'
                filename = ATARI_NAME + '_Random_Detection'
                if Flag_Write:
                    # loop in different noises
                    for i in range(len(Predicted_Empirical)): # issue, might impatible
                        newpd_em = pd.DataFrame({'Emp_All_'+curve_label[i]: Predicted_Empirical[i]})
                        newpd_MCD = pd.DataFrame({'MCD_All_'+curve_label[i]: Predicted_MCD[i]})
                        newpd_Euclidean_I = pd.DataFrame({'EucI_All_'+curve_label[i]: Predicted_Euclidean_I[i]})
                        newpd_Euclidean_Diag = pd.DataFrame({'EucDiag_All_'+curve_label[i]: Predicted_Euclidean_Diag[i]})
                        newpd_TMD = pd.DataFrame({'TMD_All_'+curve_label[i]: Predicted_TMD[i]})
                        newpd_emconformal = pd.DataFrame({'EmpConformal_All_'+curve_label[i]: Predicted_EmpiricalConformal[i]})
                        # newpd_MCDconformal = pd.DataFrame({'MCDConformal_All_'+curve_label[i]: Predicted_MCDConformal[i]})
                        newpd_EntropyConformal = pd.DataFrame({'EntConformal_All_' + curve_label[i]: Predicted_EntropyConformal[i]})
                        newpd_EnvmodelConformal = pd.DataFrame({'EnvmodelConformal_All_' + curve_label[i]: Predicted_EnvmodelConformal[i]})

                        Final_result = pd.concat([Final_result, newpd_em], axis=1)
                        Final_result = pd.concat([Final_result, newpd_MCD], axis=1)
                        Final_result = pd.concat([Final_result, newpd_Euclidean_I], axis=1)
                        Final_result = pd.concat([Final_result, newpd_Euclidean_Diag], axis=1)
                        Final_result = pd.concat([Final_result, newpd_TMD], axis=1)
                        Final_result = pd.concat([Final_result, newpd_emconformal], axis=1)
                        # Final_result = pd.concat([Final_result, newpd_MCDconformal], axis=1)
                        Final_result = pd.concat([Final_result, newpd_EntropyConformal], axis=1)
                        Final_result = pd.concat([Final_result, newpd_EnvmodelConformal], axis=1)

                else:
                    plt.figure()
                    save_plot(p_star_list, Predicted_Empirical, curve_label, xlabel, ylabel, title + 'Empirical', filename, 121)
                    save_plot(p_star_list, Predicted_MCD, curve_label, xlabel, ylabel, title + 'MCD', filename, 122)
                    plt.close()

            ########################### (4) adversarial noise setting ################
            print('Step 4: Detecting adversarial noise! ...', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            if not Flag_ADV:
                adv_epsilon_list = []
            else:
                adv_epsilon_list = [1e-5, 1e-4, 1e-3, 1e-2, 0.1] # for new generated dataset
            p_star_Predicted_Negative_adv_list_Empirical = []  # save all data
            p_star_Predicted_Positive_adv_list_Empirical = []  # save all data
            p_star_Predicted_Negative_adv_list_MCD = []  # save all data
            p_star_Predicted_Positive_adv_list_MCD = []  # save all data
            p_star_Predicted_Negative_adv_list_Euclidean_I = []  # save all data
            p_star_Predicted_Positive_adv_list_Euclidean_I = []  # save all data
            p_star_Predicted_Negative_adv_list_Euclidean_Diag = []  # save all data
            p_star_Predicted_Positive_adv_list_Euclidean_Diag = []  # save all data
            p_star_Predicted_Negative_adv_list_TMD = []  # save all data
            p_star_Predicted_Positive_adv_list_TMD = []  # save all data
            p_star_Predicted_Negative_adv_list_EmpiricalConformal = []  # save all data
            p_star_Predicted_Positive_adv_list_EmpiricalConformal = []  # save all data
            p_star_Predicted_Negative_adv_list_MCDConformal = []  # save all data
            p_star_Predicted_Positive_adv_list_MCDConformal = []  # save all data
            p_star_Predicted_Negative_adv_list_EntropyConformal = []  # save all data
            p_star_Predicted_Positive_adv_list_EntropyConformal = []  # save all data
            p_star_Predicted_Negative_adv_list_EnvmodelConformal = []  # save all data
            p_star_Predicted_Positive_adv_list_EnvmodelConformal = []  # save all data

            for adv_epsilon in adv_epsilon_list:  # for each noise strength
                adv_stepsize = adv_epsilon * 1.0 / 4 # 40
                print('For adversarial epsilon strength {}'.format(adv_epsilon), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
                # (4.1) construct all noisy data
                adv_feature_dict = {}
                adv_feature_dict_evaluation = {}
                adv_state_dict = {}
                probs_dict = {}
                probs_dict_evaluation = {}
                for i in range(n_labels):
                    adv_feature_dict[i] = deque(maxlen=data_num)
                    adv_feature_dict_evaluation[i] = deque(maxlen=data_num)
                    adv_state_dict[i] = deque(maxlen=data_num)
                    probs_dict[i] = deque(maxlen=data_num)
                    probs_dict_evaluation[i] = deque(maxlen=data_num)
                    for j in range(data_num):
                        adv_perturbation = compute_perturbation(state_dict[i][j], adv_epsilon, adv_stepsize, agent)  # [84, 84, 4]
                        adv_state_evaluation = state_dict[i][j] + adv_perturbation
                        adv_state_evaluation = adv_state_evaluation.squeeze(0)
                        if np.random.rand() < contaminated_ratio:
                            adv_state = state_dict[i][j] + adv_perturbation
                            adv_state = adv_state.squeeze(0) #[1,84,84,4]-> [84,84,4]
                        else:
                            adv_state = state_dict[i][j]  # [n_labeles/num_action=4, 5120*10, 84, 84, 4(channel)]
                        a, f, probs = agent.act([adv_state], None, None, None, train=train, current_step=0)  # to query the feature vectors
                        a_, f_, probs_ = agent.act([adv_state_evaluation], None, None, None, train=train, current_step=0)  # to query the feature vectors
                        probs_dict[i].append(probs.squeeze())
                        probs_dict_evaluation[i].append(probs_.squeeze())
                        adv_feature_dict[i].append(pca.transform(f).squeeze())
                        adv_state_dict[i].append(adv_state_evaluation.squeeze())
                        adv_feature_dict_evaluation[i].append(pca.transform(f_).squeeze())

                # (4.2) PCA transformation on the contaminated data
                # for i in range(n_labels):
                #     adv_feature_dict[i] = pca.transform(np.array(adv_feature_dict[i]))
                #     adv_feature_dict_evaluation[i] = pca.transform(np.array(adv_feature_dict_evaluation[i]))
                for covariance_estimator in Experiments['covariance']:
                    # (4.3) estimate the mean and variance via two methods
                    model_list = MeanVarEstimate(adv_feature_dict, n_labels, covariance_estimator)

                    # (4.4) 50% clean data to compute the MD again
                    if covariance_estimator == 'Envmodel_conformal':
                        Predicted_Positive_adv_list = ComputeEnvmodel_conformal(score, X_state, Y_nextstate, X_action, model, Clean=True)
                    elif covariance_estimator == 'Entropy_conformal':
                        Dist, scores = ComputeEntropy(probs_dict, n_labels, feature_length)
                        Predicted_Positive_adv_list = ComputeEntropy_conformal(Dist, scores, Clean=True)
                    elif covariance_estimator in ['MD_conformal', 'RMD_conformal']:
                        Dist, Dist_action, scores = ComputeComformal(clean_feature_dict, n_labels, model_list, feature_length, calibration=True)
                        Predicted_Positive_adv_list = ComputeAccuracy_conformal(Dist, Dist_action, scores, Clean=True)
                    else:
                        Dist_list_all = ComputeMD(clean_feature_dict, n_labels, model_list)
                        Predicted_Positive_adv_list = ComputeAccuracy(Dist_list_all, p_star_list, feature_length, Clean=True)  # < threshold is true

                    if covariance_estimator == 'Empirical':
                        p_star_Predicted_Positive_adv_list_Empirical.append(Predicted_Positive_adv_list)
                    elif covariance_estimator == 'MinCovDet':
                        p_star_Predicted_Positive_adv_list_MCD.append(Predicted_Positive_adv_list)
                    elif covariance_estimator == 'Euclidean_I':
                        p_star_Predicted_Positive_adv_list_Euclidean_I.append(Predicted_Positive_adv_list)
                    elif covariance_estimator == 'Euclidean_Diag':
                        p_star_Predicted_Positive_adv_list_Euclidean_Diag.append(Predicted_Positive_adv_list)
                    elif covariance_estimator == 'TMD':
                        p_star_Predicted_Positive_adv_list_TMD.append(Predicted_Positive_adv_list)
                    elif covariance_estimator == 'MD_conformal':
                        p_star_Predicted_Positive_adv_list_EmpiricalConformal.append(Predicted_Positive_adv_list)
                    elif covariance_estimator == 'RMD_conformal':  # RMD_conformal
                        p_star_Predicted_Positive_adv_list_MCDConformal.append(Predicted_Positive_adv_list)
                    elif covariance_estimator == 'Entropy_conformal':
                        p_star_Predicted_Positive_adv_list_EntropyConformal.append(Predicted_Positive_adv_list)
                    elif covariance_estimator == 'Envmodel_conformal':
                        p_star_Predicted_Positive_adv_list_EnvmodelConformal.append(Predicted_Positive_adv_list)
                    else:
                        pass

                    # (4.5) 50% new features to compute the MD again
                    if covariance_estimator == 'Envmodel_conformal':
                        l_advstate = [torch.tensor(list(adv_state_dict[i])) for i in range(n_labels)]
                        X_advstate = torch.cat(l_advstate, dim=0).to(model.device)  # [data_num*n_labels, 84, 84, 4]
                        Predicted_Negative_adv_list = ComputeEnvmodel_conformal(score, X_advstate, Y_nextstate, X_action, model, Clean=False)

                    elif covariance_estimator == 'Entropy_conformal':
                        Dist, scores = ComputeEntropy(probs_dict_evaluation, n_labels, feature_length)
                        Predicted_Negative_adv_list = ComputeEntropy_conformal(Dist, scores, Clean=False)
                    elif covariance_estimator in ['MD_conformal', 'RMD_conformal']:
                        Dist, Dist_action, _ = ComputeComformal(adv_feature_dict_evaluation, n_labels, model_list, feature_length, calibration=False)
                        Predicted_Negative_adv_list = ComputeAccuracy_conformal(Dist, Dist_action, scores, Clean=False)
                    else:
                        Dist_list_all = ComputeMD(adv_feature_dict_evaluation, n_labels, model_list)
                        Predicted_Negative_adv_list = ComputeAccuracy(Dist_list_all, p_star_list, feature_length, Clean=False)
                    # for plot 1*2
                    if covariance_estimator == 'Empirical':
                        p_star_Predicted_Negative_adv_list_Empirical.append(Predicted_Negative_adv_list)
                    # else:
                    #     p_star_Predicted_Negative_adv_list_MCD.append(Predicted_Negative_adv_list)
                    elif covariance_estimator == 'MinCovDet':
                        p_star_Predicted_Negative_adv_list_MCD.append(Predicted_Negative_adv_list)
                    elif covariance_estimator == 'Euclidean_I':
                        p_star_Predicted_Negative_adv_list_Euclidean_I.append(Predicted_Negative_adv_list)
                    elif covariance_estimator == 'Euclidean_Diag':
                        p_star_Predicted_Negative_adv_list_Euclidean_Diag.append(Predicted_Negative_adv_list)
                    elif covariance_estimator == 'TMD':
                        p_star_Predicted_Negative_adv_list_TMD.append(Predicted_Negative_adv_list)
                    elif covariance_estimator == 'MD_conformal':
                        p_star_Predicted_Negative_adv_list_EmpiricalConformal.append(Predicted_Negative_adv_list)
                    elif covariance_estimator == 'RMD_conformal':
                        p_star_Predicted_Negative_adv_list_MCDConformal.append(Predicted_Negative_adv_list)
                    elif covariance_estimator == 'Entropy_conformal':
                        p_star_Predicted_Negative_adv_list_EntropyConformal.append(Predicted_Negative_adv_list)
                    elif covariance_estimator == 'Envmodel_conformal':
                        p_star_Predicted_Negative_adv_list_EnvmodelConformal.append(Predicted_Negative_adv_list)
                    else:
                        pass
            # (3.6) plot
            if Flag_ADV:
                print('Plotting...........')
                xlabel = 'p*'
                curve_label = []
                for epsilon in adv_epsilon_list:
                    curve_label.append(r'$\epsilon$ = ' + str(epsilon))
                title = ATARI_NAME + ' \n Adversarial Noise '
                # (3.6.1) positive: Empirical and MCD
                Predicted_Positive_adv_Empirical = np.array(p_star_Predicted_Positive_adv_list_Empirical)
                Predicted_Positive_adv_MCD = np.array(p_star_Predicted_Positive_adv_list_MCD)
                ylabel = 'Clean/Positive Accuracy'

                filename = ATARI_NAME + '_Adversarial_CleanData'

                # (3.6.2) Negative: Empirical and MCD
                Predicted_Negative_adv_Empirical = np.array(p_star_Predicted_Negative_adv_list_Empirical)
                Predicted_Negative_adv_MCD = np.array(p_star_Predicted_Negative_adv_list_MCD)
                ylabel = 'Noisy/Negative Accuracy'
                filename = ATARI_NAME + '_Adversarial_Negative'

                # (3.6.3) Overall: Empirical and MCD
                Predicted_adv_Empirical = (np.array(p_star_Predicted_Positive_adv_list_Empirical) + np.array(p_star_Predicted_Negative_adv_list_Empirical)) / 2
                Predicted_adv_MCD = (np.array(p_star_Predicted_Positive_adv_list_MCD) + np.array(p_star_Predicted_Negative_adv_list_MCD)) / 2
                Predicted_adv_Euclidean_I = (np.array(p_star_Predicted_Positive_adv_list_Euclidean_I) + np.array(p_star_Predicted_Negative_adv_list_Euclidean_I)) / 2
                Predicted_adv_Euclidean_Diag = (np.array(p_star_Predicted_Positive_adv_list_Euclidean_Diag) + np.array(p_star_Predicted_Negative_adv_list_Euclidean_Diag)) / 2
                Predicted_adv_TMD = (np.array(p_star_Predicted_Positive_adv_list_TMD) + np.array(p_star_Predicted_Negative_adv_list_TMD)) / 2
                Predicted_adv_EmpiricalConformal = (np.array(p_star_Predicted_Positive_adv_list_EmpiricalConformal) + np.array(p_star_Predicted_Negative_adv_list_EmpiricalConformal)) / 2
                Predicted_adv_MCDConformal = (np.array(p_star_Predicted_Positive_adv_list_MCDConformal) + np.array(p_star_Predicted_Negative_adv_list_MCDConformal)) / 2
                Predicted_adv_EntropyConformal = (np.array(p_star_Predicted_Positive_adv_list_EntropyConformal) + np.array(p_star_Predicted_Negative_adv_list_EntropyConformal)) / 2
                Predicted_adv_EnvmodelConformal = (np.array(p_star_Predicted_Positive_adv_list_EnvmodelConformal) + np.array(p_star_Predicted_Negative_adv_list_EnvmodelConformal)) / 2

                ylabel = 'Overall Detection Accuracy'
                filename = ATARI_NAME + '_Adversarial_Detection'
                if Flag_Write:
                    for i in range(len(Predicted_adv_Empirical)):
                        newpd_em = pd.DataFrame({'Emp_All_'+curve_label[i]: Predicted_adv_Empirical[i]})
                        newpd_MCD = pd.DataFrame({'MCD_All_'+curve_label[i]: Predicted_adv_MCD[i]})
                        newpd_Euclidean_I = pd.DataFrame({'EucI_All_' + curve_label[i]: Predicted_adv_Euclidean_I[i]})
                        newpd_Euclidean_Diag = pd.DataFrame({'EucDiag_All_' + curve_label[i]: Predicted_adv_Euclidean_Diag[i]})
                        newpd_TMD = pd.DataFrame({'TMD_All_' + curve_label[i]: Predicted_adv_TMD[i]})
                        newpd_emconformal = pd.DataFrame({'EmpConformal_All_' + curve_label[i]: Predicted_adv_EmpiricalConformal[i]})
                        # newpd_MCDconformal = pd.DataFrame({'MCDConformal_All_' + curve_label[i]: Predicted_adv_MCDConformal[i]})
                        newpd_EntropyConformal = pd.DataFrame({'EntConformal_All_' + curve_label[i]: Predicted_adv_EntropyConformal[i]})
                        newpd_EnvmodelConformal = pd.DataFrame({'EnvmodelConformal_All_' + curve_label[i]: Predicted_adv_EnvmodelConformal[i]})

                        Final_result = pd.concat([Final_result, newpd_em], axis=1)
                        Final_result = pd.concat([Final_result, newpd_MCD], axis=1)
                        Final_result = pd.concat([Final_result, newpd_Euclidean_I], axis=1)
                        Final_result = pd.concat([Final_result, newpd_Euclidean_Diag], axis=1)
                        Final_result = pd.concat([Final_result, newpd_TMD], axis=1)
                        Final_result = pd.concat([Final_result, newpd_emconformal], axis=1)
                        # Final_result = pd.concat([Final_result, newpd_MCDconformal], axis=1)
                        Final_result = pd.concat([Final_result, newpd_EntropyConformal], axis=1)
                        Final_result = pd.concat([Final_result, newpd_EnvmodelConformal], axis=1)

                else:
                    plt.figure()
                    save_plot(p_star_list, Predicted_adv_Empirical, curve_label, xlabel, ylabel, title + 'Empirical', filename, 121)
                    save_plot(p_star_list, Predicted_adv_MCD, curve_label, xlabel, ylabel, title + 'MCD', filename, 122)
                    plt.close()

            ########################### (5) OOD setting ################
            print('Step 5: Detecting OOD data! ...', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            if not Flag_OOD:
                noise_ATARI_NAME_list = []

            p_star_ood_Predicted_Positive_list_Empirical = []
            p_star_ood_Predicted_Negative_list_Empirical = []
            p_star_ood_Predicted_Positive_list_MCD = []
            p_star_ood_Predicted_Negative_list_MCD = []
            p_star_ood_Predicted_Negative_list_Euclidean_I = []
            p_star_ood_Predicted_Positive_list_Euclidean_I = []
            p_star_ood_Predicted_Negative_list_Euclidean_Diag = []
            p_star_ood_Predicted_Positive_list_Euclidean_Diag = []
            p_star_ood_Predicted_Negative_list_TMD = []
            p_star_ood_Predicted_Positive_list_TMD = []
            p_star_ood_Predicted_Negative_list_EmpiricalConformal = []
            p_star_ood_Predicted_Positive_list_EmpiricalConformal = []
            p_star_ood_Predicted_Negative_list_MCDConformal = []
            p_star_ood_Predicted_Positive_list_MCDConformal = []
            p_star_ood_Predicted_Negative_list_EntropyConformal = []
            p_star_ood_Predicted_Positive_list_EntropyConformal = []
            p_star_ood_Predicted_Negative_list_EnvmodelConformal = []
            p_star_ood_Predicted_Positive_list_EnvmodelConformal = []

            for noise_ATARI_NAME in noise_ATARI_NAME_list: # remove the current env
                print('OOD data: {} '.format(noise_ATARI_NAME), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

                OOD_feature_dict = deepcopy(clean_feature_dict)
                if noise_ATARI_NAME == 'rain':
                    OOD_feature_dict_evaluation = OOD_feature_dict_rain
                    OOD_state_dict = OOD_state_dict_rain
                    OOD_probs_dict = OOD_prob_dict_rain
                    OOD_nextstate_dict = OOD_nextstate_dict_rain
                else:
                    OOD_feature_dict_evaluation = OOD_feature_dict_night
                    OOD_state_dict = OOD_state_dict_night
                    OOD_probs_dict = OOD_prob_dict_night
                    OOD_nextstate_dict = OOD_nextstate_dict_night

                for covariance_estimator in Experiments['covariance']:
                    # (5.3) estimate the mean and variance via two methods
                    model_list = MeanVarEstimate(OOD_feature_dict, n_labels, covariance_estimator)
                    # (5.4) 50% clean data to compute the MD again
                    if covariance_estimator == 'Envmodel_conformal':
                        OOD_Predicted_Positive_list = ComputeEnvmodel_conformal(score, X_state, Y_nextstate, X_action, model, Clean=True)
                    elif covariance_estimator == 'Entropy_conformal':
                        Dist, scores = ComputeEntropy(probs_dict, n_labels, feature_length)
                        OOD_Predicted_Positive_list = ComputeEntropy_conformal(Dist, scores, Clean=True)
                    elif covariance_estimator in ['MD_conformal', 'RMD_conformal']:
                        Dist, Dist_action, scores = ComputeComformal(clean_feature_dict, n_labels, model_list, feature_length, calibration=True)
                        OOD_Predicted_Positive_list = ComputeAccuracy_conformal(Dist, Dist_action, scores, Clean=True)
                    else:
                        Dist_list_all = ComputeMD(clean_feature_dict, n_labels, model_list)
                        OOD_Predicted_Positive_list = ComputeAccuracy(Dist_list_all, p_star_list, feature_length, Clean=True)
                    if covariance_estimator == 'Empirical':
                        p_star_ood_Predicted_Positive_list_Empirical.append(OOD_Predicted_Positive_list)
                    # else:
                    #     p_star_ood_Predicted_Positive_list_MCD.append(OOD_Predicted_Positive_list)

                    elif covariance_estimator == 'MinCovDet':
                        p_star_ood_Predicted_Positive_list_MCD.append(OOD_Predicted_Positive_list)
                    elif covariance_estimator == 'Euclidean_I':
                        p_star_ood_Predicted_Positive_list_Euclidean_I.append(OOD_Predicted_Positive_list)
                    elif covariance_estimator == 'Euclidean_Diag':
                        p_star_ood_Predicted_Positive_list_Euclidean_Diag.append(OOD_Predicted_Positive_list)
                    elif covariance_estimator == 'TMD':
                        p_star_ood_Predicted_Positive_list_TMD.append(OOD_Predicted_Positive_list)
                    elif covariance_estimator == 'MD_conformal':
                        p_star_ood_Predicted_Positive_list_EmpiricalConformal.append(OOD_Predicted_Positive_list)
                    elif covariance_estimator == 'RMD_conformal':
                        p_star_ood_Predicted_Positive_list_MCDConformal.append(OOD_Predicted_Positive_list)
                    elif covariance_estimator == 'Entropy_conformal':
                        p_star_ood_Predicted_Positive_list_EntropyConformal.append(OOD_Predicted_Positive_list)
                    elif covariance_estimator == 'Envmodel_conformal':
                        p_star_ood_Predicted_Positive_list_EnvmodelConformal.append(OOD_Predicted_Positive_list)
                    else:
                        pass

                    # (5.5) 50% new features to compute the MD again
                    if covariance_estimator == 'Envmodel_conformal':
                        l_oodstate = [torch.tensor(list(OOD_state_dict[i])) for i in range(n_labels)]
                        X_oodstate = torch.cat(l_oodstate, dim=0).to(device)  # [data_num*n_labels, 84, 84, 4]

                        ##### important: compare with the next OOD state, instead of the clean next state: comparison (1) f(s_t, a_t) vs s_t+1 (2) f(OOD_t, a_t) vs OOD_t+1;
                        ##### underlying reason: f() is more like the current env, which is similar to s_t+1 instead of O_t+1; thus, mse distributions can be different
                        ##### Remark: the comparison manner of OOD is different from adv and random noise
                        l_oodstate_next = [torch.tensor(list(OOD_nextstate_dict[i])) for i in range(n_labels)]
                        Y_oodstate_next = torch.cat(l_oodstate, dim=0).to(device)

                        OOD_Predicted_Negative_list = ComputeEnvmodel_conformal(score, X_oodstate, Y_oodstate_next, X_action, model, Clean=False)
                    elif covariance_estimator == 'Entropy_conformal':
                        Dist, scores = ComputeEntropy(OOD_probs_dict, n_labels, feature_length)
                        OOD_Predicted_Negative_list = ComputeEntropy_conformal(Dist, scores, Clean=False)
                    elif covariance_estimator in ['MD_conformal', 'RMD_conformal']:
                        Dist, Dist_action, _ = ComputeComformal(OOD_feature_dict_evaluation, n_labels, model_list, feature_length, calibration=False)
                        OOD_Predicted_Negative_list = ComputeAccuracy_conformal(Dist, Dist_action, scores, Clean=False)
                    else:
                        Dist_list_all = ComputeMD(OOD_feature_dict_evaluation, n_labels, model_list)
                        OOD_Predicted_Negative_list = ComputeAccuracy(Dist_list_all, p_star_list, feature_length, Clean=False)
                    # for plot 1*2
                    if covariance_estimator == 'Empirical':
                        p_star_ood_Predicted_Negative_list_Empirical.append(OOD_Predicted_Negative_list)
                    elif covariance_estimator == 'MinCovDet':
                        p_star_ood_Predicted_Negative_list_MCD.append(OOD_Predicted_Negative_list)
                    elif covariance_estimator == 'Euclidean_I':
                        p_star_ood_Predicted_Negative_list_Euclidean_I.append(OOD_Predicted_Negative_list)
                    elif covariance_estimator == 'Euclidean_Diag':
                        p_star_ood_Predicted_Negative_list_Euclidean_Diag.append(OOD_Predicted_Negative_list)
                    elif covariance_estimator == 'TMD':
                        p_star_ood_Predicted_Negative_list_TMD.append(OOD_Predicted_Negative_list)
                    elif covariance_estimator == 'MD_conformal':
                        p_star_ood_Predicted_Negative_list_EmpiricalConformal.append(OOD_Predicted_Negative_list)
                    elif covariance_estimator == 'RMD_conformal':
                        p_star_ood_Predicted_Negative_list_MCDConformal.append(OOD_Predicted_Negative_list)
                    elif covariance_estimator == 'Entropy_conformal':
                        p_star_ood_Predicted_Negative_list_EntropyConformal.append(OOD_Predicted_Negative_list)
                    elif covariance_estimator == 'Envmodel_conformal':
                        p_star_ood_Predicted_Negative_list_EnvmodelConformal.append(OOD_Predicted_Negative_list)
                    else:
                        pass

                        # (5.6) plot
            if Flag_OOD:
                print('Plotting...........')
                xlabel = 'p*'
                curve_label = noise_ATARI_NAME_list
                title = ATARI_NAME + ' \n OOD '
                # (5.6.1) positive: Empirical and MCD
                OOD_Predicted_Positive_Empirical = np.array(p_star_ood_Predicted_Positive_list_Empirical)
                OOD_Predicted_Positive_MCD = np.array(p_star_ood_Predicted_Positive_list_MCD)
                ylabel = 'Clean/Positive Accuracy'
                filename = ATARI_NAME + '_OOD_CleanData'
                # (5.6.2) Negative: Empirical and MCD
                OOD_Predicted_Negative_Empirical = np.array(p_star_ood_Predicted_Negative_list_Empirical)
                OOD_Predicted_Negative_MCD = np.array(p_star_ood_Predicted_Negative_list_MCD)
                ylabel = 'OOD/Negative Accuracy'
                filename = ATARI_NAME + '_OOD_Negative'
                # (5.6.3) Overall: Empirical and MCD
                OOD_Predicted_Empirical = (np.array(p_star_ood_Predicted_Positive_list_Empirical) + np.array(p_star_ood_Predicted_Negative_list_Empirical)) / 2
                OOD_Predicted_MCD = (np.array(p_star_ood_Predicted_Positive_list_MCD) + np.array(p_star_ood_Predicted_Negative_list_MCD)) / 2
                OOD_Predicted_Euclidean_I = (np.array(p_star_ood_Predicted_Positive_list_Euclidean_I) + np.array(p_star_ood_Predicted_Negative_list_Euclidean_I)) / 2
                OOD_Predicted_Euclidean_Diag = (np.array(p_star_ood_Predicted_Positive_list_Euclidean_Diag) + np.array(p_star_ood_Predicted_Negative_list_Euclidean_Diag)) / 2
                OOD_Predicted_TMD = (np.array(p_star_ood_Predicted_Positive_list_TMD) + np.array(p_star_ood_Predicted_Negative_list_TMD)) / 2
                OOD_Predicted_EmpiricalConformal = (np.array(p_star_ood_Predicted_Positive_list_EmpiricalConformal) + np.array(p_star_ood_Predicted_Negative_list_EmpiricalConformal)) / 2
                OOD_Predicted_MCDConformal = (np.array(p_star_ood_Predicted_Positive_list_MCDConformal) + np.array(p_star_ood_Predicted_Negative_list_MCDConformal)) / 2
                OOD_Predicted_EntropyConformal = (np.array(p_star_ood_Predicted_Positive_list_EntropyConformal) + np.array(p_star_ood_Predicted_Negative_list_EntropyConformal)) / 2
                OOD_Predicted_EnvmodelConformal = (np.array(p_star_ood_Predicted_Positive_list_EnvmodelConformal) + np.array(p_star_ood_Predicted_Negative_list_EnvmodelConformal)) / 2

                ylabel = 'Overall Detection Accuracy'
                filename = ATARI_NAME + '_OOD_Detection'
                if Flag_Write:
                    for i in range(len(OOD_Predicted_Empirical)):
                        newpd_em = pd.DataFrame({'Emp_All_'+curve_label[i]: OOD_Predicted_Empirical[i]})
                        newpd_MCD = pd.DataFrame({'MCD_All_'+curve_label[i]: OOD_Predicted_MCD[i]})
                        newpd_Euclidean_I = pd.DataFrame({'EucI_All_' + curve_label[i]: OOD_Predicted_Euclidean_I[i]})
                        newpd_Euclidean_Diag = pd.DataFrame({'EucDiag_All_' + curve_label[i]: OOD_Predicted_Euclidean_Diag[i]})
                        newpd_TMD = pd.DataFrame({'TMD_All_' + curve_label[i]: OOD_Predicted_TMD[i]})
                        newpd_emconformal = pd.DataFrame({'EmpConformal_All_' + curve_label[i]: OOD_Predicted_EmpiricalConformal[i]})
                        # newpd_MCDconformal = pd.DataFrame({'MCDConformal_All_' + curve_label[i]: OOD_Predicted_MCDConformal[i]})
                        newpd_EntropyConformal = pd.DataFrame({'EntConformal_All_' + curve_label[i]: OOD_Predicted_EntropyConformal[i]})
                        newpd_EnvmodelConformal = pd.DataFrame({'EnvmodelConformal_All_' + curve_label[i]: OOD_Predicted_EnvmodelConformal[i]})

                        Final_result = pd.concat([Final_result, newpd_em], axis=1)
                        Final_result = pd.concat([Final_result, newpd_MCD], axis=1)
                        Final_result = pd.concat([Final_result, newpd_Euclidean_I], axis=1)
                        Final_result = pd.concat([Final_result, newpd_Euclidean_Diag], axis=1)
                        Final_result = pd.concat([Final_result, newpd_TMD], axis=1)
                        Final_result = pd.concat([Final_result, newpd_emconformal], axis=1)
                        # Final_result = pd.concat([Final_result, newpd_MCDconformal], axis=1)
                        Final_result = pd.concat([Final_result, newpd_EntropyConformal], axis=1)
                        Final_result = pd.concat([Final_result, newpd_EnvmodelConformal], axis=1)

                else:
                    plt.figure()
                    save_plot(p_star_list, OOD_Predicted_Empirical, curve_label, xlabel, ylabel, title + 'Empirical', filename,121)
                    save_plot(p_star_list, OOD_Predicted_MCD, curve_label, xlabel, ylabel, title + 'MCD', filename, 122)
                    plt.close()
        # transpose
        Final_result = Final_result.transpose()
        print(Final_result)
        print('Contamination Ratio: ', contaminated_ratio)
        print('PCA: ', opt.feature)
        if opt.conformal == 0:
            Final_result.to_csv('log/Final_Ratio'+str(contaminated_ratio)+ATARI_NAME_list_focus[0]+'.csv')
        else:
            print('Results are combined with conformal inference!')
            if setting == 4:
                Final_result.to_csv('log/Final_Ratio' + str(contaminated_ratio) + ATARI_NAME_list_focus[0] + '_town_final.csv')
            else:
                Final_result.to_csv('log/Final_Ratio' + str(contaminated_ratio) + ATARI_NAME_list_focus[0] + '_lane_rainnight.csv')