import numpy as np
from sklearn.covariance import EmpiricalCovariance,LedoitWolf,MinCovDet
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
import torch
from sklearn.manifold import TSNE
import pandas as pd
import argparse
from autoencoder import EnsembleModel
import torch.nn.functional as F

# CUDA_VISIBLE_DEVICES=0 python detect_final.py --game breakout --pca 1 --feature 50 --conformal 1
# CUDA_VISIBLE_DEVICES=1 python detect_final.py --game asterix --pca 1 --feature 50 --conformal 1
# CUDA_VISIBLE_DEVICES=2 python detect_final.py --game spaceinvader --pca 1 --feature 50 --conformal 1
# CUDA_VISIBLE_DEVICES=0 python detect_final.py --game fishingderby --pca 1 --feature 50 --conformal 1
# CUDA_VISIBLE_DEVICES=2 python detect_final.py --game enduro --pca 1 --feature 50 --conformal 1
# CUDA_VISIBLE_DEVICES=3 python detect_final.py --game tutankham --pca 1 --feature 50 --conformal 1

parser = argparse.ArgumentParser(description='offline detection')
parser.add_argument('--game', type=str, default="mountaincar", choices=['cartpole', 'mountaincar','breakout', 'asterix',  'spaceinvader','fishingderby', 'enduro',  'tutankham']) # action space: 4, 9, 6
parser.add_argument('--pca', type=int, default=1, help="PCA or not")
parser.add_argument('--feature', type=int, default=50, help="PCA feature length")
parser.add_argument('--conformal', type=int, default=1, help="whether we use the conformal to determine the thresholding")
opt = parser.parse_args()
print(opt)

"""
step 1: use the half clean data to do mean and variance estimation for MD, and calibrate quantile threshold for MD+C
step 2: use the clean data to generate noisy data
step 3: evaluate the detection on both the clean and noisy data
"""

Experiments = {'game1':['BreakoutNoFrameskip-v4','AsterixNoFrameskip-v4','SpaceInvadersNoFrameskip-v4'],
               'game2': ['EnduroNoFrameskip-v4','FishingDerbyNoFrameskip-v4','TutankhamNoFrameskip-v4'],
               'game3': ['CartPole-v1', 'MountainCar-v0'],
               'contaminated_ratio':[0.0],
               'feature':[opt.feature]}
if opt.conformal == 1:
    Experiments['covariance'] = ['Empirical', 'MinCovDet', 'Euclidean_I', 'Euclidean_Diag', 'TMD', 'MD_conformal', 'Entropy_conformal', 'Envmodel_conformal']
else:
    Experiments['covariance'] = ['Empirical', 'MinCovDet', 'Euclidean_I', 'Euclidean_Diag', 'TMD']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if opt.game in ['breakout', 'asterix', 'spaceinvader', 'BAS']:
    ATARI_NAME_list = Experiments['game1']
elif opt.game in ['fishingderby', 'enduro',  'tutankham', 'EFT']:
    ATARI_NAME_list = Experiments['game2']
elif opt.game in ['cartpole', 'mountaincar']:
    ATARI_NAME_list = Experiments['game3']

    # modify the env wrappers
    from classic_control_ppo_eval import *

else:
    ATARI_NAME_list = None
    AssertionError('The game name is wrong!')

if opt.game == 'breakout':
    ATARI_NAME_list_focus = ['BreakoutNoFrameskip-v4']
elif opt.game == 'asterix':
    ATARI_NAME_list_focus = ['AsterixNoFrameskip-v4'] # loop other env in ATARI_NAME_list later
elif opt.game == 'spaceinvader':
    ATARI_NAME_list_focus = ['SpaceInvadersNoFrameskip-v4'] # loop other env in ATARI_NAME_list later
elif opt.game == 'enduro':
    ATARI_NAME_list_focus = ['EnduroNoFrameskip-v4'] # loop other env in ATARI_NAME_list later
elif opt.game == 'fishingderby':
    ATARI_NAME_list_focus = ['FishingDerbyNoFrameskip-v4'] # loop other env in ATARI_NAME_list later
elif opt.game == 'tutankham':
    ATARI_NAME_list_focus = ['TutankhamNoFrameskip-v4'] # loop other env in ATARI_NAME_list later
elif opt.game == 'cartpole':
    ATARI_NAME_list_focus = ['CartPole-v1']
elif opt.game == 'mountaincar':
    ATARI_NAME_list_focus = ['MountainCar-v0']
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
Flag_Write = True # False, True
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
        SAMPLES = np.array(Feature_dict[0]) # to compute all

    # fit the mean and covariance for each label
    for i in range(n_labels):
        samples = np.array(Feature_dict[i])
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
            print('covariance_estimator is wrong ...............')
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

def ComputeMD(Feature_dict, n_labels, model_list):

    dis_list_all = np.array([])
    for i in range(n_labels): # for each feature to compute MD from the center of each action distribution (model_list)
        dis_list = []
        for j in range(n_labels):
            value = np.array(Feature_dict[i]) - model_list[j].location_ # broadcast
            x = np.matmul(np.matmul(value, model_list[j].precision_), value.T)
            diag = np.diag(x)
            dis_list.append(diag)
        #### bug: dist_list shoudl be cleared
        dis = np.min(dis_list, axis=0) # each feature i to the closest class [5120] * actions -> [5120]
        dis_list_all = np.concatenate([dis_list_all, dis])
    return dis_list_all




def ComputeEnvmodel(X_state, Y_nextstate, X_action, autoencoder):
    batch_size = int((40)/args.batch_env.action_space)
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
    if opt.game in ['cartpole', 'mountaincar']:
        batch_size = 20
    else:
        batch_size = int((40)/args.batch_env.action_space)

    mse_aggre = np.zeros((autoencoder.model_num, 1)) # initialize one column
    for i in range(int(X_noisystate.shape[0]/batch_size)):
        X = X_noisystate[i * batch_size:(i + 1) * batch_size]
        act = X_action[i * batch_size:(i + 1) * batch_size]
        Y = Y_nextstate[i * batch_size:(i + 1) * batch_size]
        mse = autoencoder.compute_mse(X, act, Y)  # [model_num, batch_size]
        mse_aggre = np.concatenate([mse_aggre, mse], axis=1)
    mse_aggre = mse_aggre[:, 1:]  # [model_num, data_num*n_labels]
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
        dis = -torch.sum(p * torch.log(p), dim=1)
        dis_list_all.append(dis)
        try:
            scores[i] = np.quantile(dis, 0.95)  # may be empty for a certain set
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
    return Predicted_Negative_list


# evaluate the distances and then compute the conformal score
def ComputeComformal(Feature_dict, n_labels, model_list, feature_length, calibration=True):
    # [calibration] Feature_dict: [num_actions] -> [data_nums, dims]
    # step 1: compute MD and collect distances with actions
    dis_list_all = np.array([])
    dis_list_all_action = np.array([])
    for i in range(n_labels):  # for each feature to compute MD from the center of each action distribution (model_list)
        dis_list = []
        for j in range(n_labels):
            # calculate the Mahalanobis distance
            value = np.array(Feature_dict[i]) - model_list[j].location_  # broadcast
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
                scores[i] = np.quantile(dis_temp, 0.95) # may be empty for a certain set
            except:
                scores[i] = stats.chi2.ppf([0.95], df=feature_length) # for the special case: just use the chi2 under the assumption of normality
    else:
        scores = None
    return dis_list_all, dis_list_all_action, scores

# compare distances with the threshold (scores) and evaluate the accuracy
def ComputeAccuracy_conformal(Dist, Dist_action, scores, Clean=False):
    Predicted_Negative_list = []
    fasle_positive, true_negative = 0.0, 0.0
    for i in range(n_labels):
        threshold = scores[i]
        dis_temp = Dist[Dist_action == i]
        fasle_positive += sum(dis_temp <= threshold)
        true_negative += sum(dis_temp > threshold)
    # evaluate the accuracy: for clean, we wish all within the threshold, while for noisy, we wish all out of the threshold.
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
    if opt.game in ['cartpole', 'mountaincar']:
        random_noise = np.random.uniform(-adv_epsilon, adv_epsilon, size=(args.batch_env.observation_space))
    else:
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

        All_loop = len(Experiments['feature']) * len(Experiments['contaminated_ratio']) * len(ATARI_NAME_list_focus)
        # for ATARI_NAME in ATARI_NAME_list:
        for ATARI_NAME in ATARI_NAME_list_focus: # focus on breakout ###########################
            loop += 1
            print('#########################################################################################')
            print('Loop {}/{}, Game: {}, PCA:{}, Contaminated Ratio: {} '.format(loop, All_loop, ATARI_NAME, feature_length, contaminated_ratio), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            noise_ATARI_NAME_list = copy.copy(ATARI_NAME_list)
            noise_ATARI_NAME_list.remove(ATARI_NAME) # other environment as OOD

            train = False
            render = False

            ######################## (1) load model, initialization
            print('Step 1: Load Model',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            if opt.game in ['cartpole', 'mountaincar']:
                args.NUMBER_ENV = 1
                args.batch_env = Baselines_DummyVecEnv(env_id=ATARI_NAME,num_env=args.NUMBER_ENV) # batch size: args.NUMBER_ENV
                agent = PPO_Agent(args.batch_env.action_space, args.batch_env.observation_space, mlp)
                Length = 1000000 if ATARI_NAME == 'MountainCar-v0' else 400000

                # agent.load_model('model/CartPole-v1_44_400000_1000')
                if torch.cuda.is_available():
                    agent.load_model(f'model/{ATARI_NAME}_44_{Length}_1000')
                else:
                    agent.load_model(f'model/{ATARI_NAME}_44_{Length}_1000', GPU=False)
            else:
                args.batch_env = Baselines_DummyVecEnv(env_id=ATARI_NAME,num_env=args.NUMBER_ENV) # batch size: args.NUMBER_ENV
                agent = PPO_Agent(args.batch_env.action_space,args.batch_env.observation_space,nature_cnn)


                if torch.cuda.is_available():
                    agent.load_model('model/'+ATARI_NAME+'_0.00025_1000.pth')
                else:
                    agent.load_model('model/'+ATARI_NAME+'_0.00025_1000.pth', GPU=False)

            n_labels = args.batch_env.action_space # breakout: 4
            feature_dict = {}
            state_dict = {}
            nextstate_dict = {} # for envmodel detection
            # data_num = 512*int((40)/args.batch_env.action_space) # ??????????????????????? the number of data, 512 * 10
            if torch.cuda.is_available():
                data_num = 512*int((40)/args.batch_env.action_space)
            else:
                if opt.game in ['cartpole', 'mountaincar']:
                    data_num = 5120
                else:
                    data_num = 512 # for debug
            # else:
            #     data_num = 512*int((20)/args.batch_env.action_space) # for my computer
            # print('data_num :',data_num)
            for i in range(n_labels):
                feature_dict[i] = deque(maxlen=data_num)
                state_dict[i] = deque(maxlen=data_num)
                nextstate_dict[i] = deque(maxlen=data_num)

            def current_num():
                num = [0]*n_labels
                for i in range(n_labels):
                    num[i] = len(feature_dict[i])
                return num

            ######################## (2) interact clean data in feature_dict, state_dict
            print('Step 2: Collect Clean Data ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            states = args.batch_env.reset()
            rewards, dones, info = None,None,None
            current_step = 0
            tstart = time.time()
            while min(current_num())<data_num: # for each action class, we need to accumulate enough data, i.e., 512*10
                actions,feature_vectors, _ = agent.act(states,rewards, dones, info, train=train,current_step=current_step)
                # IMPORTANT: collect state and feature in each class over each
                next_states, rewards, dones, info = args.batch_env.step(actions)

                for i in range(args.NUMBER_ENV):
                    feature_dict[actions[i]].append(feature_vectors[i]) # feature [j] <- current features [i]
                    state_dict[actions[i]].append(states[i])
                    nextstate_dict[actions[i]].append(next_states[i])

                if render:
                    args.batch_env.render()
                states = next_states
                current_step += args.batch_env.get_num_of_envs()
                if current_step % 10000 == 0:
                    tnow = time.time()
                    fps = current_step / (tnow - tstart)
                    print('Game: {}, run: {}, current_step: {}, time: {:.2f}, fps: {:.2f}, mean reward: {}, mean length: {},current num: {}'.format(
                        ATARI_NAME,
                        [args.run,args.SEED_LIST],
                        current_step,
                        tnow - tstart,
                        fps,
                        args.batch_env.get_episode_rewmean(),
                        args.batch_env.get_episode_lenmean(),
                        current_num()))

            temp_p_list = [0.05]
            p_star_list = np.array(temp_p_list) # basic setting
            ##### PCA on all the data
            if use_PCA:
                pca, clean_feature_dict = ApplyPCA(feature_dict, feature_length, data_num, n_labels)
            else:
                pca = None
                clean_feature_dict = deepcopy(feature_dict)

            ##### for model-based detection: evaluate the scores tau by using the pretrained autoencoder model
            if 'Envmodel_conformal' in Experiments['covariance']:
                if opt.game in ['cartpole', 'mountaincar']:
                    from autoencoder_classical import EnsembleModel
                    model = EnsembleModel(args.batch_env.action_space, args.batch_env.observation_space, model_num=5)
                else:
                    model = EnsembleModel(action_dim=n_labels, model_num=5)
                if torch.cuda.is_available():
                    model.load_state_dict(torch.load('model/autoencoder_{}.pth'.format(opt.game)))
                else:
                    model.load_state_dict(torch.load('model/autoencoder_{}.pth'.format(opt.game), map_location=torch.device('cpu')))

                X_action = torch.arange(n_labels).repeat_interleave(data_num)
                X_action = F.one_hot(X_action, num_classes=n_labels).float().to(model.device)
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
                if ATARI_NAME == 'AsterixNoFrameskip-v4':
                    noisy_std_list = [0.1, 0.15, 0.2] # for game break and spaceinvader, 0.1 is overly large
                elif ATARI_NAME == 'TutankhamNoFrameskip-v4':
                    noisy_std_list = [0.02, 0.04, 0.06]
                elif ATARI_NAME in ['FishingDerbyNoFrameskip-v4', 'EnduroNoFrameskip-v4']:
                    noisy_std_list = [0.1, 0.2, 0.3]
                elif ATARI_NAME == 'CartPole-v1':
                    noisy_std_list = [0.1, 0.2, 0.3, 0.4, 0.5]
                elif ATARI_NAME == 'MountainCar-v0':
                    noisy_std_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                else:
                    # noisy_std_list = [0.01, 0.03, 0.05] #0.1 is overly large
                    noisy_std_list = [0.02, 0.04, 0.06] #0.1 is overly large
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
                        if opt.game in ['cartpole', 'mountaincar']:
                            noise = np.random.normal(loc=0, scale=noisy_std, size=(args.batch_env.observation_space))
                        else:
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
                        noisy_state_dict[i].append(noise_state_evaluation.squeeze()) # for envmodel detection
                        # noisy_feature_dict: clean vs noisy_feature_dict_evaluation: noisy
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

                ####### loop for each detection method in Experiments['covariance']
                for covariance_estimator in Experiments['covariance']:
                    # (3.3) detector construction: estimate the mean and variance via two methods in clean data (contamination ratio = 0.0)!!!!!!!!!!!
                    model_list = MeanVarEstimate(noisy_feature_dict, n_labels, covariance_estimator)

                    ############ Evaluation on balanced data
                    # (3.4) 50% clean data to compute the MD again
                    if covariance_estimator == 'Envmodel_conformal':
                        Predicted_Positive_list = ComputeEnvmodel_conformal(score, X_state, Y_nextstate, X_action, model, Clean=True)
                    elif covariance_estimator == 'Entropy_conformal':
                        Dist, scores = ComputeEntropy(probs_dict, n_labels, feature_length)
                        Predicted_Positive_list = ComputeEntropy_conformal(Dist, scores, Clean=True)
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

                    # (3.5) 50% new noisy features to compute the MD again based on noisy_feature_dict_evaluation（use existing scores!!!!）
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
            # Plot_Random = True
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
                        newpd_EntropyConformal = pd.DataFrame({'EntConformal_All_'+curve_label[i]: Predicted_EntropyConformal[i]})
                        newpd_EnvmodelConformal = pd.DataFrame({'EnvmodelConformal_All_'+curve_label[i]: Predicted_EnvmodelConformal[i]})
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

            if ATARI_NAME == 'CartPole-v1':
                adv_epsilon_list = [0.06, 0.08, 0.1, 0.15, 0.2]
            else:
                # adv_epsilon_list = [0.001, 0.01, 0.05, 0.1] # 0.2 is overly large
                adv_epsilon_list = [0.001, 0.01, 0.05] # 0.2 is overly large
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
                        adv_feature_dict_evaluation[i].append(pca.transform(f_).squeeze())
                        adv_state_dict[i].append(adv_state_evaluation.squeeze())  # for envmodel detection

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
            # Plot_ADV = False
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

                noise_batch_env = Baselines_DummyVecEnv(env_id=noise_ATARI_NAME, num_env=args.NUMBER_ENV)
                states = noise_batch_env.reset()

                # (5.1) construct all OOD data
                ood_feature_queue = deque(maxlen=data_num*n_labels)
                ood_probs_queue = deque(maxlen=data_num*n_labels)
                ood_state_queue = deque(maxlen=data_num*n_labels)


                # initialize via random action
                for i in range(300):
                    random_actions = np.random.randint(low = 0,high = noise_batch_env.action_space,size = args.NUMBER_ENV)
                    states, rewards, dones, info = noise_batch_env.step(random_actions)
                # execute random action and store feature vectors (the data size is still data_num*n_labels)
                while len(ood_feature_queue) < data_num*n_labels:

                    random_actions = np.random.randint(low = 0,high = noise_batch_env.action_space,size = args.NUMBER_ENV)
                    states, rewards, dones, info = noise_batch_env.step(random_actions)
                    # if opt.game in ['cartpole', 'mountaincar']:
                    #     _, feature_vectors, probs = agent_data.act(states, None, None, None, train=train, current_step=0)
                    # else:
                    ####### align the state dimension for the cartpole and mountaincar
                    if opt.game == 'cartpole': # state: 4
                        states = [np.concatenate([states[0], np.random.normal(0, 1, 2)])]
                    elif opt.game == 'mountaincar': # state 2
                        states = [states[0][:2]]
                    else:
                        pass
                    _, feature_vectors, probs = agent.act(states, None, None, None, train=train, current_step=0)
                    ood_feature_queue.extend(pca.transform(feature_vectors))
                    ood_probs_queue.extend(probs)
                    ood_state_queue.extend(states)
                # record the next state
                random_actions = np.random.randint(low=0, high=noise_batch_env.action_space, size=args.NUMBER_ENV)
                nextstates, rewards, dones, info = noise_batch_env.step(random_actions)


                ood_feature_queue = np.array(ood_feature_queue)
                ood_probs_queue = np.array(ood_probs_queue)
                ood_state_queue = np.array(ood_state_queue) # [n_labels*data_num, 84, 84, 4]
                OOD_feature_dict = {}
                OOD_feature_dict_evaluation = {}
                OOD_state_dict = {}
                OOD_nextstate_dict = {}
                OOD_probs_dict = {}
                for i in range(n_labels):
                    OOD_feature_dict[i] = deque(maxlen=data_num)
                    OOD_feature_dict_evaluation[i] = deque(maxlen=data_num)
                    OOD_state_dict[i] = deque(maxlen=data_num)
                    OOD_nextstate_dict[i] = deque(maxlen=data_num)
                    OOD_probs_dict[i] = deque(maxlen=data_num)
                    # save the current state
                    temp_state = ood_state_queue[i * data_num:(i + 1) * data_num]
                    # save the next state
                    if i < n_labels - 1:
                        temp_nextstate = ood_state_queue[i * data_num+1:(i + 1) * data_num+1] #
                    else:
                        if opt.game == 'cartpole':
                            nextstates = [np.concatenate([nextstates[0], np.random.normal(0, 1, 2)])]
                        if opt.game == 'mountaincar':
                            nextstates = [nextstates[0][:2]]
                        temp_nextstate = np.concatenate([ood_state_queue[i * data_num+1:], nextstates], axis=0)
                    # save the probs
                    temp_prob = ood_probs_queue[i * data_num:(i + 1) * data_num]
                    # all OOD
                    temp_ = ood_feature_queue[i*data_num:(i+1)*data_num]
                    # Contaminated OOD
                    index = np.random.choice(data_num, int(data_num*contaminated_ratio), replace=False)
                    temp = np.array(clean_feature_dict[i])
                    temp[index] = ood_feature_queue[i*data_num + index]
                    for j in range(data_num):
                        OOD_probs_dict[i].append(temp_prob[j, :].squeeze())
                        OOD_feature_dict_evaluation[i].append(temp_[j,:].squeeze())
                        OOD_feature_dict[i].append(temp[j,:].squeeze()) # [1,5120,512]->[5120,512]
                        OOD_state_dict[i].append(temp_state[j].squeeze())  # for envmodel detection
                        OOD_nextstate_dict[i].append(temp_nextstate[j].squeeze())  # for envmodel detection

                if (not Flag_ADV) and (not Flag_Random):
                    probs_dict = {}
                    for i in range(n_labels):
                        probs_dict[i] = deque(maxlen=data_num)
                        for j in range(data_num):
                            a, f, probs = agent.act([state_dict[i][j]], None, None, None, train=train, current_step=0)  # to query the feature vectors
                            probs_dict[i].append(probs.squeeze())

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
                        l_oodstate = [torch.tensor(list(OOD_state_dict[i])).float() for i in range(n_labels)]
                        X_oodstate = torch.cat(l_oodstate, dim=0).to(device)  # [data_num*n_labels, 84, 84, 4]

                        ##### important: compare with the next OOD state, instead of the clean next state: comparison (1) f(s_t, a_t) vs s_t+1 (2) f(OOD_t, a_t) vs OOD_t+1;
                        ##### underlying reason: f() is more like the current env, which is similar to s_t+1 instead of O_t+1; thus, mse distributions can be different
                        ##### Remark: the comparison manner of OOD is different from adv and random noise
                        l_oodstate_next = [torch.tensor(list(OOD_nextstate_dict[i])).float() for i in range(n_labels)]
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
            # Plot_OOD = False
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
        if opt.game in ['cartpole', 'mountaincar']:
            Final_result.to_csv('log/classical/Final_Ratio' + str(contaminated_ratio) + ATARI_NAME_list_focus[0] + '.csv')
        else:
            if opt.conformal == 0:
                Final_result.to_csv('log/Final_Ratio'+str(contaminated_ratio)+ATARI_NAME_list_focus[0]+'.csv')
            else:
                print('Results are combined with conformal inference!')
                Final_result.to_csv('log/Final_Ratio' + str(contaminated_ratio) + ATARI_NAME_list_focus[0] + '_conformal_envmodel.csv')