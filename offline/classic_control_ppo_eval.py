import gym
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from agents import PPO_Agent

class PreprocessWrapper(gym.Wrapper):
    def __init__(self, env, r_preprocess=None, s_preprocess=None):
        '''
        reward & state preprocess
        record info like real reward, episode length, etc
        Be careful: when an episode is done: check info['episode'] for information
        '''
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.r_preprocess = r_preprocess
        self.s_preprocess = s_preprocess
        # self.s_preprocess = lambda x:x/255.
        self.rewards = []

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = state.astype('float32')
        self.rewards.append(reward)
        if done:
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen}
            assert isinstance(info,dict)
            if isinstance(info,dict):
                info['episode'] = epinfo
            self.rewards = []
        # preprocess reward
        if self.r_preprocess is not None:
            reward = self.r_preprocess(reward)
        # preprocess state
        if self.s_preprocess is not None:
            state = self.s_preprocess(state)
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        state = state.astype('float32')
        # preprocess state
        if self.s_preprocess is not None:
            state = self.s_preprocess(state)
        return state

    def render(self, mode='human'):
        return self.env.render(mode=mode)

class BatchEnvWrapper:
    def __init__(self, envs):
        self.envs = envs
        self.observation_space = envs[0].observation_space.shape[0]
        # self.observation_space = [84,84,1]
        self.action_space = envs[0].action_space.n
        self.epinfobuf = deque(maxlen=100)

    def step(self, actions):
        states = []
        rewards = []
        dones = []
        infos = []
        for i, env in enumerate(self.envs):
            state, reward, done, info = env.step(actions[i])
            if done:
                state = env.reset()
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            maybeepinfo = info.get('episode')

            if maybeepinfo:
                self.epinfobuf.append(maybeepinfo)

        # print(infos)
        return states, rewards, dones, infos

    def reset(self):
        return [self.envs[i].reset() for i in range(self.get_num_of_envs())]

    def render(self, mode='human'):
        return self.envs[0].render(mode=mode)

    def get_num_of_envs(self):
        return len(self.envs)

    def get_episode_rewmean(self):
        #print([epinfo['r'] for epinfo in self.epinfobuf])
        #input()
        return round(self.safemean([epinfo['r'] for epinfo in self.epinfobuf]),2)

    def get_list_of_episode(self):
        return [epinfo['r'] for epinfo in self.epinfobuf]

    def get_episode_lenmean(self):
        return round(self.safemean([epinfo['l'] for epinfo in self.epinfobuf]),2)

    def safemean(self,xs):
        return np.nan if len(xs) == 0 else np.mean(xs)

def Baselines_DummyVecEnv(env_id,num_env):
    envs = []
    for i in range(num_env):
        env = gym.make(env_id)
        env = PreprocessWrapper(env) # record reward, episode length, etc
        envs.append(env)
    batch_env = BatchEnvWrapper(envs) # vectorized environment
    return batch_env

class mlp(nn.Module):
    def __init__(self,action_space,state_space):
        super(mlp, self).__init__()
        # print('state_space',state_space)
        # print('action_space',action_space)

        self.mlp1 = nn.Linear(state_space,128)
        self.mlp2 = nn.Linear(128,128)
        self.logits = nn.Linear(128,action_space)
        self.value = nn.Linear(128,1)

    def forward(self,state):
        s = nn.functional.relu(self.mlp1(state))
        feature = self.mlp2(s)
        s = nn.functional.relu(feature)

        logits = self.logits(s)
        p = torch.nn.Softmax(dim=-1)(logits) + 1e-8
        policy_head = Categorical(probs=p)

        value = self.value(s)

        return policy_head,value,feature

if __name__ == '__main__':
    # batch_env = Baselines_DummyVecEnv('CartPole-v1',num_env=1)
    batch_env = Baselines_DummyVecEnv('MountainCar-v0',num_env=1)
    print('observation_space:',batch_env.observation_space)
    print('action_space:',batch_env.action_space)

    agent = PPO_Agent(batch_env.action_space, batch_env.observation_space, mlp)
    # agent.load_model('model/CartPole-v1_44_400000_1000')
    agent.load_model('model/MountainCar-v0_44_1000000_1000')

    states = batch_env.reset()
    rewards, dones, info = None, None, None
    for i in range(10000):
        policy_head, _, _ = agent.net(torch.from_numpy(np.array(states)).to(agent.device).float())
        actions = policy_head.sample()
        actions = actions.detach().cpu().numpy()

        states, rewards, dones, info = batch_env.step(actions)
        batch_env.render()
        if i % 1000 == 0:
            print('rewards:',batch_env.get_episode_rewmean(),'episode_len:',batch_env.get_episode_lenmean())