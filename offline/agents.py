import numpy as np
import torch
from rl_algorithms import PPO_clip
from schedules import LinearSchedule
import arguments as args
from buffers import BatchBuffer
import matplotlib.pyplot as plt
import arguments as args

class PPO_Agent():
    def __init__(self,action_space, state_space,net):
        self.action_space = action_space
        self.state_space = state_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"

        self.net = net(self.action_space,self.state_space).to(self.device)
        self.decay = LinearSchedule(schedule_timesteps=args.FINAL_STEP, final_p=0.)
        self.update = PPO_clip(self.net, self.decay,self.device)
        self.batch_buffer = BatchBuffer(buffer_num=args.NUMBER_ENV,gamma=0.99,lam=0.95)

    def save_model(self,path):
        torch.save(self.net.state_dict(),path)

    def load_model(self,path, GPU=True):
        if GPU:
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def act(self,states,rewards,dones,info,train,current_step):
        # print(type(states))
        # if isinstance(states, torch.Tensor):
        #     pass
        # else:
        states = torch.from_numpy(np.array(states)).to(self.device).float() # [8, 84, 84, 4]
        policy_head, values,feature_vectors = self.net(states) # IMPORTANT: output 3 items from the feature vectors --- policy_head.probs: [8,4]
        # print('policy head: ',policy_head)
        # print('values: ',values)
        actions = policy_head.sample()
        log_probs = policy_head.log_prob(actions)

        if train:
            self.train(states.detach().cpu().numpy(),
                       actions.detach().cpu().numpy(),
                       rewards,
                       dones,
                       values.detach().cpu().numpy(),
                       log_probs.detach().cpu().numpy(),
                       current_step)

        # return actions.detach().cpu().numpy(),feature_vectors.detach().cpu().numpy()
        return actions.detach().cpu().numpy(),feature_vectors.detach().cpu().numpy(), policy_head.probs.detach().cpu().numpy() # return the prob to evaluate the entropy !!

    def train(self,states,actions,rewards,dones,values,log_probs,current_step):
        values = np.reshape(values, [-1])
        # (1) firstly explore some data (buffer_num in total) add data
        if rewards is None and dones is None:
            for i in range(self.batch_buffer.buffer_num):
                self.batch_buffer.buffer_list[i].add_data(
                    state_t=states[i],
                    action_t=actions[i],
                    value_t=values[i],
                    log_prob_t=log_probs[i])
        else:
            for i in range(self.batch_buffer.buffer_num):
                self.batch_buffer.buffer_list[i].add_data(
                    state_t=states[i],
                    action_t=actions[i],
                    reward_t=rewards[i],
                    terminal_t=dones[i],
                    value_t=values[i],
                    log_prob_t=log_probs[i])

        # (2) sample data from the buffer and then train the agent
        if current_step > 0 and current_step / self.batch_buffer.buffer_num % self.update.time_horizon == 0:
            #print(np.shape(self.batch_buffer.buffer_list))
            args.tempM.append(args.batch_env.get_episode_rewmean())
            args.tempMeanLength.append(args.batch_env.get_episode_lenmean())

            # (2.1) get all the explored data
            s, a, ret, v, logp, adv = self.batch_buffer.get_data() # all the data?

            print('data shape:', s.shape, a.shape, ret.shape, v.shape,logp.shape, adv.shape)
            # print('data type:', s.dtype, a.dtype, ret.dtype, v.dtype,logp.dtype, adv.dtype)
            for epoch in range(self.update.training_epoch):
                # (2.2) shuffle the data
                s, a, ret, v, logp, adv = self.batch_buffer.shuffle_data(s, a, ret, v, logp, adv)

                sample_size = self.update.batch_size // self.batch_buffer.buffer_num
                for i in range(self.update.time_horizon // sample_size):
                    # (2.3) batch sample from the shuffled data
                    batch_s, batch_a, batch_ret, batch_v, batch_logp, batch_adv = self.batch_buffer.get_minibatch(i*sample_size,
                                                                                                                  sample_size,
                                                                                                                  [],
                                                                                                                  self.state_space,
                                                                                                                  s, a,
                                                                                                                  ret,
                                                                                                                  v,
                                                                                                                  logp,
                                                                                                                  adv)

                    # print('minibatch data shape:', s.shape, a.shape, ret.shape, v.shape,logp.shape, adv.shape)
                    # print('minibatch data shape:', batch_s.shape, batch_a.shape, batch_ret.shape, batch_v.shape,batch_logp.shape, batch_adv.shape)

                    # (2.4) IMPORTANT: train the agent
                    self.update.learn(current_step, batch_s, batch_a, batch_ret, batch_v, batch_logp, batch_adv)

            #reward
            self.batch_buffer.initialize_buffer_list()

            states = torch.from_numpy(states).to(self.device)
            actions = torch.from_numpy(actions).to(self.device)
            policy_head, values, feature_vectors = self.net(states)
            log_probs = policy_head.log_prob(actions)
            log_probs = log_probs.detach().cpu().numpy()
            states = states.detach().cpu().numpy()
            actions = actions.detach().cpu().numpy()
            values = values.detach().cpu().numpy()
            values = np.reshape(values, [-1])

            for i in range(self.batch_buffer.buffer_num):
                self.batch_buffer.buffer_list[i].add_data(
                    state_t=states[i],
                    action_t=actions[i],
                    value_t=values[i],
                    log_prob_t=log_probs[i]) # here may be the problem
