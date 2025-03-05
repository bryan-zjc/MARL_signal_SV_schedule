import os
import torch
import random
import numpy as np
from src.network.networks import MLP
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class DQNAgent():
    def __init__(self, state_size, action_size, hidden_size, device, agent_select = 'DQN'):
        self.agent_select = agent_select
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = 1e-3 # 1e-3
        self.gamma = 0.95
        self.lr = 3e-5
        
        self.network = MLP(state_size=self.state_size,
                                action_size=self.action_size,
                                layer_size=hidden_size
                                ).to(self.device)
        self.target_net = MLP(state_size=self.state_size,
                                action_size=self.action_size,
                                layer_size=hidden_size
                                ).to(self.device)
        
        # centralised Q network
        if self.agent_select == 'CQL':
            self.cql_beta = 0.00001
        # online dqn
        elif self.agent_select == 'DQN':
            self.cql_beta = 0
        
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=self.lr)
        
    
    def get_action_online(self, state, epsilon):
        if random.random() > epsilon:
            # print('take action based on RL')
            # state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            state = torch.from_numpy(np.array(state)).float().to(self.device)
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()
            # print('action values: ', action_values)
            action = torch.argmax(action_values, dim=1).item()
        else:
            # print('take action randomly')
            action = random.choices(np.arange(self.action_size), k=1)[0]
        # print(action)
        return action


    def cql_loss(self, q_values, current_action):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)
    
        return (logsumexp - q_a).mean()


    def learn(self, experiences):
        
        states, actions, rewards, next_states, dones = experiences[0], experiences[1], experiences[2], experiences[3], experiences[4]
        actions = actions.to(torch.int64)
        
        # print('actions', actions)
        
        with torch.no_grad():
            Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_a_s = self.network(states)
        # print('Q_a_s', Q_a_s)
        Q_expected = Q_a_s.gather(1, actions)
        # print('Q_expected', Q_expected)
        
        cql1_loss = self.cql_loss(Q_a_s, actions)

        bellman_error = F.mse_loss(Q_expected, Q_targets)
        
        q1_loss = self.cql_beta * cql1_loss + 0.5 * bellman_error
        
        self.optimizer.zero_grad()
        q1_loss.backward()
        clip_grad_norm_(self.network.parameters(), 1.)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_net)
        return q1_loss.detach().item(), cql1_loss.detach().item(), bellman_error.detach().item()
        
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
            
    def lr_decay(self, total_steps, num_episodes):
        # Learning rate decay
        # ensure the learning rate is positive
        self.lr_current = max(1e-10,self.lr * (1 - total_steps / (num_episodes+1)))
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr_current
            
    
    def save(self, checkpoint_path, epsilon_number):
        # Save checkpoint
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.save(self.network.state_dict(), os.path.join(checkpoint_path, f'{self.agent_select}_net_episode_' + str(epsilon_number)))
        torch.save(self.target_net.state_dict(), os.path.join(checkpoint_path, f'{self.agent_select}_target_net_episode_' + str(epsilon_number)))


    def load(self, checkpoint_path, epsilon_number):
        # Load checkpoint
        self.network.load_state_dict(torch.load(os.path.join(checkpoint_path, f'{self.agent_select}_net_episode_' + str(epsilon_number))))
