# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:44:00 2024

@author: Jichen Zhu
"""

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from src.network.networks_sac_transformer import Critic, Actor
import numpy as np
import math
import copy
import random


class SACTFAgent(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        i,
                        RV_state_size,
                        PL_state_size,
                        seq_len,
                        action_size,
                        hidden_size,
                        device,
                        agent_select
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(SACTFAgent, self).__init__()
        self.agent_select = agent_select
        self.RV_state_size = RV_state_size
        self.PL_state_size = PL_state_size
        self.action_size = action_size
        self.seq_len = seq_len
        self.inter_id = i

        self.device = device
        
        self.gamma = 0.99
        self.tau = 1e-2
        self.lr = 3e-4
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.lr) 
        
        # Actor Network 
        self.actor_local = Actor(RV_state_size, PL_state_size, seq_len, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr)     
        
        # Critic Network (w/ Target Network)
        self.critic1 = Critic(RV_state_size, PL_state_size, seq_len, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(RV_state_size, PL_state_size, seq_len, action_size, hidden_size, 1).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(RV_state_size, PL_state_size, seq_len, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(RV_state_size, PL_state_size, seq_len, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr) 
        self.softmax = nn.Softmax(dim=-1)
        

    
    def get_action_online(self, RV_state, PL_state):
        """Returns actions for given state as per current policy."""
        # state = torch.from_numpy(state).float().to(self.device)
        RV_state = torch.from_numpy(np.array(RV_state)).float().to(self.device)
        PL_state = torch.from_numpy(np.array(PL_state)).float().to(self.device)
        with torch.no_grad():
            action = self.actor_local.get_det_action(RV_state, PL_state)
        return int(action)

    def calc_policy_loss(self, RV_states, PL_states, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(RV_states, PL_states)

        q1 = self.critic1(RV_states, PL_states)   
        q2 = self.critic2(RV_states, PL_states)
        min_Q = torch.min(q1,q2)
        actor_loss = (action_probs * (alpha.to(self.device) * log_pis - min_Q )).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi
    
    def learn(self, experiences):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        RV_states, PL_states, actions, rewards, next_RV_states, next_PL_states, dones = experiences

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(RV_states, PL_states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_RV_states, next_PL_states)
            Q_target1_next = self.critic1_target(next_RV_states, next_PL_states)
            Q_target2_next = self.critic2_target(next_RV_states, next_PL_states)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1)) 


        # Compute critic loss
        q1 = self.critic1(RV_states, PL_states)
        q2 = self.critic2(RV_states, PL_states)
        
        q1_ = q1.gather(1, actions.long())
        q2_ = q2.gather(1, actions.long())
        
        critic1_loss = F.mse_loss(q1_, Q_targets)
        critic2_loss = F.mse_loss(q2_, Q_targets)
        
        
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item() #, cql1_scaled_loss.item(), cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item()

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


    def lr_decay(self, total_steps, num_episodes):
        # Learning rate decay
        lr_current = self.lr * (1 - total_steps / num_episodes)
        # self.alpha_optimizer
        # self.actor_optimizer
        # self.critic1_optimizer
        # self.critic2_optimizer
        for p in self.alpha_optimizer.param_groups:
            p['lr'] = lr_current
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_current
        for p in self.critic1_optimizer.param_groups:
            p['lr'] = lr_current
        for p in self.critic2_optimizer.param_groups:
            p['lr'] = lr_current
            
    def save(self, checkpoint_path, epsilon_number):
        # Save checkpoint
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.save(self.actor_local.state_dict(), os.path.join(checkpoint_path, f'{self.agent_select}_net_episode_' + str(epsilon_number)))
        # torch.save(self.target_net.state_dict(), os.path.join(checkpoint_path, f'{self.agent_select}_target_net_episode_' + str(epsilon_number)))


    def load(self, checkpoint_path, epsilon_number):
        # Load checkpoint
        self.actor_local.load_state_dict(torch.load(os.path.join(checkpoint_path, f'{self.agent_select}_net_episode_' + str(epsilon_number))))

            