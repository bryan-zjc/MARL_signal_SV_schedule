# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 09:38:20 2025

@author: poc
"""


import torch
import torch.nn as nn
from torch.distributions import Categorical 
import numpy as np
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)



class TransformerFeatureExtractor(nn.Module):
    def __init__(self, state_size, seq_len, hidden_size, num_layers=2, num_heads=4):
        """
        Transformer-based feature extractor for vehicle state sequences.
        Params:
        ======
            state_size (int): Dimension of each vehicle's state (e.g., 3 for position, velocity, and delay)
            hidden_size (int): Size of the hidden layer (also embedding size for transformer)
            num_layers (int): Number of Transformer encoder layers
            num_heads (int): Number of attention heads
        """
        super(TransformerFeatureExtractor, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # Input linear layer to project state to embedding size
        self.input_projection = nn.Linear(state_size, hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=num_heads, dim_feedforward=64, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output linear layer to project back to desired output size
        # self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, state):
        """
        Forward pass through the transformer feature extractor.
        Params:
        =======
            state (Tensor): Input tensor of shape (batch_size, seq_len, state_size)
                            where each seq_len represents a vehicle's state.
                            Example: (batch_size, N, 3) for N vehicles with 3 state features (position, velocity, delay)
        
        Returns:
        ========
            Tensor: Output tensor after transformer processing, shape (batch_size, seq_len, hidden_size)
        """
        mask = (state == 0).any(dim=-1)
        if len(mask.shape) == 1:
            mask[-1] = False  # Ensure the last position is always valid (not masked)
        else:
            mask[:, -1] = False  # buffer

        # Project state to embedding size
        state = self.input_projection(state)  # (batch_size, seq_len, hidden_size)

        # Positional encoding
        positional_encoding = nn.Parameter(torch.zeros(self.seq_len, self.hidden_size))
        positional_encoding = positional_encoding.to(self.device)
        state += positional_encoding  # Add positional encoding for the current sequence length

        # # Pass through Transformer encoder
        # state = self.transformer_encoder(state)
        # Pass through Transformer encoder with mask
        state = self.transformer_encoder(state, src_key_padding_mask=mask)  # (batch_size, seq_len, hidden_size)

        # Project to output size
        # state = self.output_projection(state)
        return state



class Actor(nn.Module):
    def __init__(self, RV_state_size, seq_len, action_size, hidden_size):
        super(Actor, self).__init__()
        # Replace the input layers with a Transformer feature extractor
        self.feature_extractor = TransformerFeatureExtractor(RV_state_size, seq_len, hidden_size)
        # Policy network
        state_size = seq_len
        self.fc_policy = nn.Linear(state_size, action_size)
        self.softmax = nn.Softmax(dim=-1)
        self.reset_parameters()
        self.seq_len = seq_len

    def forward(self, RV_state):
        # Assume input state is (batch_size, seq_len, state_size)
        features = self.feature_extractor(RV_state)  # Extract features
        if features.dim() == 2:
            features = features.mean(dim=1)
        elif features.dim() == 3:  # buffer
            features = features.mean(dim=-1)
        new_state = features
        # Compute action probabilities
        action_probs = self.softmax(self.fc_policy(new_state))
        return action_probs

    def reset_parameters(self):
        self.fc_policy.weight.data.uniform_(-3e-3, 3e-3)

    def evaluate(self, RV_state, epsilon=1e-6):
        action_probs = self.forward(RV_state)
        dist = Categorical(action_probs)
        action = dist.sample().to(RV_state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach(), action_probs, log_action_probabilities        

    def get_action(self, RV_state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(RV_state)
        dist = Categorical(action_probs)
        action = dist.sample().to(RV_state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach(), action_probs, log_action_probabilities

    def get_det_action(self, RV_state):
        action_probs = self.forward(RV_state)
        dist = Categorical(action_probs)
        action = dist.sample().to(RV_state.device)
        return action.detach()



class Critic(nn.Module):
    def __init__(self, RV_state_size, seq_len, action_size, hidden_size, seed=1):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Replace the input layers with a Transformer feature extractor
        self.feature_extractor = TransformerFeatureExtractor(RV_state_size, seq_len, hidden_size)
        state_size = seq_len
        # Value network
        self.fc_value = nn.Linear(state_size, action_size)
        self.reset_parameters()
        self.seq_len = seq_len

    def reset_parameters(self):
        self.fc_value.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, RV_state):
        # Assume input state is (batch_size, seq_len, state_size)
        features = self.feature_extractor(RV_state)  # Extract features
        if features.dim() == 2:
            features = features.mean(dim=1)
        elif features.dim() == 3:  # buffer
            features = features.mean(dim=-1)
        new_state = features
        # Compute value
        value = self.fc_value(new_state)
        return value

