# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:39:20 2024

@author: Jichen Zhu
"""

import os
import time
import wandb
import torch
import numpy as np
import pandas as pd
import src.utils.utils as utils
from src.agent.sac_transformer_agent import SACTFAgent #proposed
from src.road_env.env_network_transformer import Intersec_Env
from src.network.replay_buffer_transformer import ReplayBuffer
import json
import gc

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


# Set the maximum size for memory allocations to 1024 MiB, which helps avoid fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
# Release all unused cached memory that PyTorch has cached for future use.
torch.cuda.empty_cache()
# Force garbage collection to clean up any unreferenced objects in memory.
gc.collect()


def online_train(agent_list, env, buffer_list, inters_list, RV_schedule, num_episodes, save_path = None):
    episode_rewards = []
    
    # eps_list = {}
    traj_current = {} #states, action, rewards, next_states, dones
    traj_last = {}
    for tls_id in inters_list:
        # eps_list[tls_id] = 1
        traj_current[tls_id] = []
        traj_last[tls_id] = []
    
    total_steps = 0
    
    for episode in range(num_episodes + 1):
        print('-------------  current episode: '+str(episode)+'  --------------')
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        # print('current eps: ', eps_list)
        # Reset the environment
        RV_states, PL_states, net_info = env.reset(RV_schedule, save_path, episode)
        done = False
        episode_reward = {}
        decision_flag = {}
        action={}
        for tls_id in net_info.keys():
            decision_flag[tls_id] = 0
            action[tls_id] = 0
            episode_reward[tls_id] = 0
        start = time.time()

        while not done:
            # print('----------------------------------------------------------------')
           
            num_act_agent = 0
            act_agent_ids = []
            for tls_id in net_info.keys():
                if decision_flag[tls_id] == 1:
                    num_act_agent+=1
                    act_agent_ids.append(tls_id)

            if num_act_agent > 0:
                for tls_id in act_agent_ids:
                    # print(f'signal {tls_id} take a decision')
                    agent = agent_list[tls_id]

                    action[tls_id] = agent.get_action_online(RV_states[tls_id], PL_states[tls_id])
                    
                    # the trjectory is already cleared
                    traj_current[tls_id].append(RV_states[tls_id])
                    traj_current[tls_id].append(PL_states[tls_id])
                    traj_current[tls_id].append([action[tls_id]])
            
            next_RV_states, next_PL_states, rewards, dones, next_decision_flag, travel_t = env.step(action, RV_schedule)
            
            if num_act_agent > 0:
                for tls_id in act_agent_ids:
                    if len(traj_last[tls_id]) != 0:
                        episode_reward[tls_id] += rewards[tls_id][0]
                        # rewards of last action
                        traj_last[tls_id].append(rewards[tls_id])
                        #current state is the next state of last action
                        traj_last[tls_id].append(RV_states[tls_id])
                        traj_last[tls_id].append(PL_states[tls_id])
                        traj_last[tls_id].append(dones[tls_id])
                        # print(traj_last)
                        buffer = buffer_list[tls_id]                     
                        buffer.add_transition(traj_last[tls_id][0],traj_last[tls_id][1],
                                              traj_last[tls_id][2],traj_last[tls_id][3],
                                              traj_last[tls_id][4],traj_last[tls_id][5],
                                              traj_last[tls_id][6])
                                          
                        if buffer._size > sample_size:
                            agent = agent_list[tls_id]

                            policy_loss, alpha_loss, bellmann_error1, bellmann_error2 = agent.learn(buffer.sample(sample_size))
                            
                # 释放 GPU 缓存，避免中间变量残留
                torch.cuda.empty_cache()
                gc.collect()
              
            if num_act_agent > 0:
                for tls_id in act_agent_ids:
                    #clear trajectory
                    traj_last[tls_id] = traj_current[tls_id]
                    traj_current[tls_id] = []
                    
            # update the state
            RV_states = next_RV_states
            PL_states = next_PL_states
            
            done = []
            for tls_id in inters_list:
                done.append(dones[tls_id][0])
            done = np.any(done)
            
            decision_flag = next_decision_flag
            
            # save policy
            if done and episode % 10 == 0:
                env.save_policy(episode, save_path)
                # env.save_veh_num_vs_light(episode, save_path)    
        
        total_steps += 1
        # update the learning rate
        for tls_id in inters_list:
            agent = agent_list[tls_id]
            agent.lr_decay(total_steps, num_episodes)
            
        # update eps greedy policy
        # for tls_id in inters_list:
        #     eps_list[tls_id] = max(min_eps, eps_list[tls_id] * eps_decay_rate)
        
        # veh_avg_delay
        veh_avg_delay, RV_avg_sd = env.get_delay_avg_total(RV_schedule, save_path, episode)
        # kill the sumo process
        env.close()
        # save the rewards
        episode_rewards.append(episode_reward)
        # save the model parms
        if episode % 50 == 0:
            for tls_id in inters_list:
                agent = agent_list[tls_id]
                agent.save(os.path.join(save_path, f"agent{tls_id}_" + f"{agent_select}" + "_model_parameters"), episode)
            
        end = time.time()
        
        # print the result
        print("Episode: {}, Agent Reward: {}, Veh_avg_delay: {}, RV_avg_sd: {}".format(episode, episode_reward, veh_avg_delay, RV_avg_sd), ", Time: %s"%(end - start))
        if WANDB:
            wandb.log({"RV_avg_sd": RV_avg_sd, "veh_avg_delay": veh_avg_delay, "episode_reward": episode_reward})

    return episode_rewards






if __name__ == "__main__":
    # configs
    # wandb parameter
    WANDB = True
    # wandb key
    if WANDB:
        wandb.login(key='6d79f8ce510ee51b794332f57a750f6647ad72d7')
        
    # online_train
    mode_online_train = True
    
    # sumo setting
    sumo_visualization = False
    
    # agent
    agent_select = 'SACTF-mask-beta000' # SACTF-mask
    
    # random seed (date)
    seed=20250102
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"CUDA is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # simulation time
    T = 30
    dt = 60
    max_steps = T * dt
    
    # simulation env
    cell_num = 3
    intersection = 'qinzhou'
    p = 20 #initial RV penetration rate (%)
    RV_schedule_file = f"envs/{intersection}/{intersection}_{p}_schedule.json"
    sumocfg_file = f"./envs/{intersection}/{intersection}.sumocfg"
    net_file = f"./envs/{intersection}/{intersection}.net.xml"
    rou_file = f"./envs/{intersection}/Demand_{p}.rou.xml"
    MPtripinfo_path =  f"./envs/{intersection}/tripinfo_MP.xml"
    
    # RL agent setting
    buffer_size = 99999
    sample_size = 32
    num_episodes = 200
    
    # multi agents dict
    agent_list = {}
    buffer_list = {}
    
    # state and action dim for buffer replay (SACTF)
    action_dim = 1
    RV_state_dim = 2 # each vehicle state
    PL_state_dim = 33
    seq_len = 10*8+1 # each movement collects 10 vehicles, and a special token   
    # total size of state and action(SACTF)
    action_size = 4
    RV_state_size = 2 # each vehicle state
    PL_state_size = 33
    # network design
    hidden_size = 8

    
    # save setting
    online_train_save_path = f'./online_train/{intersection}'
    
    if WANDB:
        run = wandb.init(
        # Set the project where this run will be logged
        project=f"{intersection}-{agent_select}-{seed}",
        # Track hyperparameters and run metadata
        config={
            "epochs": num_episodes,
        },
        group="experiment_1",
        # group="experiment_1"
        )


    
    #----------------------- online train the agent -----------------------#
    if mode_online_train:
        
        inters_list = utils.extract_trafficlight_ids(net_file)
        num_inters = len(inters_list)
        num_agents = len(inters_list)
        
        # create online train dictionary
        online_train_save_path = os.path.join(online_train_save_path, f'online_train_{agent_select}_multi_agents')
        if not os.path.exists(online_train_save_path):
            os.makedirs(online_train_save_path)
        
        # create 'tripinfo' dictionary
        tripinfo_path = os.path.join(online_train_save_path, 'tripinfo')
        if not os.path.exists(tripinfo_path):
            os.makedirs(tripinfo_path)
        
        # generator_offline_train = TrafficGenerator(net_file_static, trip_file, veh_params, demand, online_train_save_path)
        sumo_cmd_offline_train = utils.set_sumo(sumo_visualization, sumocfg_file, max_steps)
        
        env_online_train = Intersec_Env(rou_file,
                                        net_file,
                                        num_agents,
                                        sumo_cmd_offline_train,
                                        max_steps,
                                        seq_len,
                                        cell_num,
                                        WANDB,
                                        seed)
        # RV_schedule = env_online_train.set_RV_schedule(RV_schedule_file, sumocfg_file, net_file, rou_file, MPtripinfo_path)
        # with open(RV_schedule_file, 'w', encoding='utf-8') as json_file:
        #     json.dump(RV_schedule, json_file, ensure_ascii=False, indent=4)
            
        if os.path.exists(RV_schedule_file):
            print('Schedule alreay exits!!!')
            with open(RV_schedule_file, 'r', encoding='utf-8') as json_file:
                RV_schedule = json.load(json_file)
            print('Successfully read schedule file!!!')
        else:
            print('Schedule does not exit, we need to run MP to creat schedule')
            RV_schedule = env_online_train.set_RV_schedule(RV_schedule_file, sumocfg_file, net_file, rou_file, MPtripinfo_path)
            with open(RV_schedule_file, 'w', encoding='utf-8') as json_file:
                json.dump(RV_schedule, json_file, ensure_ascii=False, indent=4)
            print('Schedule has been created by MP control!!!')
        
        # state size is 33+24*2
        for i in inters_list:
            buffer_list[str(i)]=ReplayBuffer(RV_state_dim=RV_state_dim,
                                             PL_state_dim=PL_state_dim,
                                             seq_len=seq_len,
                                             action_dim=action_dim,
                                             buffer_size=buffer_size,
                                             device=device)
        
        # train the agent
        for i in inters_list:
            agent_list[str(i)]=SACTFAgent(i,
                                          RV_state_size,
                                          PL_state_size,
                                          seq_len,
                                          action_size,
                                          hidden_size=hidden_size,
                                          device=device,
                                          agent_select=agent_select)
        
      
        episode_rewards = online_train(agent_list, env_online_train, buffer_list, inters_list, RV_schedule, num_episodes=num_episodes, save_path = online_train_save_path)
        # Save the training data
        episode_rewards_pd = pd.DataFrame(episode_rewards).to_csv(os.path.join(online_train_save_path, f'{agent_select}_episode_rewards.csv'))
        
        
