# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:57:23 2024

@author: poc
"""

import os
import time
import wandb
import torch
import numpy as np
import pandas as pd
import src.utils.utils as utils
from src.agent.sac_transformer_agent import SACTFAgent #proposed
from src.agent.sac_transformer_agent_onlySV import SACTFAgent as SACTFAgent_onlySV
from src.agent.sac_agent import SACAgent

from src.road_env.env_network_transformer import Intersec_Env as Intersec_Env_SACTF
from src.road_env.env_network_transformer_onlySV import Intersec_Env as Intersec_Env_SACTF_onlySV
from src.road_env.env_network_PL import Intersec_Env as Intersec_Env_PL
from src.road_env.env_network_multi_agents_TSP import Intersec_Env as Intersec_Env_top3SV

import json
import gc
import bs4
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


# Set the maximum size for memory allocations to 1024 MiB, which helps avoid fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
# Release all unused cached memory that PyTorch has cached for future use.
torch.cuda.empty_cache()
# Force garbage collection to clean up any unreferenced objects in memory.
gc.collect()

def evaluate(agent_list, env, inters_list, RV_schedule, seed, data_df, save_path = None):
    total_steps = 0
    
    for episode in range(1):
        print('-------------  current random seed: '+str(seed)+'  --------------')
        
        if agent_select == 'SACTF-mask':
            RV_states, PL_states, net_info = env.reset(RV_schedule, save_path, episode)
        elif agent_select == 'SACTF-mask-onlySV':
            RV_states, net_info = env.reset(RV_schedule, save_path, episode)
        elif agent_select == 'SAC':
            states,net_info = env.reset(RV_schedule, save_path, episode)
        elif agent_select == 'SAC-top3SV':
            states,net_info = env.reset(RV_schedule, save_path, episode)
        
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
                    if agent_select == 'SACTF-mask':
                        action[tls_id] = agent.get_action_online(RV_states[tls_id], PL_states[tls_id])
                    elif agent_select == 'SACTF-mask-onlySV':
                        action[tls_id] = agent.get_action_online(RV_states[tls_id])
                    elif agent_select == 'SAC':
                        action[tls_id] = agent.get_action_online(states[tls_id])
                    elif agent_select == 'SAC-top3SV':
                        action[tls_id] = agent.get_action_online(states[tls_id])
                    
            if agent_select == 'SACTF-mask':
                next_RV_states, next_PL_states, rewards, dones, next_decision_flag, travel_t = env.step(action, RV_schedule)
            elif agent_select == 'SACTF-mask-onlySV':
                next_RV_states, rewards, dones, next_decision_flag, travel_t = env.step(action, RV_schedule)
            elif agent_select == 'SAC':
                next_states, rewards, dones, next_decision_flag = env.step(action, RV_schedule)
            elif agent_select == 'SAC-top3SV':
                next_states, rewards, dones, next_decision_flag = env.step(action, RV_schedule)
            

            # update the state
            if agent_select == 'SACTF-mask':
                RV_states = next_RV_states
                PL_states = next_PL_states
            elif agent_select == 'SACTF-mask-onlySV':
                RV_states = next_RV_states
            elif agent_select == 'SAC':
                states = next_states
            elif agent_select == 'SAC-top3SV':
                states = next_states
            
            done = []
            for tls_id in inters_list:
                done.append(dones[tls_id][0])
            done = np.any(done)
            
            decision_flag = next_decision_flag
            
        
        total_steps += 1

        
        v_d = [] #delay of NV
        RV_d = [] #schedule delay of RV
        RV_list = list(RV_schedule.keys())
        tripinfo_file = save_path+'/tripinfo.xml'
        soup = bs4.BeautifulSoup(open(tripinfo_file),'html.parser')
        soup = soup.find_all('tripinfo')
        for item in soup:
            real_arrival = float(item['arrival'])
            if item['id'] in RV_list:
                schedule_arrival = RV_schedule[item['id']]
                RV_d.append(real_arrival-schedule_arrival)
            else:
                free_arrival = float(item['depart'])+float(item['routelength'])/13.89
                v_d.append(real_arrival-free_arrival)
        
        veh_avg_delay = np.mean(v_d)
        RV_avg_sd = np.mean(RV_d)
        
        print('veh_avg_delay: ', veh_avg_delay)
        print('RV_avg_sd', RV_avg_sd)
        
        # kill the sumo process
        env.close()

        end = time.time()
        print(f'simulation time: {end-start}s')
        
        number = len(data_df)
        data_df.loc[number, 'Average delay'] = veh_avg_delay
        data_df.loc[number, 'Average schedule delay'] = RV_avg_sd
        
        if agent_select == 'SACTF-mask':
            return data_df, travel_t
        else:
            return data_df  


if __name__ == "__main__":
    # configs
    # wandb parameter
    WANDB = False
    # # wandb key
    # if WANDB:
    #     wandb.login(key='6d79f8ce510ee51b794332f57a750f6647ad72d7')
        
    # online_train
    mode_online_train = False
    
    # sumo setting
    sumo_visualization = False
    
    # agent
     #SACTF-mask(proposed),SAC(PL),SACTF-mask-onlySV(Only-SV),SAC-top3SV(Fixed-SV)
    all_agent_list = [ 'SACTF-mask'] # 'SACTF-mask', 'SAC', 'SACTF-mask-onlySV', 'SAC-top3SV'

    metric = ['Average delay', 'Average schedule delay']
    
    for agent_select in all_agent_list:
        print(f'curren agent: {agent_select}')
        
        data_df = pd.DataFrame(columns=metric)
        
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
        intersection = '5x5'
        p = 20 #RV penetration rate (%)
        
        sumocfg_file = f"./envs/{intersection}/{intersection}.sumocfg"
        net_file = f"./envs/{intersection}/{intersection}.net.xml"
        rou_file = f"./envs/{intersection}/Demand_{p}.rou.xml"
        MPtripinfo_path =  f"./envs/{intersection}/tripinfo_MP.xml"
        
        # RL agent setting
        # buffer_size = 99999
        # sample_size = 32
        # num_episodes = 200
        
        # multi agents dict
        agent_list = {}
        # buffer_list = {}
        
        action_size = 4
        
        if agent_select == 'SACTF-mask':
            # state and action dim for buffer replay (proposed-SACTF)
            seq_len = 10*8+1 # each movement collects 10 vehicles, and a special token   
            # total size of state and action(SACTF)
            RV_state_size = 2 # each vehicle state
            PL_state_size = 33
            # network design
            hidden_size = 8
            
        elif agent_select == 'SACTF-mask-onlySV':
            seq_len = 10*8+1 # each movement collects 10 vehicles, and a special token   
            # total size of state and action(SACTF)
            RV_state_size = 2 # each vehicle state
            # network design
            hidden_size = 8
            
        elif agent_select == 'SAC':
            # total size of state and action (SAC-PL)
            state_size = 33
            action_size = 4
            
        elif agent_select == 'SAC-top3SV':
            state_size = 33+24*2
            action_size = 4
        
    
        
        #----------------------- evaluate -----------------------#
        if not mode_online_train:
            # random seed (date)
            for n in range(10):
                seed=n
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                
                inters_list = utils.extract_trafficlight_ids(net_file)
                num_inters = len(inters_list)
                num_agents = len(inters_list)
                
                # online_train save setting
                online_train_save_path = f'./online_train/{intersection}'
                
                # create online train dictionary
                online_train_save_path = os.path.join(online_train_save_path, f'online_train_{agent_select}_multi_agents')
                
                
                evaluate_save_path = os.path.join( f'./evaluate/{intersection}', f'evaluate_{agent_select}_multi_agents',f'random_{seed}')
                if not os.path.exists(evaluate_save_path):
                    os.makedirs(evaluate_save_path)
                
                RV_schedule_file = f"evaluate/{intersection}/evaluate_{agent_select}_multi_agents/random_{seed}/RV_schedule_{p}.json"
                
                # generator_offline_train = TrafficGenerator(net_file_static, trip_file, veh_params, demand, online_train_save_path)
                sumo_cmd_offline_train = utils.set_sumo(sumo_visualization, sumocfg_file, max_steps)
                
                if agent_select == 'SACTF-mask':
                    env_evaluate = Intersec_Env_SACTF(rou_file, net_file,
                                                    num_agents,
                                                    sumo_cmd_offline_train,
                                                    max_steps,
                                                    seq_len,
                                                    cell_num,
                                                    WANDB,
                                                    seed)

                    if os.path.exists(RV_schedule_file):
                        print('Schedule alreay exits!!!')
                        with open(RV_schedule_file, 'r', encoding='utf-8') as json_file:
                            RV_schedule = json.load(json_file)
                        print('Successfully read schedule file!!!')
                    else:
                        print('Schedule does not exit, we need to run MP to creat schedule')
                        RV_schedule = env_evaluate.set_RV_schedule(RV_schedule_file, sumocfg_file, net_file, rou_file, MPtripinfo_path)
                        with open(RV_schedule_file, 'w', encoding='utf-8') as json_file:
                            json.dump(RV_schedule, json_file, ensure_ascii=False, indent=4)
                        print('Schedule has been created by MP control!!!')
                    
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
        
        
                elif agent_select == 'SACTF-mask-onlySV':
                    env_evaluate = Intersec_Env_SACTF_onlySV(rou_file, net_file,
                                                            num_agents,
                                                            sumo_cmd_offline_train,
                                                            max_steps,
                                                            seq_len,
                                                            cell_num,
                                                            WANDB,
                                                            seed)

                    if os.path.exists(RV_schedule_file):
                        print('Schedule alreay exits!!!')
                        with open(RV_schedule_file, 'r', encoding='utf-8') as json_file:
                            RV_schedule = json.load(json_file)
                        print('Successfully read schedule file!!!')
                    else:
                        print('Schedule does not exit, we need to run MP to creat schedule')
                        RV_schedule = env_evaluate.set_RV_schedule(RV_schedule_file, sumocfg_file, net_file, rou_file, MPtripinfo_path)
                        with open(RV_schedule_file, 'w', encoding='utf-8') as json_file:
                            json.dump(RV_schedule, json_file, ensure_ascii=False, indent=4)
                        print('Schedule has been created by MP control!!!')
                    
                    # train the agent
                    for i in inters_list:
                        agent_list[str(i)]=SACTFAgent_onlySV(i,
                                                              RV_state_size,
                                                              seq_len,
                                                              action_size,
                                                              hidden_size=hidden_size,
                                                              device=device,
                                                              agent_select=agent_select)
                 
                        
                elif agent_select == 'SAC':
                    env_evaluate = Intersec_Env_PL(num_agents, sumo_cmd_offline_train, max_steps, cell_num, WANDB, seed)

                    if os.path.exists(RV_schedule_file):
                        print('Schedule alreay exits!!!')
                        with open(RV_schedule_file, 'r', encoding='utf-8') as json_file:
                            RV_schedule = json.load(json_file)
                        print('Successfully read schedule file!!!')
                    else:
                        print('Schedule does not exit, we need to run MP to creat schedule')
                        RV_schedule = env_evaluate.set_RV_schedule(RV_schedule_file, sumocfg_file, net_file, rou_file, MPtripinfo_path)
                        with open(RV_schedule_file, 'w', encoding='utf-8') as json_file:
                            json.dump(RV_schedule, json_file, ensure_ascii=False, indent=4)
                        print('Schedule has been created by MP control!!!')
                        
                    # train the agent
                    for i in inters_list:
                        agent_list[str(i)]=SACAgent(state_size=state_size, action_size=action_size, hidden_size=256, device=device, agent_select=agent_select)
                    
        
                
                elif agent_select == 'SAC-top3SV':
                    env_evaluate = Intersec_Env_top3SV(num_agents, sumo_cmd_offline_train, max_steps, cell_num, WANDB, seed)

                    if os.path.exists(RV_schedule_file):
                        print('Schedule alreay exits!!!')
                        with open(RV_schedule_file, 'r', encoding='utf-8') as json_file:
                            RV_schedule = json.load(json_file)
                        print('Successfully read schedule file!!!')
                    else:
                        print('Schedule does not exit, we need to run MP to creat schedule')
                        RV_schedule = env_evaluate.set_RV_schedule(RV_schedule_file, sumocfg_file, net_file, rou_file, MPtripinfo_path)
                        with open(RV_schedule_file, 'w', encoding='utf-8') as json_file:
                            json.dump(RV_schedule, json_file, ensure_ascii=False, indent=4)
                        print('Schedule has been created by MP control!!!')
                    
                    # train the agent
                    for i in inters_list:
                        agent_list[str(i)]=SACAgent(state_size=state_size, action_size=action_size, hidden_size=256, device=device, agent_select=agent_select)
                
                    
                # load actor parameter
                for i in inters_list:
                    agent_list[str(i)].load(os.path.join(online_train_save_path, f"agent{i}_" + f"{agent_select}" + "_model_parameters"), 200)
                
                if agent_select == 'SACTF-mask':
                    data_df, travel_t = evaluate(agent_list, env_evaluate, inters_list, RV_schedule, seed, data_df, save_path = evaluate_save_path)
                    travel_t_file = f"evaluate/{intersection}/evaluate_{agent_select}_multi_agents/random_{seed}/travel_t.json"
                    with open(travel_t_file, 'w', encoding='utf-8') as json_file:
                        json.dump(travel_t, json_file, ensure_ascii=False, indent=4)
                else:
                    data_df = evaluate(agent_list, env_evaluate, inters_list, RV_schedule, seed, data_df, save_path = evaluate_save_path)
            
        data_df.to_csv(os.path.join( f'./evaluate/{intersection}', f'evaluate_{agent_select}_multi_agents', f'{agent_select}_metric.csv'), index=False)
        
