import os
import time
import wandb
import torch
import numpy as np
import pandas as pd
import src.utils.utils as utils
from src.agent.sac_agent import SACAgent
from src.road_env.env_network_multi_agents_TSP import Intersec_Env
from src.network.replay_buffer import ReplayBuffer
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
    
    eps_list = {}
    traj_current = {} #states, action, rewards, next_states, dones
    traj_last = {}
    for tls_id in inters_list:
        eps_list[tls_id] = 1
        traj_current[tls_id] = []
        traj_last[tls_id] = []
    
    min_eps = 0
    eps_decay_rate = 0.98
    # d_eps = 1 - min_eps
    # eps_frames = 20000
    # eps_step_list = [0 for _ in range(num_agents)]
    total_steps = 0
    
    for episode in range(num_episodes + 1):
        print('-------------  current episode: '+str(episode)+'  --------------')
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        # print('current eps: ', eps_list)
        # Reset the environment
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
            # print('num_act_agent',num_act_agent)
            # print('act_agent_ids',act_agent_ids)
            if num_act_agent > 0:
                for tls_id in act_agent_ids:
                    # print(f'signal {tls_id} take a decision')
                    agent = agent_list[tls_id]
                    if agent_select == 'SAC-top3SV':
                        action[tls_id] = agent.get_action_online(states[tls_id])
                
                    # the trjectory is already cleared
                    traj_current[tls_id].append(states[tls_id])
                    traj_current[tls_id].append([action[tls_id]])
            
            next_states, rewards, dones, next_decision_flag = env.step(action, RV_schedule)
            
            if num_act_agent > 0:
                for tls_id in act_agent_ids:
                    if len(traj_last[tls_id]) != 0:
                        # print(f'signal {tls_id} record the trajectory')
                        episode_reward[tls_id] += rewards[tls_id][0]
                        # rewards of last action
                        traj_last[tls_id].append(rewards[tls_id])
                        #current state is the next state of last action
                        traj_last[tls_id].append(states[tls_id])
                        traj_last[tls_id].append(dones[tls_id])
                        # print(traj_last)
                        buffer = buffer_list[tls_id]
                        
                        # print('states: ', traj_last[int(tls_id)][0])
                        # print('action: ',traj_last[int(tls_id)][1])
                        # print('rewards: ', traj_last[int(tls_id)][2])
                        # print('next_states: ', traj_last[int(tls_id)][3])
                        # print('dones: ', traj_last[int(tls_id)][4])
                        
                        buffer.add_transition(traj_last[tls_id][0],traj_last[tls_id][1],
                                              traj_last[tls_id][2],traj_last[tls_id][3],
                                              traj_last[tls_id][4])
                        
                        
                        if buffer._size > sample_size:
                            agent = agent_list[tls_id]
                            # Train the agent
                            if agent_select == 'SAC-top3SV':
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
            states = next_states
            
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
            agent.lr_decay(total_steps,num_episodes)
            
        # update eps greedy policy
        for tls_id in inters_list:
            eps_list[tls_id] = max(min_eps, eps_list[tls_id] * eps_decay_rate)
        
        # veh_avg_delay
        veh_avg_delay,RV_avg_sd = env.get_delay_avg_total(RV_schedule, save_path, episode)
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
            # if agent_select == 'CQL' or agent_select == 'DQN':
            #     wandb.log({"RV_avg_sd": RV_avg_sd, "veh_avg_delay": veh_avg_delay, "episode_reward": episode_reward}) #"loss": loss, "cql_loss": cql_loss, "bellman_error": bellman_error})
            # elif agent_select == 'SAC':
            #     wandb.log({"RV_avg_sd": RV_avg_sd, "veh_avg_delay": veh_avg_delay, "episode_reward": episode_reward}) #"policy_loss": policy_loss, "alpha_loss": alpha_loss, "bellmann_error1": bellmann_error1, "bellmann_error2": bellmann_error2, "cql1_loss": cql1_loss, "cql2_loss": cql2_loss, "current_alpha": current_alpha, "lagrange_alpha_loss": lagrange_alpha_loss, "lagrange_alpha": lagrange_alpha})
            
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
    agent_select = 'SAC-top3SV' # CQL, DQN, SAC, D3QN, A2C
    
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
    intersection = '5x5'
    p = 20 #RV penetration rate (%)
    RV_schedule_file = f"envs/{intersection}/{intersection}_{p}_schedule.json"
    sumocfg_file = f"./envs/{intersection}/{intersection}.sumocfg"
    net_file = f"./envs/{intersection}/{intersection}.net.xml"
    rou_file = f"./envs/{intersection}/Demand_{p}.rou.xml"
    MPtripinfo_path =  f"./envs/{intersection}/tripinfo_MP.xml"
    
    # RL agent setting
    buffer_size = 99999
    sample_size = 128
    num_episodes = 200
    
    # multi agents dict
    agent_list = {}
    buffer_list = {}
    
    # state and action dim for buffer replay
    state_dim = 33+24*2
    action_dim = 1
    
    # total size of state and action
    state_size = 33+24*2
    action_size = 4
    
    # save setting
    online_train_save_path = f'./online_train/{intersection}'
    
    if WANDB:
        run = wandb.init(
        # Set the project where this run will be logged
        project=f"{intersection}-TSP-{agent_select}-{seed}",
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
        env_online_train = Intersec_Env(num_agents, sumo_cmd_offline_train, max_steps, cell_num, WANDB, seed)
        
        # RV_schedule = env_online_train.set_RV_schedule(rou_file)
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
            buffer_list[str(i)]=ReplayBuffer(state_dim=state_dim, action_dim=action_dim, buffer_size=buffer_size, device=device)
        
        # train the agent
        for i in inters_list:
            if agent_select == 'SAC-top3SV':
                agent_list[str(i)]=SACAgent(state_size=state_size, action_size=action_size, hidden_size=256, device=device, agent_select=agent_select)

                
        episode_rewards = online_train(agent_list, env_online_train, buffer_list, inters_list, RV_schedule, num_episodes=num_episodes, save_path = online_train_save_path)
        # Save the training data
        episode_rewards_pd = pd.DataFrame(episode_rewards).to_csv(os.path.join(online_train_save_path, f'{agent_select}_episode_rewards.csv'))
        
        
