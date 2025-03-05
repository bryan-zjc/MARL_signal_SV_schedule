# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:39:20 2024

@author: Jichen Zhu
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from src.utils import utils
from src.utils.agent.MP_agent import MPAgent 
from src.utils.road_env.env_MP import Intersec_Env
import json
import xml.etree.ElementTree as ET
from collections import defaultdict



class schedule_generator:
    def __init__(self, RV_schedule_file, sumocfg_file, net_file, rou_file, MPtripinfo_path):
        super(schedule_generator, self).__init__()
        self.sumo_visualization = False
        self.RV_schedule_file, self.sumocfg_file, self.net_file, self.rou_file = RV_schedule_file, sumocfg_file, net_file, rou_file
        self.MPtripinfo_path = MPtripinfo_path
        
    def exe_MP(self):
        # random seed (date)
        seed=20250102
        np.random.seed(seed)

        # simulation time
        T = 30
        dt = 60
        max_steps = T * dt
        
        # simulation env
        intersection = '5x5'
        
        # multi agents dict
        agent_list = {}
        
        # save setting
        MPtripinfo_path = self.MPtripinfo_path
     
        inters_list = utils.extract_trafficlight_ids(self.net_file)
        num_inters = len(inters_list)
        num_agents = len(inters_list)
        
      
        # generator_offline_train = TrafficGenerator(net_file_static, trip_file, veh_params, demand, online_train_save_path)
        sumo_cmd_offline_train = utils.set_sumo(self.sumo_visualization, self.sumocfg_file, max_steps)
        
        env_online_train = Intersec_Env(self.rou_file,
                                        self.net_file,
                                        num_agents,
                                        sumo_cmd_offline_train,
                                        max_steps,
                                        seed)
        # train the agent
        for i in inters_list:
            agent_list[str(i)]=MPAgent(i)

        self.runMP(agent_list, env_online_train, inters_list, MPtripinfo_path)



    def runMP(self, agent_list, env, inters_list, MPtripinfo_path = None):
        total_steps = 0
        # Reset the environment
        total_pressure, net_info = env.reset(MPtripinfo_path)
        # print(net_info)
        done = False
        decision_flag = {}
        action={}
        for tls_id in net_info.keys():
            decision_flag[tls_id] = 0
            action[tls_id] = 0
    
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
    
                    action[tls_id] = agent.get_action(total_pressure[tls_id])
                    
            
            total_pressure, dones, next_decision_flag = env.step(action)
            
            
            done = []
            for tls_id in inters_list:
                done.append(dones[tls_id][0])
            done = np.any(done)
            
            decision_flag = next_decision_flag
              
        
        total_steps += 1
    
        # kill the sumo process
        env.close()


    def get_avg_travel_time(self, tripinfo_path):
        tree = ET.parse(tripinfo_path)
        root = tree.getroot()
        
        # 2. 创建一个字典，用于存储每个 OD 对对应的旅行时间列表
        od_travel_times = defaultdict(list)
        
        # 3. 遍历 XML 中每个 tripinfo 元素
        for trip in root.findall('tripinfo'):
            # 获取车辆 id，例如 "OD98_NV.0"
            vehicle_id = trip.get("id")
            # 获取旅行时间（duration），转成 float 类型
            duration = float(trip.get("duration"))
            
            # 根据车辆 id 提取 OD 对信息：取下划线 "_" 前的部分
            # 例如 "OD98_NV.0" -> "OD98"
            od_pair = vehicle_id.split('_')[0]
            
            # 将该车辆的旅行时间加入对应 OD 对的列表中
            od_travel_times[od_pair].append(duration)
        
        od_travel_time = {}
        
        for od, durations in od_travel_times.items():
            # 如果该 OD 对下有数据，则计算平均旅行时间
            if durations:
                average_duration = sum(durations) / len(durations)
            else:
                average_duration = 0
            od_travel_time[od] = average_duration
            
        return od_travel_time
    
    def schedule_creat(self, vehID, depart_time, od_travel_time):
        od = vehID.split('_')[0]
        schedule_time = round(depart_time + od_travel_time[od])
        return schedule_time


        
