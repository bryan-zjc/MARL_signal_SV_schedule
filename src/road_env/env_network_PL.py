import os
import traci
# import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import bs4
import random
import copy
import src.utils.schedule_generator as schedule_generator

'''
Intersec_Env for rl offline data collection, training, and fine-tuning
'''

class Intersec_Env:
    def __init__(self, num_agent, sumo_cmd, max_steps, cell_num, wandb, seed):
        # sumo config
        # self.MP_PR = MP_PR
        self._max_steps = max_steps
        self._sumo_cmd = sumo_cmd
        # self.t_sequence = t_sequence
        # self._generator = generator

        # signal control parameters config
        self._yellow = 3
        self._green_min = 10
        self._green_max = 40
        
        # agent config
        self.num_agent = num_agent
        self.cell_num = 3
        # self.control_input = [0 for _ in range(num_agent)]
        self.net_info = None # a dict to store the basic intersection info
        self.net_links = [] # a list to store the edge id in the network (except the out links at the border)
        
        # code config
        # self.wandb = wandb
        
        # random seed
        self.seed = seed
        
        # expected speed for RV
        self.v_exp = 20/3.6
        


    def reset(self,RV_schedule, save_path, episode):
        # start sumo with traci
        if 'online_train' in save_path: #online_train
            sumo_cmd1 = self._sumo_cmd + ["--tripinfo-output",save_path+f'/tripinfo/tripinfo_ep{episode}.xml']
        elif 'sa' in save_path: #sa
            sumo_cmd1 = self._sumo_cmd + ["--tripinfo-output",save_path+'/tripinfo.xml']
        else: #evaluate
            sumo_cmd1 = self._sumo_cmd + ["--tripinfo-output",save_path+'/tripinfo.xml',"--fcd-output",save_path+"/fcd.xml"]
        traci.start(sumo_cmd1)
        print("----- connection to sumo established")
        self.save_path = save_path
        self.episode = episode

        # retrieve the basic inter info
        if not self.net_info:
            tls_ids = traci.trafficlight.getIDList()
            self.net_info = defaultdict(dict)
            for tls_id in tls_ids:
                phase_num = len((traci.trafficlight.getAllProgramLogics(tls_id)[0]).phases)
                lanes_all, lanes_in, lanes_out, lanes_connect, edge_in = self._read_in_and_out(traci.trafficlight.getControlledLinks(tls_id))
                self.net_info[tls_id]['phase_num'] = phase_num
                self.net_info[tls_id]['lanes_all'] = lanes_all
                self.net_info[tls_id]['lanes_in'] = lanes_in
                self.net_info[tls_id]['lanes_out'] = lanes_out
                self.net_info[tls_id]['lanes_connect'] = lanes_connect
                self.net_info[tls_id]['edges_in'] = edge_in
                self.net_info[tls_id]['phase2lane'], self.net_info[tls_id]['lane2phase'] = self._read_phase2lane(tls_id)
                
                self.net_info[tls_id]['lanes_in_no_right'] = []
                self.net_info[tls_id]['lanes_out_no_right'] = []
                for lane in lanes_in:
                    if lane.split('_')[-1] != '0':
                        self.net_info[tls_id]['lanes_in_no_right'].append(lane)
                for lane in lanes_out:
                    if lane.split('_')[-1] != '0':
                        self.net_info[tls_id]['lanes_out_no_right'].append(lane)
                
                self.net_links+=edge_in
                
        # print(self.net_info)
        
        # initialize the env
        self.phase_switch_pointer = {} # save the start time of current phase
        self._policy_dict = {} # save the policy info
        
        self.g_max_flag = {}
        
        for tls_id in self.net_info.keys():
            traci.trafficlight.setPhase(tls_id, 0)
            self.phase_switch_pointer[tls_id] = 0
            self._policy_dict[tls_id] = []
            self.g_max_flag[tls_id] = 0
        
        self._step = 0

        self._simulate()
        self.decision_step = {}
        
        for tls_id in self.net_info.keys():
            self.decision_step[tls_id] = self._step + self._green_min
        
        
        print(f'----- Env including {len(self.net_info.keys())} intersections is ready.')
        
        states = {}
        for tls_id in self.net_info.keys():
            states[tls_id] = self._get_state(tls_id, RV_schedule)
        # states = np.stack(states)

        return states,self.net_info


    def close(self):
        traci.close()
        print("----- connection to sumo closed")


    def step(self, action, RV_schedule):
        # self.control_input = action
        # print('self.g_max_flag: ', self.g_max_flag)
        self.control_input = {}
        for tls_id in self.net_info.keys():
            if self.g_max_flag[tls_id] == 1:
                current_phase = traci.trafficlight.getPhase(tls_id)
                self.control_input[tls_id] = ((current_phase + 1) % self.net_info[tls_id]['phase_num'])/2 #人为修改下一次的执行相位，此时相位为黄灯
            else:
                self.control_input[tls_id] = action[tls_id] # apply action to the signal
        # print('control input: ', self.control_input)
        self._update_env() # update the env
        

        states = {}
        rewards = {}
        dones = {}
        # infos = []
        for tls_id in self.net_info.keys():
            # print(tls_id)
            states[tls_id]=[self._get_state(tls_id, RV_schedule)]
            rewards[tls_id]=[self._get_reward(tls_id)]
            dones[tls_id]=[self._step >= self._max_steps-15]
            # infos+=(self._get_info(tls_id))
        # states = np.stack(states)
        # rewards = np.stack(rewards)
        # dones = np.stack(dones)
        # infos = np.stack(infos)

        return states, rewards, dones, self.next_decision_flag


    def _get_state(self, tls_id, RV_schedule):
        # The state is adopted the define in PressLight
        PL_state = self._collect_observation_PL(tls_id)
        # print(f'PL state: {PL_state}')

        state = PL_state
        return state


    def _get_reward(self, tls_id):
        # print(tls_id)
        r1 = self._collect_pressure(tls_id)
        # print("r1:{}".format(r1))
        r1_min = -1
        r1_max = 0
        r1 = (r1 - r1_min) / (r1_max - r1_min)
        # print("r1 normilized:{}".format(r1))

        
        reward = r1
        return reward


    # state function candidates
    def _collect_observation_PL(self, tls_id):
        # do not consider the right turnning movement
        num_info_in = [0] * len(self.net_info[tls_id]['lanes_in_no_right']) * self.cell_num
        num_info_out = [0] * len(self.net_info[tls_id]['lanes_out_no_right'])
        car_list_in = []
        for lane_id in self.net_info[tls_id]['lanes_in_no_right']:
            car_list_in += traci.lane.getLastStepVehicleIDs(lane_id)
        car_list_out = []
        for lane_id in self.net_info[tls_id]['lanes_out_no_right']:
            car_list_out += traci.lane.getLastStepVehicleIDs(lane_id)
        
        for car_id in car_list_in:
            lane_id, lane_pos = traci.vehicle.getLaneID(car_id), traci.vehicle.getLanePosition(car_id)
            # do not consider the right turnning movement
            if lane_id in (self.net_info[tls_id]['lanes_in_no_right']):
                lane_length = traci.lane.getLength(lane_id)
                # num info 
                lane_idx_1 = self.net_info[tls_id]['lanes_in_no_right'].index(lane_id)
                cell_idx = min(int(lane_pos / (lane_length / self.cell_num)), self.cell_num-1) #防止cell_num超出索引
                num_info_in[lane_idx_1 * self.cell_num + cell_idx] += 1
        
        for car_id in car_list_out:      
            lane_id, lane_pos = traci.vehicle.getLaneID(car_id), traci.vehicle.getLanePosition(car_id)
            if lane_id in (self.net_info[tls_id]['lanes_out_no_right']):
                lane_idx_2 = self.net_info[tls_id]['lanes_out_no_right'].index(lane_id)
                num_info_out[lane_idx_2] += 1
        
        current_phase = []
        current_phase.append(traci.trafficlight.getPhase(tls_id))
        
        state = current_phase + num_info_in + num_info_out
        
        return state
        
        
    # reward function candidates
    def _collect_pressure(self, tls_id):
        pressure = 0
        # calculate the scaled total num of vehicles in incoming lanes
        for lane_id in self.net_info[tls_id]['lanes_in_no_right']:
            pressure += traci.lane.getLastStepVehicleNumber(lane_id) / (traci.lane.getLength(lane_id)/7) # 7 is the stop length of each vehicle
        # calculate the scaled total num of vehicles in outgoing lanes
        for lane_id in self.net_info[tls_id]['lanes_out_no_right']:
            pressure -= traci.lane.getLastStepVehicleNumber(lane_id) / (traci.lane.getLength(lane_id)/7)
        # absoulte value
        pressure = abs(pressure)
        return -pressure

    
    
    def _collect_waiting_veh_num(self, tls_id):
        # calculate the num of vehicles which speed is less than 1m/s in incoming lanes
        waiting_veh_num = 0
        for lane_id in self.net_info[tls_id]['lanes_in_no_right']:
            for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                if traci.vehicle.getSpeed(veh_id) < 1:
                    waiting_veh_num += 1
        # scale the waiting_veh_num
        waiting_veh_num = waiting_veh_num / len(self.net_info[tls_id]['lanes_in'])
        return waiting_veh_num
    
    
    def _collect_incoming_veh_num(self, tls_id):
        # calculate the num of vehicles in incoming lanes
        incoming_veh_num = 0
        for lane_id in self.net_info[tls_id]['lanes_in_no_right']:
            incoming_veh_num += traci.lane.getLastStepVehicleNumber(lane_id) / (traci.lane.getLength(lane_id)/7) # 7 is the stop length of each vehicle
        # scale the incoming_veh_num
        incoming_veh_num = incoming_veh_num / len(self.net_info[tls_id]['lanes_in'])
        return incoming_veh_num
    
    
    def _collect_change_of_delay(self, tls_id):
        change_of_delay = max(self._veh_total_delay[-1] - self._veh_total_delay[-2], 0)
        change_of_delay = change_of_delay / 8 / (289.60 / 13.89)
        return change_of_delay

    # network information reading
    def _read_in_and_out(self, controlledlinks):
        lanes_all = []
        lanes_in = []
        lanes_out = []
        lanes_connect = []
        edges_in = []
        for sublist in controlledlinks:
            for item in sublist:
                lanes_in.append(item[0])
                lanes_out.append(item[1])
                lanes_connect.append(item[2])
        lanes_all = (lanes_in + lanes_out)
        # lanes_in is sorted by north-east-south-west
        
        # lanes_in = list(set(lanes_in))
        # lanes_out = list(set(lanes_out))
        # lanes_connect = list(set(lanes_connect))
        # 排序
        # lanes_all.sort()
        # lanes_in.sort()
        # lanes_out.sort()
        # lanes_connect.sort()
        
        for lane in lanes_in:
            edges_in.append(lane[:-2])
        edges_in = list(set(edges_in))
        edges_in.sort()
        
        return tuple(lanes_all), tuple(lanes_in), tuple(lanes_out), tuple(lanes_connect), tuple(edges_in)
    
    
    def _read_phase2lane(self, tls_id):
        phase2lane = {}
        lane2phase = {}
        for phase_idx in range(self.net_info[tls_id]['phase_num']):
            signal_state = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases[phase_idx].state
            # extract the index of G and g in the signal state 并集
            G_idx = [i for i, x in enumerate(signal_state) if x == 'G' or x == 'g']
            list = []
            for idx in G_idx:
                list.append(traci.trafficlight.getControlledLinks(tls_id)[idx][0][0])
                phase2lane[phase_idx] = set(list)
                for edge in list:
                    lane2phase[edge] = phase_idx
        
        return phase2lane, lane2phase
    

    def _update_env(self):
        phases = {}
        switch_flag = {}
        decision_flag = {}
        self.next_decision_flag = {}
        for tls_id in self.net_info.keys():
            phases[tls_id] = traci.trafficlight.getPhase(tls_id)
            # print( traci.simulation.getTime())
            if self.decision_step[tls_id] == traci.simulation.getTime():
                decision_flag[tls_id] = 1
            else:
                decision_flag[tls_id] = 0
            if self.control_input[tls_id]*2 != phases[tls_id]:
                switch_flag[tls_id] = 1
            else:
                switch_flag[tls_id] = 0
        # print('g_max_flag: ', self.g_max_flag)
        # print('current step: ', self._step)
        # print('decision flag:', decision_flag)
        # print('switch flag: ', switch_flag)
        # print('control input: ', self.control_input)
        # print('decision_step: ', self.decision_step)
        # print('phase_switch_pointer: ', self.phase_switch_pointer)
        # print('-------------------------------------------------')
        # set the action to signal switch a large step when the phase is in yellow
        for tls_id in self.net_info.keys():
            if decision_flag[tls_id] == 1:
                if switch_flag[tls_id] == 1:
                    # exectute yellow phase
                    self._set_next_phase(tls_id)
                    self.decision_step[tls_id] = traci.simulation.getTime()+self._yellow+self._green_min
                else:
                    if traci.simulation.getTime() == self.phase_switch_pointer[tls_id] + self._green_max:
                        self.g_max_flag[tls_id] = 1
                        self._set_next_phase(tls_id)
                        self.decision_step[tls_id] = traci.simulation.getTime()+self._yellow+self._green_min
                    else:
                        self._set_control_phase(tls_id)
                        self.decision_step[tls_id] = traci.simulation.getTime()+1
            else: #decision_flag = 0
                if traci.simulation.getTime() == self.phase_switch_pointer[tls_id]+self._yellow and traci.simulation.getTime()>3 and phases[tls_id]%2 == 1:
                    self._set_control_phase(tls_id)
                    self.g_max_flag[tls_id] = 0

        
        self._simulate(steps_todo=1)
        # print('decision_step:',self.decision_step)
        
        for tls_id in self.net_info.keys():
            if self.decision_step[tls_id] == traci.simulation.getTime():
                self.next_decision_flag[tls_id] = 1
            else:
                self.next_decision_flag[tls_id] = 0
        
        
    def _simulate(self, steps_todo=1):
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step
        while steps_todo > 0:
            phases = {}
            for tls_id in self.net_info.keys():
                phases[tls_id] = traci.trafficlight.getPhase(tls_id)
                # print('phases', phases[tls_id])

            traci.simulationStep()  # simulate 1 step in sumo
            
            # collect data to train reward estimator
            next_phases = {}
            for tls_id in self.net_info.keys():
                next_phases[tls_id] = traci.trafficlight.getPhase(tls_id)
                # print('next_phases', next_phases[tls_id])
                self._collect_policy_info(tls_id, next_phases[tls_id])  # save the policy info
                # if next_phases[tls_id] != phases[tls_id]:
                #     self.phase_switch_pointer[tls_id] = traci.simulation.getTime() -1
            
            self._step += 1 # update the step counter
            steps_todo -= 1
            
            # self._veh_delay_collect() # collect the delay info of the vehicles
            
        return self._step
    
    

    def _veh_delay_collect(self):
        for veh_id in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(veh_id)
            for tls_id in self.net_info.keys():
                if lane_id in self.net_info[tls_id]['lanes_in_no_right']:
                    travel_distance = traci.vehicle.getDistance(veh_id)
                    depature_time = traci.vehicle.getDeparture(veh_id)
                    self._veh_delay_info[veh_id] = max(traci.simulation.getTime() - depature_time - 1 - (travel_distance / 13.89), 0)
                    
    
        
        
    def _veh_total_delay_collect(self):
        veh_delay = 0
        for veh_id in self._veh_delay_info.keys():
            veh_delay += self._veh_delay_info[veh_id]
        self._veh_total_delay.append(veh_delay)
    
    
    def get_veh_avg_delay(self):
        veh_delay = 0
        veh_num = 0
        for veh_id in self._veh_delay_info.keys():
            veh_delay += self._veh_delay_info[veh_id]
            veh_num += 1
        return veh_delay / veh_num, veh_delay
    
    
    def _set_next_phase(self, tls_id):
        current_phase = traci.trafficlight.getPhase(tls_id)
        next_phase = (current_phase + 1) % self.net_info[tls_id]['phase_num']
        # print(f'current phase: {current_phase}, next phase: {next_phase}')
        traci.trafficlight.setPhase(tls_id, next_phase)
        
        self.phase_switch_pointer[tls_id] = traci.simulation.getTime()
        # print('time to change the phase', traci.simulation.getTime())
    
    def _set_control_phase(self,tls_id):
        current_phase = traci.trafficlight.getPhase(tls_id)
        next_phase = self.control_input[tls_id]*2
        traci.trafficlight.setPhase(tls_id, next_phase)
        # print('tls_id: ', tls_id)
        # print('current phase: ', current_phase)
        # print('next phase: ', next_phase)
        if current_phase != next_phase:
            self.phase_switch_pointer[tls_id] = traci.simulation.getTime()
    
    
    def _collect_policy_info(self, tls_id, phase):
        self._policy_dict[tls_id].append((self._step, phase))
            
        
    def save_policy(self, episode, path):
        # episode = 0
        for tls_id in self.net_info.keys():
            # save self._policy_dict[tls_id] as csv
            policy = self._policy_dict[tls_id]
            df_policy = pd.DataFrame(policy)
            file_name = os.path.join(path, f'policy_{tls_id}_{episode}.csv')
            df_policy.to_csv(file_name, index=False)
            
            # df_policy是一段连续的相位序列，从中统计出每个相位的持续时间并保存其分布画图
            phase_duration = []
            phase = policy[0][1]
            count = 1
            for i in range(1, len(policy)):
                if policy[i][1] == phase:
                    count += 1
                else:
                    phase_duration.append([phase, count])
                    phase = policy[i][1]
                    count = 1
            df_phase_duration = pd.DataFrame(phase_duration, columns=['phase', 'duration'])
            # save df_phase_duration as csv
            file_name = os.path.join(path, f'phase_duration_{tls_id}_{episode}.csv')
            df_phase_duration.to_csv(file_name, index=False)
            
    
    def set_RV_schedule(self,RV_schedule_file, sumocfg_file, net_file, rou_file, MPtripinfo_path):
        # demand_soup = bs4.BeautifulSoup(open(rou_file),'html.parser')
        # vehicle_soup = demand_soup.find_all('vehicle')
        # route_soup = demand_soup.find_all('route')
        # RV_list = []
        # RV_schedule = {}
        # random.seed(self.seed)
        # for i in range(len(vehicle_soup)):
        #     if vehicle_soup[i]['type'] == 'RV':
        #         RV_list.append(vehicle_soup[i]['id'])
        #         total_dis = len(route_soup[i]['edges'].split())*280
        #         # generate shedule for each RV (the expected arrival time of RV's terminal point)
        #         RV_schedule[vehicle_soup[i]['id']] = round(float(vehicle_soup[i]['depart']) + round(total_dis/self.v_exp)) #+ random.randint(-120,60))
        # self.RV_schedule = RV_schedule
        # return RV_schedule
        
        
        RV_schedule_generator = schedule_generator.schedule_generator(RV_schedule_file, sumocfg_file, net_file, rou_file, MPtripinfo_path)
        if not os.path.exists(MPtripinfo_path):
            print('run MP control')
            RV_schedule_generator.exe_MP()
        
        od_travel_time = RV_schedule_generator.get_avg_travel_time(MPtripinfo_path)
        
        demand_soup = bs4.BeautifulSoup(open(rou_file),'html.parser')
        vehicle_soup = demand_soup.find_all('vehicle')

        RV_list = []
        RV_schedule = {}
        random.seed(self.seed)
        for i in range(len(vehicle_soup)):
            if vehicle_soup[i]['type'] == 'RV':
                RV_list.append(vehicle_soup[i]['id'])
                depart_time = float(vehicle_soup[i]['depart'])
                schedule_time = RV_schedule_generator.schedule_creat(vehicle_soup[i]['id'], depart_time, od_travel_time)
                
                RV_schedule[vehicle_soup[i]['id']] = schedule_time + random.randint(-60,60)
        self.RV_schedule = RV_schedule
        return RV_schedule
    
    def get_delay_avg_total(self,RV_schedule,save_path, episode):
        v_d = [] #delay of NV
        RV_d = [] #schedule delay of RV
        RV_list = list(RV_schedule.keys())
        tripinfo_file = save_path+f'/tripinfo/tripinfo_ep{episode}.xml'
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
        
        return np.mean(v_d), np.mean(RV_d)
        
        
    