import os
import traci
# import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

'''
Intersec_Env for rl offline data collection, training, and fine-tuning
'''

class Intersec_Env:
    def __init__(self, num_agent, sumo_cmd, max_steps, cell_num, wandb):
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
        self.control_input = 0
        self.net_info = None # a dict to store the basic intersection info
        
        # code config
        # self.wandb = wandb


    def reset(self):
        # generate veh trips
        # self._generator.generate_tripfile(self.t_sequence)
        # self._generator.sampling_CVs(self.MP_PR)

        # start sumo with traci
        traci.start(self._sumo_cmd)
        print("----- connection to sumo established")

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

        # initialize the env
        self._veh_traj = defaultdict(list) # collect the traj of the veh
        self._veh_partial_traj = {} # collect the traj of the veh (only the first entry in each edge)
        self._veh_delay_info = {} # collect the delay info of the veh
        self._veh_total_delay = [] # collect the total delay of the veh    
        self.phase_switch_pointer = {} # save the start time of current phase
        self._policy_dict = {} # save the policy info
        self.veh_num_vs_light = defaultdict(list) # save the num of vehicles vs light
        
        for tls_id in self.net_info.keys():
            traci.trafficlight.setPhase(tls_id, 0)
            self.phase_switch_pointer[tls_id] = 0
            self._policy_dict[tls_id] = []
        
        self._step = 0
        self.control_input = 0
        self._simulate(self._green_min)
        self._veh_total_delay_collect() # collect for first step
        print(f'----- Env including {len(self.net_info.keys())} intersections is ready.')
        
        states = []
        for tls_id in self.net_info.keys():
            states.append(self._get_state(tls_id))
        states = np.stack(states)

        return states


    def close(self):
        traci.close()
        print("----- connection to sumo closed")


    def step(self, action):
        self.control_input = action # apply action to the signal

        self._update_env() # update the env
        
        self._veh_total_delay_collect() # collect the total delay of the vehicles
        states = []
        rewards = []
        dones = []
        infos = []
        for tls_id in self.net_info.keys():
            states.append(self._get_state(tls_id))
            rewards.append(self._get_reward(tls_id))
            dones.append(self._step >= self._max_steps-15)
            infos.append(self._get_info(tls_id))
        states = np.stack(states)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = np.stack(infos)

        return states, rewards, dones, infos


    def step_prepolicy(self):
        phases = {}
        for tls_id in self.net_info.keys():
            phases[tls_id] = traci.trafficlight.getPhase(tls_id)
        # debug
        # print(f'phases_{self._step}', phases)

        next_phases = self._update_env_prepolicy()
        self._veh_total_delay_collect() # collect the total delay of the vehicles
        # debug
        # print(f'next_phases_{self._step}', next_phases)

        states = []
        actions = []
        rewards = []
        dones = []
        infos = []
        for tls_id in self.net_info.keys():
            states.append(self._get_state(tls_id))
            actions.append(self._get_action(tls_id, phases[tls_id], next_phases[tls_id]))
            rewards.append(self._get_reward(tls_id))
            dones.append(self._step >= self._max_steps-15)
            infos.append(self._get_info(tls_id))
        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = np.stack(infos)

        return states, actions, rewards, dones, infos


    def _get_info(self, tls_id):
        return 0
        
        
    def _get_action(self, tls_id, phase, next_phase): 
        if next_phase == phase:
            action = 0
        else:
            action = 1
        # debug
        # print(f'action_{self._step}', action)
        return action
    
    def _get_state(self, tls_id):
        # The state is adopted the define in PressLight
        state = self._collect_observation_PL(tls_id)
        # MA_info = self._collect_MA_info(tls_id)
        # MP_info = self._collect_MP_info(tls_id)
        # state = MA_info + MP_info
        return state


    def _get_reward(self, tls_id):
        # reward = self._collect_waiting_veh_num(tls_id)
        reward = self._collect_pressure(tls_id)
        return -reward


    # state function candidates
    def _collect_observation_PL(self, tls_id):
        num_info_in = [0] * len(self.net_info[tls_id]['lanes_in']) * self.cell_num
        num_info_out = [0] * len(self.net_info[tls_id]['lanes_out'])
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            # All vehicles information can be observed
            lane_id, lane_pos = traci.vehicle.getLaneID(car_id), traci.vehicle.getLanePosition(car_id)
            if lane_id in (self.net_info[tls_id]['lanes_in']):
                lane_length = traci.lane.getLength(lane_id)
                # num info 
                lane_idx_1 = self.net_info[tls_id]['lanes_in'].index(lane_id)
                cell_idx = int(lane_pos / (lane_length / self.cell_num))
                num_info_in[lane_idx_1 * self.cell_num + cell_idx] += 1
            elif lane_id in (self.net_info[tls_id]['lanes_out']):
                lane_idx_2 = self.net_info[tls_id]['lanes_out'].index(lane_id)
                num_info_out[lane_idx_2] += 1
        current_phase = []
        for tls_id in self.net_info.keys():
            current_phase.append(traci.trafficlight.getPhase(tls_id))
        
        state = current_phase + num_info_in + num_info_out
        return state

    
    # reward function candidates
    def _collect_pressure(self, tls_id):
        pressure = 0
        # calculate the scaled total num of vehicles in incoming lanes
        for lane_id in self.net_info[tls_id]['lanes_in']:
            pressure += traci.lane.getLastStepVehicleNumber(lane_id) / (traci.lane.getLength(lane_id)/7) # 7 is the stop length of each vehicle
        # calculate the scaled total num of vehicles in outgoing lanes
        for lane_id in self.net_info[tls_id]['lanes_out']:
            pressure -= traci.lane.getLastStepVehicleNumber(lane_id) / (traci.lane.getLength(lane_id)/7)
        # absoulte value
        pressure = abs(pressure)
        return pressure
    
    
    def _collect_waiting_veh_num(self, tls_id):
        # calculate the num of vehicles which speed is less than 1m/s in incoming lanes
        waiting_veh_num = 0
        for lane_id in self.net_info[tls_id]['lanes_in']:
            for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                if traci.vehicle.getSpeed(veh_id) < 1:
                    waiting_veh_num += 1
        # scale the waiting_veh_num
        waiting_veh_num = waiting_veh_num / len(self.net_info[tls_id]['lanes_in'])
        return waiting_veh_num
    
    
    def _collect_incoming_veh_num(self, tls_id):
        # calculate the num of vehicles in incoming lanes
        incoming_veh_num = 0
        for lane_id in self.net_info[tls_id]['lanes_in']:
            incoming_veh_num += traci.lane.getLastStepVehicleNumber(lane_id) / (traci.lane.getLength(lane_id)/7) # 7 is the stop length of each vehicle
        # scale the incoming_veh_num
        incoming_veh_num = incoming_veh_num / len(self.net_info[tls_id]['lanes_in'])
        return incoming_veh_num
    
    
    def _collect_change_of_delay(self, tls_id):
        change_of_delay = max(self._veh_total_delay[-1] - self._veh_total_delay[-2], 0)
        change_of_delay = change_of_delay / 8 / (289.60 / 13.89)
        return change_of_delay
    


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
        lanes_all = list(set(lanes_in + lanes_out))
        lanes_in = list(set(lanes_in))
        lanes_out = list(set(lanes_out))
        lanes_connect = list(set(lanes_connect))
        # 排序
        lanes_all.sort()
        lanes_in.sort()
        lanes_out.sort()
        lanes_connect.sort()
        
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
        '''
        no policy in sumo, need to control the signal
        '''
        phases = {}
        switch_flag = {}
        for tls_id in self.net_info.keys():
            phases[tls_id] = traci.trafficlight.getPhase(tls_id)
            if self.control_input*2 != phases[tls_id]:
                switch_flag[tls_id] = 1
            else:
                switch_flag[tls_id] = 0
        
        # set the action to signal switch a large step when the phase is in yellow
        for tls_id in self.net_info.keys():
            if switch_flag[tls_id] == 1:
                self._set_next_phase(tls_id)
                self._simulate(self._yellow)
                self._set_control_phase(tls_id)
                self._simulate(self._green_min)
            else:
                # switch a large step when approaching the max green
                if traci.simulation.getTime() == self.phase_switch_pointer[tls_id] + self._green_max:
                    self._set_next_phase(tls_id)
                    self._simulate(self._yellow)
                    self._set_next_phase(tls_id)
                    self._simulate(self._green_min)
                else:
                    self._simulate(1)


    def _update_env_prepolicy(self):
        '''
        cannot update multiple signal lights like this, 
        because each signal light will be unsynchronized
        
        the actuated signal phase is presetted in the sumo, so just run no need control
        '''
        self._simulate(1)

        # get next phases for infering the action
        next_phases = {}
        for tls_id in self.net_info.keys():
            next_phases[tls_id] = traci.trafficlight.getPhase(tls_id)

        for tls_id in self.net_info.keys():
            # if yellow
            # switch a large step when the phase is in yellow
            if traci.trafficlight.getPhase(tls_id) % 2 == 1:
                self._simulate(self._yellow + self._green_min - 1) # minus 1
            # if green
            else:
                # switch a large step when approaching the max green
                if traci.simulation.getTime() == self.phase_switch_pointer[tls_id] + self._green_max:
                    self._simulate(self._yellow + self._green_min)

        return next_phases
    

    def _simulate(self, steps_todo):
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
                if next_phases[tls_id] != phases[tls_id]:
                    self.phase_switch_pointer[tls_id] = traci.simulation.getTime() -1
                    # print('time to change the pointer', traci.simulation.getTime())
                # print('phase_switch_pointer', self.phase_switch_pointer)
            
            # debug
            # print('self.phase_switch_pointer', self.phase_switch_pointer)
            # print('traci.simulation.getTime()', traci.simulation.getTime())
            # print('light', traci.trafficlight.getPhase('A0'))

            self._step += 1 # update the step counter
            steps_todo -= 1
            
            self._veh_delay_collect() # collect the delay info of the vehicles
            # self._veh_traj_collect() # collect the trajectory of the vehicles
            # self._veh_num_vs_light_collect() # collect the num of vehicles vs light
            
        return self._step
    
    
    def _veh_traj_collect(self):
        for veh_id in traci.vehicle.getIDList():
            edge = traci.vehicle.getRoadID(veh_id)
            pos = traci.vehicle.getLanePosition(veh_id)
            if edge in self.net_info['0']['edges_in'] and pos >= 200:
                self._veh_traj[edge].append((edge, veh_id, self._step, pos))
    

    def _veh_delay_collect(self):
        for veh_id in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(veh_id)
            if lane_id in self.net_info['0']['lanes_in']:
                travel_distance = traci.vehicle.getDistance(veh_id)
                depature_time = traci.vehicle.getDeparture(veh_id)
                self._veh_delay_info[veh_id] = max(traci.simulation.getTime() - depature_time - 1 - (travel_distance / 13.89), 0)
                
    
    def _veh_total_delay_collect_current(self):
        veh_delay = 0
        veh_id_list = traci.vehicle.getIDList()
        for veh_id in veh_id_list:
            veh_delay += self._veh_delay_info[veh_id]
        self._veh_total_delay.append(veh_delay)
        
        
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
        current_phase = traci.trafficlight.getPhase("0")
        next_phase = (current_phase + 1) % self.net_info[tls_id]['phase_num']
        # print(f'current phase: {current_phase}, next phase: {next_phase}')
        traci.trafficlight.setPhase(tls_id, next_phase)
        
        self.phase_switch_pointer[tls_id] = traci.simulation.getTime()
        # print('time to change the pointer', traci.simulation.getTime())
    
    def _set_control_phase(self,tls_id):
        next_phase = self.control_input*2
        traci.trafficlight.setPhase(tls_id, next_phase)
        self.phase_switch_pointer[tls_id] = traci.simulation.getTime()
    
    
    def _collect_policy_info(self, tls_id, phase):
        self._policy_dict[tls_id].append((self._step, phase))
        # if self.wandb:
        #     wandb.log({"phase": phase})
            # print(f'phase: {phase}')
            # wandb.log({"phase2": phase})
            # wandb.log({"phase3": phase})
            # wandb.log({"phase4": phase})
            
    def _veh_num_vs_light_collect(self):
        # retrieve the num of vehicles in each incoming edges and the current phase for each edge
        for edge in self.net_info['0']['edges_in']:
            edge_veh_num = traci.edge.getLastStepVehicleNumber(edge)
            phase = self._policy_dict['0'][-1][1]
            # 如果phase和self.net_info['0']['edge2phase'][edge]相同，说明这个边是绿灯，否则是红灯
            if phase == self.net_info['0']['lane2phase'][edge]:
                edge_phase = 1
            else:
                edge_phase = 0
            self.veh_num_vs_light[edge].append((edge_veh_num, edge_phase))
    
    
    # def save_veh_num_vs_light(self, episode, path):
    #     for edge in self.net_info['0']['edges_in']:
    #         df = pd.DataFrame(self.veh_num_vs_light[edge], columns=['veh_num', 'phase'])
    #         df.to_csv(os.path.join(path, f'veh_num_vs_light_{edge}_{episode}.csv'), index=False)
        # plot the veh_num_vs_light 
        # plot veh_num for light = 1 plot green, for light = 0 plot red
        # for edge in self.net_info['0']['edges_in']:
        #     df = pd.DataFrame(self.veh_num_vs_light[edge], columns=['veh_num', 'phase'])
        #     plt.figure(figsize=(10, 6))
        #     # 前一个信号灯状态，初始化为None
        #     prev_phase = None
            
        #     # 开始和结束索引，用于绘制同一状态的线段
        #     start_idx = 0
        #     for i in range(len(df)):
        #         # 如果信号灯状态改变，或者是最后一个数据点
        #         if df.loc[i, 'phase'] != prev_phase or i == len(df) - 1:
        #             if prev_phase is not None:
        #                 # 绘制上一个状态的线段
        #                 end_idx = i if df.loc[i, 'phase'] != prev_phase else i + 1
        #                 plt.plot(df.index[start_idx:end_idx], df['veh_num'][start_idx:end_idx], 'g-' if prev_phase == 1 else 'r-')
        #             # 更新开始索引和前一个状态
        #             start_idx = i
        #             prev_phase = df.loc[i, 'phase']
            
        #     plt.title(f'Vehicle Number vs Light for {edge}')
        #     plt.xlabel('Step')
        #     plt.ylabel('Vehicle Number')
        #     plt.savefig(os.path.join(path, f'veh_num_vs_light_{edge}_{episode}.png'))
        #     plt.close()
    
            

        
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
            
            # plot the distribution of phase duration from df_phase_duration
            # Get unique phases to define histogram colors consistently
            # phases = df_phase_duration['phase'].unique()
            # colors = plt.cm.jet(np.linspace(0, 1, len(phases)))

            # plt.figure(figsize=(10, 6))

            # # Plot histogram for each phase
            # for color, phase in zip(colors, phases):
            #     # Filter durations by phase
            #     durations = df_phase_duration[df_phase_duration['phase'] == phase]['duration']
            #     plt.hist(durations, bins='auto', color=color, alpha=0.7, label=f'Phase {phase}', density=True)

            # plt.title('Distribution of Durations by Phase')
            # plt.xlabel('Duration')
            # plt.ylabel('Density')
            # plt.legend()
            # # save the plot
            # file_name = os.path.join(path, f'phase_duration_{tls_id}_{episode}.png')
            # plt.savefig(file_name)
            # plt.close()
            
            
            
            
            
            

