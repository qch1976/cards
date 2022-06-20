#v5.0, aglin to env v5.0

import os
import time
import numpy as np
import pandas as pd
import sys

import deal_cards_5_2 as dc

#################################
# GPU memory tracking
#################################
# on LINUX with GPU only
from pynvml import *
import platform

def is_Linux():
    '''
    判断当前运行平台
    '''
    sysstr = platform.system()
    #print("OS=", sysstr)
    if (sysstr == "Windows"):
        return False
    elif (sysstr == "Linux"):
        return True
    else:
        print ("Other System ")
    return False

def read_GPU_memory_usage(gpu_id):  #gpu_id=0
    gpu_mem_total = 0
    gpu_mem_used = 0
    if is_Linux():
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpu_id)
        memInfo = nvmlDeviceGetMemoryInfo(handle)
        mem_total = str(memInfo.total / 1024 / 1024) + ' MB'
        mem_used = str(memInfo.used / 1024 / 1024) + ' MB'
        mem_free = str(memInfo.total / 1024 / 1024 - memInfo.used / 1024 / 1024) + ' MB'
        
        #print(mem_total, mem_used, mem_free)
        gpu_mem_total, gpu_mem_used = mem_total, mem_used
    return gpu_mem_total, gpu_mem_used


#################################
# object memory counting
#################################
obj_counter=0
def dump(obj):
    for attr in dir(obj):#dir显示类的所有方法
        print(" obj.%s = %r" % (attr, getattr(obj, attr)))
  
def get_size(obj, seen=None, render=False):
    # From https://goshippo.com/blog/measure-real-size-any-python-object/
    # Recursively finds size of objects
    # doesn't work for NN net. takes long long time
    global obj_counter
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if render==True:
        print("obj", obj, obj_id, obj_counter)
    if obj_id in seen:
        return 0
    obj_counter += 1
# Important mark as seen *before* entering recursion to gracefully handle
# self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen, render=render) for v in obj.values()])
        size += sum([get_size(k, seen, render=render) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen, render=render)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen, render=render) for i in obj])
    return size


###################################
# agent measurements
###################################
class Measurements:
    def __init__(self, game_his):
        self.game_his = game_his   #used in local sub-processor, shape(games,2,batch)
        self.game_ids = []         #by this, read the local result record
        self.np_game_records = np.array([]) #used in main processor
        self.df_game_records = pd.DataFrame()
        self.np_agent_results = np.array([])
        self.df_agent_results = pd.DataFrame()

    def win_rate(self, param_set):
        if len(self.game_his) == 0:
            win_games_sn, win_games_ew, levels_raised_sn, levels_raised_ew = 0, 0, 0, 0
        else:
            np_winner_his0 = np.array(self.game_his) #, dtype='uint8') #shape(games, 2, n)
            np_winner_his = np_winner_his0.swapaxes(1,2).reshape(-1, np_winner_his0.shape[1])
        
            win_sn = np.where(np_winner_his[:,0]==dc.Players.SOUTH)[0]
            win_ew = np.where(np_winner_his[:,0]==dc.Players.EAST)[0]
            win_games_sn = len(win_sn)
            win_games_ew = len(win_ew)
            
            levels_raised_sn = np.sum(np_winner_his[win_sn,1]).astype(int)
            levels_raised_ew = np.sum(np_winner_his[win_ew,1]).astype(int)
            print("demo: win rate: agent#[S, N](", param_set.agent_class_s, param_set.agent_class_n, ")vs agent#[E, W](", param_set.agent_class_e, param_set.agent_class_w,"): ",  win_games_sn, win_games_ew)
            print("demo: win levels: agent[S, N] vs agent[E, W]: ",  levels_raised_sn, levels_raised_ew)
        
        return win_games_sn, win_games_ew, levels_raised_sn, levels_raised_ew

    def write_game_record(self, param_set, win_games_sn, win_games_ew, levels_raised_sn, levels_raised_ew):
        record_file_name = './results/game_record_' + str(param_set.game_id) + '.csv'
        f = open(record_file_name, 'w')  #use ASCII format for easily reading
        
        if win_games_sn > win_games_ew:
            winner = param_set.agent_class_s
        else:
            winner = param_set.agent_class_e
            
        record = str(param_set.agent_class_s) + ',' + str(param_set.agent_class_e) + ',' +   \
                 str(winner) + ',' + str(win_games_sn) + ',' + str(win_games_ew) + ',' +     \
                 str(levels_raised_sn) + ',' + str(levels_raised_ew)
        f.write(record)
        f.close()
        
    def assemble_records(self):
        time.sleep(2)

        game_records = []
        for file_id in self.game_ids:
            record_file_name = './results/game_record_' + str(file_id) + '.csv'
            f = open(record_file_name, 'r')
            line0 = f.readline()
            line = line0.split(',')
            one_record = []
            one_record.append(file_id)
            for char in line:
                one_record.append(int(char))
            game_records.append(one_record)
            f.close()
        
        self.np_game_records = np.array(game_records)
        self.df_game_records = pd.DataFrame(game_records, columns=('Game', 'SNID', 'EWID', 'WIN', 'SNWIN', 'EWWIN', 'SNLR', 'EWLR'))
        filename = './results/game_results_' + str(self.game_ids[0]) + '_' + str(len(self.game_ids)) + '.csv' 
        self.df_game_records.to_csv(filename, index=False)
        print("competition game result file ", filename)
        
        time.sleep(2)
        os.popen('rm ./results/game_record_*.csv')
    
    def add_game_id(self, game_id):    
        self.game_ids.append(game_id)
        
    def analyze_competition_result(self):
        agents_on_stage = self.np_game_records[:,0:2]
        agents = set(agents_on_stage.reshape(-1))
        winners = self.np_game_records[:,2]
        
        agent_results = []
        for agent in agents:
            agent_win = (winners==agent)
            agent_on_stage = (agents_on_stage == agent)
            wins = np.sum(agent_win)
            on_stages = np.sum(agent_on_stage)
            agent_results.append([agent, wins, on_stages, wins*100/on_stages])    
        
        np_agent_results = np.array(agent_results)
        results_sorted_index = np.argsort(-np_agent_results[:,1])
        self.np_agent_results = np_agent_results[results_sorted_index]
        self.df_agent_results = pd.DataFrame(self.np_agent_results, columns=('agent', 'wins', 'onstage', 'rate%'))
        
        filename = './results/agent_results_' + str(self.game_ids[0]) + '_' + str(len(self.game_ids)) + '.csv' 
        self.df_agent_results.to_csv(filename, index=False)
        
        print("competition result analysis file ", filename)
        
        
###################################
# game CPU performance
###################################
class PerformanceBase:
    def __init__(self):
        self.perf_diff_0 = 0
        self.perf_diff_1 = 0
        self.perf_diff_2 = 0
        self.perf_diff_3 = 0
        self.perf_diff_4 = 0
        self.perf_diff_5 = 0
        self.perf_diff_6 = 0
        self.perf_diff_7 = 0
        self.perf_diff_8 = 0
        self.perf_diff_9 = 0
        self.perf_diff_a = 0
        self.perf_diff_b = 0
        self.perf_diff_c = 0
        self.perf_diff_d = 0
        self.perf_diff_e = 0
        self.perf_diff_f = 0
        self.perf_diff_g = 0

    def perf_print(self):
        print("CPU 0-7: ", self.perf_diff_0, self.perf_diff_1, self.perf_diff_2, self.perf_diff_3, self.perf_diff_4, self.perf_diff_5, self.perf_diff_6, self.perf_diff_7)
        print("CPU 8-f: ", self.perf_diff_8, self.perf_diff_9, self.perf_diff_a, self.perf_diff_b, self.perf_diff_c, self.perf_diff_d, self.perf_diff_e, self.perf_diff_f)
        print("CPU g  : ", self.perf_diff_g)
        perfs = [self.perf_diff_0, self.perf_diff_1, self.perf_diff_2, self.perf_diff_3, self.perf_diff_4, self.perf_diff_5, self.perf_diff_6, self.perf_diff_7,
                 self.perf_diff_8, self.perf_diff_9, self.perf_diff_a, self.perf_diff_b, self.perf_diff_c, self.perf_diff_d, self.perf_diff_e, self.perf_diff_f,
                 self.perf_diff_g]
        return perfs

class PerformanceGame(PerformanceBase):
    def __init__(self):
        super().__init__()

    def perf_print(self, game_id):
        print("\nGAME performanes ", game_id)
        perfs= super().perf_print()
        print("game CPU3{play_one_round()} = CPU8+..+CPUc", self.perf_diff_8 + self.perf_diff_9 + self.perf_diff_a + self.perf_diff_b + self.perf_diff_c)
        print("game CPU8 is decide() biggest cost in training() and demo()")
        print("game CPU5 is single learning(), which is still smaller")
        perfs.insert(0, 'Game')
        return perfs
        
class PerformanceEnv(PerformanceBase):
    def __init__(self):
        super().__init__()

    def perf_print(self):
        print("\nENV performanes ")
        perfs = super().perf_print()
        print("env CPU 0-9 is reset() cost: ", self.perf_diff_0 + self.perf_diff_1 + self.perf_diff_2 + self.perf_diff_3 + self.perf_diff_4 + self.perf_diff_5 + self.perf_diff_6 + self.perf_diff_7 + self.perf_diff_8 + self.perf_diff_9)
        print("env CPU a is discard() biggest cost: ", self.perf_diff_a)
        perfs.insert(0, 'Env')
        return perfs

class PerformancePoker(PerformanceBase):
    def __init__(self):
        super().__init__()

    def perf_print(self):
        print("\nPOKER performanes ")
        perfs = super().perf_print()
        print("TBD: poker specific info added")
        perfs.insert(0, 'Poker')
        return perfs

class Performances:
    def __init__(self):
        self.game = PerformanceGame()
        self.env = PerformanceEnv()
        self.poker = PerformancePoker()
    
    def performance_report(self, game_id=-1):  #default as not writing into csv
        game_perf_baseline = self.game.perf_print(game_id)
        env_perf_baseline = self.env.perf_print()
        poker_perf_baseline = self.poker.perf_print()

        if -1 != game_id and game_id in [900, 901, 902, 903, 904, 905]:
            np_game_perf_baseline = np.array(game_perf_baseline)[np.newaxis,:]
            np_env_perf_baseline = np.array(env_perf_baseline)[np.newaxis,:]
            np_poker_perf_baseline = np.array(poker_perf_baseline)[np.newaxis,:]
            np_all_perf = np.concatenate((np_game_perf_baseline, np_env_perf_baseline, np_poker_perf_baseline), axis=0)
            df_all_perf = pd.DataFrame(np_all_perf, columns=['obj', 'CPU-0', 'CPU-1', 'CPU-2', 'CPU-3', 'CPU-4', 'CPU-5', 'CPU-6', 'CPU-7', 'CPU-8', 'CPU-9', 'CPU-a', 'CPU-b', 'CPU-c', 'CPU-d', 'CPU-e', 'CPU-f', 'CPU-g'])
            filename = 'perf_baseline_' + str(game_id) + '.csv'
            df_all_perf.to_csv(filename, index=False)
            
perfs = Performances()
#perfs.performance_report()