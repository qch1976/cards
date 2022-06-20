#v4.1 due to env 4.1 (reset and save optm)
#v5: due to env 5.0 parallel running optm. discard alg is only 'dump'
#    add GPU support
#v5.2: add 54 output with 54 headers 
 
import os
import time
import numpy as np
import pandas as pd
import random as rd
#import tensorflow.compat.v2 as tf
#from tensorflow import keras
#import tensorflow as tf1

from tensorflow import keras
import tensorflow as tf


#in root@CentOS: apply all CPUs(28 cores) at >60%: 
#in ute@Debian: apply all CPUs(28 cores) at >20%. does 'root' impact that? A: no, both root and ute take CPU >20%
#BTW, sudo or su, doesn't make the conda env(. block running python discard_testbench...) A: 'conda init bash' will add conda env to root rc
#CPU mode: 5.6h(GPU-133 28 CPUs with 60%+) vs. 36h(ute 1 CPU=100%)

#below 3 lines would make GPU 2-3% increase. but don't the efficiency is imprvoed or not. need meas the full time cost
tf.compat.v1.enable_eager_execution()
tf.compat.v1.enable_v2_behavior()
tf.compat.v1.enable_v2_tensorshape()

####################################################
# GPU enable NOTES:
# MUST: tf.compat.v1.keras.backend.set_session() at very beginning of py file, before any program code
# if move into GPU_prepare(), tf will report "no GPU device found' when tf init
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #实现卡号匹配
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5  #default = 100%(actual=~96%)
config.gpu_options.allow_growth = True  #按需分配
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
#TBD: tf.test.is_gpu_available(). in console, it found and created the GPU. not verify in .py

#from absl import flags
def GPU_prepare(gpu_id):
    #verify
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("YDL1: GPU: ", gpus)

    #######################
    # no meaningful code in GPU_prepare()
    #######################
    
    # it is late if perf below lines here. MUST run them before any program code !!
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #实现卡号匹配
    #config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.9
    #tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    return
    '''
    def get_gpu_name(valid_gpus):
        return [':'.join(gpu.name.split(':')[1:]) for gpu in valid_gpus]

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    tf.keras.backend.set_learning_phase(1)
    FLAGS = flags.FLAGS

    flags.DEFINE_multi_integer('gpus', default=[0,1], help="Specific gpu indexes to run") #ERR: alway assign twice in spyder(no clear the config in mem). in console of Centos, it works well. so does any console, I guess.
    flags_dict = FLAGS.flag_values_dict()
    print("flags_dict: ", flags_dict)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("YDL1: GPU: ", gpus, flags_dict['gpus'])
    if gpus:
        gpu_indexs = [int(gpu.name.split(':')[-1]) for gpu in gpus]
        print("YDL2: ", gpu_indexs)
        valid_gpu_indexs = list(filter(lambda gpu: gpu in flags_dict['gpus'],gpu_indexs))
        print("YDL3: ", valid_gpu_indexs)
        valid_gpus = [gpus[index] for index in valid_gpu_indexs]
        print("YDL4: ", valid_gpus)
        tf.config.experimental.set_visible_devices(valid_gpus, 'GPU')
        flags_dict['gpus'] = get_gpu_name(valid_gpus)
        print("YDL5: ", flags_dict['gpus'])
        print("YDL6: ", FLAGS['gpus'])
    '''


import sys, getopt
from threading import Thread
from multiprocessing import Process, Manager, Lock, Pool, Array, set_start_method
import psutil
import copy


import deal_cards_5_2 as dc
import cards_env_5_2 as ce
import discard_q_net_keras_5_2 as q_net

ticks_d_1_diff = 0
ticks_d_2_diff = 0
ticks_d_3_diff = 0
ticks_d_4_diff = 0
ticks_d_5_diff = 0
ticks_d_6_diff = 0
ticks_d_7_diff = 0
ticks_d_8_diff = 0
ticks_d_9_diff = 0
ticks_d_a_diff = 0


def test_MC_result_onego(env, agent, state2s, best_discards_oindexes, train=True):
    #best_discards_oindex = [16, 18, 34, 41, 42, 43]  #oindex
    #net only predict 1 card as best candidate in corss-entropy mode.
    #onego works only in mse mode. but uncompatible to auto-playing
    metrix = []

    interim_results, action0_index, action0 = agent.decide_onego(state2s)
    interim_results.sort(kind='mergesort', axis=1)
    batch_size = len(interim_results)

    covered_by_best = [(6-len(np.setdiff1d(interim_results[i], best_discards_oindexes[i], assume_unique=True))) for i in range(batch_size)]
    
    if False == train:
        covered_by_18 = [(8-len(np.setdiff1d(action0_index[i], env.oindex_full[i], assume_unique=True))) for i in range(len(action0_index))]
        metrix = [action0_index, covered_by_18, action0, np.array([255]*batch_size)[:,np.newaxis], [255]*batch_size, np.array([[255]*batch_size]).reshape(-1, 1), [255]*batch_size ]
        
    return interim_results, covered_by_best, metrix


def play_dump(agent, state2s_batch, best_discards_oindexes, rewards_batch, next_state2s_batch, dones_batch, behaviors_batch): #need onego as test and exam method
    history = agent.pre_learn_dump(state2s_batch, best_discards_oindexes, rewards_batch)  #replaced by 54 headers net
    return history

def keep_reset(env, batch_size, keep):
    bankers = np.array([dc.Players.SOUTH, dc.Players.SOUTH, dc.Players.SOUTH, dc.Players.SOUTH]*(int(batch_size/4+1))).reshape(-1)
    _, state2s, best_discards_oindexes, _, rewards0 = env.reset(bankers[:batch_size], keep=keep, render=False) #True)
    rewards = np.array([[rewards0]*batch_size]*6).reshape(-1,6).astype(int)
    '''
    self.state3s, self.state2s, self.best_discards_oindex, self.np_start_player_cards, self.reward_times_10    
    np_tmp = np.array(tmp)
    state2s = np.array(np_tmp[:,1].tolist()) #
    best_discards_oindexes = np.array(np_tmp[:,2].tolist())
    rewards = np.array([np_tmp[:,4]]*6).reshape(-1,6).astype(int)
    '''
    return state2s, best_discards_oindexes, rewards


def play_MC_alg(env, agent, learning_f, test_f, epoch=500, to_best=6, batch_size=1):
    global ticks_d_1_diff, ticks_d_2_diff, ticks_d_3_diff, ticks_d_4_diff, ticks_d_5_diff, ticks_d_9_diff, ticks_d_a_diff
    his_matrix = []
    covered_by_best = [0 for _ in range(batch_size)]
    interim_result = np.array([[0,0,0,0,0,0]]*batch_size)
    ydl_stop = 0
    
    for e in range(epoch): #epoch=inner_epoch

        ticks11 = time.time()
        state2s, best_discards_oindexes, rewards = keep_reset(env, batch_size, True)
        ticks21 = time.time()
        ticks_d_1_diff += ticks21 - ticks11


        ticks12 = time.time()
        state2s_batch = []
        actions_batch = []
        next_state2s_batch = []
        dones_batch = []
        behaviors_batch = []
        rewards_batch = []
        
        if learning_f == play_dump: #only support dump in v5.0
            #train by 'best' directly
            #.append([state2, best_discards_oindex, [reward]*6 ])
            state2s_batch = state2s
            actions_batch = best_discards_oindexes
            rewards_batch = rewards
        else:
            print("wrong learning func. only dump is available ", learning_f)

        ticks22 = time.time()
        ticks_d_2_diff += ticks22 - ticks12
        
        if e%200 == 1:
            ydl_stop = 1
        if e%1==0 or e >= (epoch-1):
            ticks13 = time.time()
            #history = learning_f(agent, state2s, best_discards_oindexes, rewards, behaviors)
            history = learning_f(agent, state2s_batch, actions_batch, rewards_batch, next_state2s_batch, dones_batch, behaviors_batch)

            #print(history.history)
            his_matrix.append(history.history["loss"])
            #his_matrix.append(history.history["acc"]) #ydl_measure"])

            ticks23 = time.time()
            ticks_d_3_diff += ticks23 - ticks13
            
            if to_best > 0 :
                ticks14 = time.time()
                state2s, best_discards_oindexes, _ = keep_reset(env, batch_size, True)
                ticks24 = time.time()
                ticks_d_4_diff += ticks24 - ticks14
                
                ticks15 = time.time()
                interim_result, covered_by_best, _ = test_f(env, agent, state2s, best_discards_oindexes)
                ticks25 = time.time()
                ticks_d_5_diff += ticks25 - ticks15
                ydl_t = time.time()  
                print(ydl_t, "inner epoch ", agent.pid, e, interim_result, covered_by_best)
                
                if covered_by_best >= [to_best]*batch_size :
                    break #e
    ydl_t = time.time()  
    #print(ydl_t, "episod steps to complance:", e, covered_by_best, to_best)
    return interim_result, e, covered_by_best, np.mean(np.array(his_matrix)), history.history["loss"][0]  #last just has 1 value




class TestBench:
    def __init__(self, parameters, reload0=False, net0_list=0, seed=13):
#0: id,  train-method,   test-method,   exam-method,  net-conf_pi,
#5: lr,  R(*10), to-best, batch, net_input_format, 
#10: separator, flash_t,  epsilon, net-conf_v,  gamma,
        
        self.para_id = parameters[0]
        self.train_f = parameters[1]
        self.test_f  = parameters[2]
        self.exam_f  = parameters[3]
        self.hidden_layers_1 = parameters[4]
        self.learning_rate   = parameters[5]
        self.reward_times = parameters[6]
        self.be_best      = parameters[7]
        self.batch_size   = parameters[8]
        self.net_input_format = parameters[9]
        self.flash_t      = parameters[11]
        self.epsilon      = parameters[12]
        self.hidden_layers_2 = parameters[13]
        self.gamma       = parameters[14]
        self.outer_epoch = parameters[15]
        self.inner_epoch = parameters[16]
        
        #"checkpoints/route-rnn-{}.ckpt".format(index)
        
        if self.train_f == play_dump:
            filename_1 = 'q_dump_e'
            filename_2 = 'q_dump_t'
        else:
            filename_1 = 'ydl_1'
            filename_2 = 'ydl_2'

        if self.test_f == test_MC_result_onego:
            filename_1 += '_t1'
            filename_2 += '_t1'
        else:
            filename_1 += '_ydl1'
            filename_2 += '_ydl1'

        if self.exam_f == test_MC_result_onego:
            filename_1 += '_e1_'
            filename_2 += '_e1_'
        else:
            filename_1 += '_ydl2_'
            filename_2 += '_ydl2_'

        filename_1 += str(self.para_id) + '.h5'
        filename_2 += str(self.para_id) + '.h5'

        if self.train_f in [play_dump]:
            self.envs = ce.PokerEnvironment_6_1(111, self.batch_size, False, seed, reward_times=self.reward_times, input_format=self.net_input_format)
            
            #self.agent = q_net.DiscardAgent_net6_Qmax2(self.hidden_layers_1, filename_1, filename_2, learning_rate=self.learning_rate, epsilon=self.epsilon, reload=reload0, flash_t=self.flash_t, net0_list=net0_list)
            self.agent = q_net.DiscardAgent_net6_Qmax2_54headers(self.hidden_layers_1, filename_1, filename_2, learning_rate=self.learning_rate, epsilon=self.epsilon, reload=reload0, flash_t=self.flash_t, net0_list=net0_list)
            #self.agent = q_net.DiscardAgent_net6_Qmax2_CNN(self.hidden_layers_1, filename_1, filename_2, learning_rate=self.learning_rate, epsilon=self.epsilon, reload=reload0, flash_t=self.flash_t, net0_list=net0_list)
        else:
            print("wrong train_f. only play_dump support in v5.0", self.train_f)
            return
            
        print("param id and ID env and agent ", self.para_id, id(self.envs), id(self.agent))

        
    def update_parameters(self, parameters):
        self.__init__(parameters)
        
    #partially fixed
    def debug_PR29(self):
        #doesn't change self.best_discards_oindexes. but self.cards_status, self.np_start_player_cards
        #save_best_discards_oindexes = copy.deepcopy(self.games.card_envs.best_discards_oindexes)
        save_cards_status = copy.deepcopy(self.envs.cards_status)
        #the reference from env.players_cards[] would be break!!
        save_np_start_player_cards = copy.deepcopy(self.envs.np_start_player_cards)
        save_allowed_discarded_score = copy.deepcopy(self.envs.allowed_discarded_score)
        '''
        #printed log
        best status = [0, 1, 15, 16, 17, 18]
        cardset =
        [[43  4  5  5  0  1  0  0]
         [47  4  9  0  0  1  0  0]
         [48  4 10 10  0  1  0  0]
         [30  3  5  5  0  1  0  0]
         [35  3 10 10  0  1  0  0]
         [26  3  1  0  0  1  0  0]
         [17  2  5  5  0  1  0  0]
         [22  2 10 10  0  1  0  0]
         [25  2 13 10  0  1  0  0]
         [ 4  1  5  5  1  1  0  0]
         [ 6  1  7  0  1  1  0  0]
         [ 7  1  8  0  1  1  0  0]
         [ 8  1  9  0  1  1  0  0]
         [ 9  1 10 10  1  1  0  0]
         [ 0  1  1  0  1  1  0  0]
         [40  4  2  0  1  1  0  0]
         [ 1  1  2  0  1  1  0  0]
         [52  5 14  0  1  1  0  0]]
        '''
        err_cardset = np.array(
                [[43,  4,  5,  5,  0,  1,  0,  0],
                 [47,  4,  9,  0,  0,  1,  0,  0],
                 [48,  4, 10, 10,  0,  1,  0,  0],
                 [30,  3,  5,  5,  0,  1,  0,  0],
                 [35,  3, 10, 10,  0,  1,  0,  0],
                 [26,  3,  1,  0,  0,  1,  0,  0],
                 [17,  2,  5,  5,  0,  1,  0,  0],
                 [22,  2, 10, 10,  0,  1,  0,  0],
                 [25,  2, 13, 10,  0,  1,  0,  0],
                 [ 4,  1,  5,  5,  1,  1,  0,  0],
                 [ 6,  1,  7,  0,  1,  1,  0,  0],
                 [ 7,  1,  8,  0,  1,  1,  0,  0],
                 [ 8,  1,  9,  0,  1,  1,  0,  0],
                 [ 9,  1, 10, 10,  1,  1,  0,  0],
                 [ 0,  1,  1,  0,  1,  1,  0,  0],
                 [40,  4,  2,  0,  1,  1,  0,  0],
                 [ 1,  1,  2,  0,  1,  1,  0,  0],
                 [52,  5, 14,  0,  1,  1,  0,  0]] )
        ydlenv = self.envs
        self.envs.cards_status[ce.PRIORITY_ID][0] = np.arange(18) #the cardset in log was ordered by priority already
        self.envs.np_start_player_cards[0] = err_cardset
        #self.envs.allowed_discarded_score[0] = 6
        player_importency, player_deviation, priorities = self.envs.create_cards_status_evaluation()
        self.envs.allowed_discarded_score = self.envs.create_allowed_score(player_importency, player_deviation)
        ydl = self.envs.create_best_discards()

        #self.games.card_envs.best_discards_oindexes = copy.deepcopy(save_best_discards_oindexes)
        self.envs.cards_status = copy.deepcopy(save_cards_status)
        self.envs.np_start_player_cards = copy.deepcopy(save_np_start_player_cards)
        self.envs.allowed_discarded_score = copy.deepcopy(save_allowed_discarded_score)
        
        pass
    
    def train(self, param_id, episodes=10, seed=13):
        global ticks_d_6_diff, ticks_d_7_diff, ticks_d_8_diff
        matrix = []
        matrix2 = []
        for e in range(self.outer_epoch):
            #np.random.seed(13)  #keep same random sequence in every epoch
            np.random.seed(seed)
            ydl_t = time.time()  
            print(ydl_t, "train outer epoch ", param_id, e)
            for i in range(0, episodes, self.batch_size): #episodes=self_test玩多少盘; inner_epoch=为达到best,同一盘重复
                ydl_t = time.time()
                if i%50000 == 0:
                    print(ydl_t, "episode ", param_id, i)
                ticks16 = time.time()
                best_discards_oindexes = [] #TBD, wrong
                _, best_discards_oindexes, _= keep_reset(self.envs, self.batch_size, False)
                
                #verify. debug PR#29 only, stop when formal run since it will change the card data in env
                #self.debug_PR29()
                
                #verify best oindex. fix issue#17
                np_best_discards_oindexes0 = np.array(best_discards_oindexes)
                ydl_shape = np_best_discards_oindexes0.shape
                np_best_discards_oindexes = np_best_discards_oindexes0.reshape(-1)
                ydl = dc.oindex_2_suit_name(np_best_discards_oindexes)
                
                ticks26 = time.time()
                ticks_d_6_diff += ticks26 - ticks16
                
                ticks17 = time.time()
                discard_cards, convergency, covered_by_best, his_mean, his_last = play_MC_alg(self.envs, self.agent, self.train_f, self.test_f, epoch=self.inner_epoch, to_best=self.be_best, batch_size=self.batch_size)

                ticks27 = time.time()
                ticks_d_7_diff += ticks27 - ticks17

                matrix.append([best_discards_oindexes, discard_cards, covered_by_best, convergency, his_mean, his_last])

            #################################
            # sampled test for every outer loop
            np.random.seed(seed)
            ticks18 = time.time()
            state2s, best_discards_oindexes, _ = keep_reset(self.envs, self.batch_size, False) #only apply first batch as sampling
            interim_result, covered_by_best, _ = self.test_f(self.envs, self.agent, state2s, best_discards_oindexes)
            ticks28 = time.time()
            ticks_d_8_diff += ticks28 - ticks18

            mean = np.mean(covered_by_best)
            print("train outer result: first batch episode ", param_id, self.agent.pid, e, "averag covered_by_best ", mean)
            matrix2.append([e, mean])

        self.agent.save_models()
        np_matrix = np.array(matrix)
        tmp = np.array(np_matrix[:, 2].tolist())
        print("train final result: paramid episode ", param_id, episodes, "averag covered_by_best ", np.mean(tmp))

        return matrix, matrix2

        
    def test(self, param_id, episodes=10):
        matrix = []
        for i in range(0, episodes, self.batch_size):
            #state2, best_discards_oindex, _, _ = self.env.reset(keep=False) #new card deal
            #[self.env[j].reset(keep=False) for j in range(self.batch_size)]
            state2s, best_discards_oindexes, rewards = keep_reset(self.envs, self.batch_size, False)
            discard_cards, covered_by_best, top6s = self.test_f(self.envs, self.agent, state2s, best_discards_oindexes, train=False)
            matrix.append([best_discards_oindexes, discard_cards, covered_by_best, top6s])
        np_matrix = np.array(matrix)
        tmp = np.array(np_matrix[:, 2].tolist())
        print("test result: paramid episode ", param_id, episodes, "averag best ", np.mean(tmp))
        return matrix

        
    def exam(self, param_id, episodes=1):
        matrix = []
        for i in range(0, episodes, self.batch_size):
            state2s, best_discards_oindexes, rewards = keep_reset(self.envs, self.batch_size, False)
            interim_result, covered_by_best, top6 = self.exam_f(self.envs, self.agent, state2s, best_discards_oindexes, train=False)
            matrix.append([best_discards_oindexes, interim_result, covered_by_best, top6])
        np_matrix = np.array(matrix)
        tmp = np.array(np_matrix[:, 2].tolist())
        print("exam result: paramid episode ", param_id, episodes, "averag best ", np.mean(tmp))
        return matrix


parameters_set=[]
#note1: net_input_format: 'not in-hand' MUST BE: <=0
#note2: G alg: batch MUST BE 1

########################################
# 1. net scale impact
#Q                    #id,   train-method,        test-method,          exam-method,            net-conf,                                                                                                                                   lr,         R(*10), to-best, batch,  net_input_format, separator, flash_t,  epsilon, NNNNNNNNNN,                 NNNNN, outer,   inner, self_test, exam_test
parameters_set.append([0,    play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128, 256], 'kernal_sizes':[2, 2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,1), (1,2), (1,1)]}},        0.0001,     5,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     1000,      10 ])

#Q                   #id,    train-method,        test-method,          exam-method,           net-conf,                               lr,       R(*10), to-best, batch,  net_input_format, separator, flash_t,  epsilon, NNNNNNNNNN,                 NNNNN, outer,   inner, self_test, exam_test
parameters_set.append([1,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[256, 0.2]],                           0.01,     5,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     2,       1,     1000,      200 ])
parameters_set.append([2,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.1,      5,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     2,       1,     1000,      10 ])
parameters_set.append([3,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.01,     5,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     2,       1,     100,       10 ])

###############
# BN and regular test
parameters_set.append([4,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     0,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([5,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    0,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([6,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.0001,   0,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([7,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     1,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([8,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    1,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([9,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.0001,   1,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([10,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     5,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([11,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    5,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([12,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.0001,   5,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([13,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     0,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([14,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    0,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([15,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.0001,   0,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([16,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     1,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([17,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    1,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([18,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.0001,   1,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([19,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     5,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([20,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    5,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([21,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.0001,   5,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])

#Q                    #id,   train-method,        test-method,          exam-method,            net-conf,                                                                                                                    lr,       R(*10), to-best, batch,  net_input_format, separator, flash_t,  epsilon, NNNNNNNNNN,                 NNNNN, outer,   inner, self_test, exam_test
parameters_set.append([30,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.01,     0,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([31,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.001,    0,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([32,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.0001,   0,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([33,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.01,     1,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([34,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.001,    1,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([35,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.0001,   1,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([36,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.01,     5,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([37,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.001,    5,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([38,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.0001,   5,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([39,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.01,     0,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([40,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.001,    0,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([41,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.0001,   0,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([42,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.01,     1,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([43,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.001,    1,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([44,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.0001,   1,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([45,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.01,     5,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([46,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.001,    5,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])
parameters_set.append([47,   play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128], 'kernal_sizes':[2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,2), (1,1)]}},        0.0001,   5,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     5,       1,     50000,      10 ])




##########################################################################################################################################
# 1. net size impact
# net params: ?0K, ?00K, ?M
# expactation: 1 convergency verlocity, 2 covered by best, 3 CPU
#                                                                                                                                                                          (dont be same)
#Q                    #id,     train-method,        test-method,          exam-method,           net-conf_pi,                            lr,       R(*10), to-best, batch, net_input_format, separator, flash_t,  epsilon, NNNNNNNNNN,                 NNNNN, outer,  inner, self_test, exam_test
parameters_set.append([106,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[256, 0.2]],                           0.01,     5,      0,       100,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     10000,     100 ])
parameters_set.append([107,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     5,      0,       100,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     10000,     100 ])
parameters_set.append([108,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.01,     5,      0,       100,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     10000,     100 ])


########################################
# 3. learning rate impact
# net params: depends on 1) net size
# expactation: 1 convergency verlocity, 2 covered by best, 3 CPU
# TBD: onego?
parameters_set.append([134,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.1,      5,      0,       100,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     10000,     100 ])
parameters_set.append([135,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    5,      0,       100,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     10000,     100 ])

########################################
# 4. input format impact
# net params: depends on 1) net size
# expactation: 1 convergency verlocity, 2 covered by best, 3 CPU, 4 loss?
# TBD: onego?, LR
#Q                    #id,     train-method,        test-method,          exam-method,           net-conf_pi,                            lr,       R(*10), to-best, batch, net_input_format, separator, flash_t,  epsilon, NNNNNNNNNN,                 NNNNN, outer,  inner, self_test, exam_test
parameters_set.append([154,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     5,      0,       100,   [0,  0, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     10000,     100 ])
parameters_set.append([155,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     5,      0,       100,   [0,  0, 0.5, 1],  '#',       False,    0.2,     [],                         0,     50,     1,     10000,     100 ])


########################################
# 5. reward impact
# net params: depends on 1) net size
# expactation: 1 convergency verlocity, 2 covered by best, 3 CPU
# TBD: onego? LR, onego
#Q                    #id,     train-method,        test-method,          exam-method,           net-conf_pi,                            lr,       R(*10), to-best, batch, net_input_format, separator, flash_t,  epsilon, NNNNNNNNNN,                 NNNNN, outer,  inner, self_test, exam_test
parameters_set.append([174,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     0,      0,       100,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     10000,     100 ])
parameters_set.append([175,    play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     1,      0,       100,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     10000,     100 ])

########################################
# long study
# outer = 20
#Q                    #id,     train-method,        test-method,          exam-method,           net-conf_pi,                            lr,       R(*10), to-best, batch, net_input_format, separator, flash_t,  epsilon, NNNNNNNNNN,                 NNNNN, outer,  inner, self_test, exam_test
parameters_set.append([1006,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    1,      0,       100,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     20,     1,     50000,     100 ]) #mt
parameters_set.append([1007,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    1,      0,       100,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     20,     1,     500000,    100 ]) #1t comparison

########################################
# long study
# outer = 1 with resume
#Q                    #id,     train-method,        test-method,          exam-method,           net-conf_pi,                            lr,       R(*10), to-best, batch, net_input_format, separator, flash_t,  epsilon, NNNNNNNNNN,                 NNNNN, outer,  inner, self_test, exam_test
parameters_set.append([1016,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    1,      0,       100,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     1,      1,     100000,     100 ]) #mt
parameters_set.append([1017,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    1,      0,       100,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     1,      1,     1000000,    100 ]) #1t comparison


                       
                       
                       
########################################
# re-study after PR#17. previous result would be wrong
# net scale
#Q                    #id,     train-method,        test-method,          exam-method,           net-conf_pi,                            lr,       R(*10), to-best, batch, net_input_format, separator, flash_t,  epsilon, NNNNNNNNNN,                 NNNNN, outer,  inner, self_test, exam_test
parameters_set.append([2000,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[256, 0.2]],                           0.1,      5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2001,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.1,      5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2002,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.1,      5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2003,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[256, 0.2]],                           0.01,     5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2004,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2005,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.01,     5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2006,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[256, 0.2]],                           0.001,    5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2007,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2008,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.001,    5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2009,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[256, 0.2]],                           0.0001,   5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2010,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.0001,   5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2011,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.0001,   5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])

parameters_set.append([2012,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[256, 0.2]],                           0.01,     1,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2013,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.01,     1,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2014,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.01,     1,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2015,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[256, 0.2]],                           0.001,    1,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2016,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    1,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2017,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.001,    1,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2018,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[256, 0.2]],                           0.0001,   1,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2019,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.0001,   1,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2020,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.0001,   1,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])

#larger net with smaller LR got better result. reward fmt has a little impact
parameters_set.append([2021,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.00001,  5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([20210,  play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.00001,  5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2022,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[2048, 0.2], [8192, 0.2],[1024, 0.2]], 0.0001,   5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2023,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[2048, 0.2], [8192, 0.2],[1024, 0.2]], 0.00001,  5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2024,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.00001,  1,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2025,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[2048, 0.2], [8192, 0.2],[1024, 0.2]], 0.0001,   1,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2026,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[2048, 0.2], [8192, 0.2],[1024, 0.2]], 0.00001,  1,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])

#reward=0. 5 is a bit better than 1. just try one case
parameters_set.append([2027,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.0001,   0,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2028,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.00001,  0,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2029,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[2048, 0.2], [8192, 0.2],[1024, 0.2]], 0.0001,   0,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2030,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[2048, 0.2], [8192, 0.2],[1024, 0.2]], 0.00001,  0,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
                       
#input format
parameters_set.append([2031,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    5,      0,       200,   [0,  0, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2032,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.001,    5,      0,       200,   [0,  0, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2033,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.0001,   5,      0,       200,   [0,  0, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2034,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.0001,   5,      0,       200,   [0,  0, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2035,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    5,      0,       200,   [0,  0, 0.5, 1],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2036,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.001,    5,      0,       200,   [0,  0, 0.5, 1],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2037,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.0001,   5,      0,       200,   [0,  0, 0.5, 1],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2038,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.0001,   5,      0,       200,   [0,  0, 0.5, 1],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2039,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.001,    5,      0,       200,   [0, -2, 2,   5],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2040,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.001,    5,      0,       200,   [0, -2, 2,   5],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2041,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.0001,   5,      0,       200,   [0, -2, 2,   5],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2042,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[4096, 0.2], [512, 0.2], [128, 0.2]],  0.0001,   5,      0,       200,   [0, -2, 2,   5],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])


#### DNN 54 heads for comparison
# code change NEEDED!!! : xx.build_network() BN=n or y, regular=n ; TestBench(), agent class
# config: same to 2010 and 2019
#Q                    #id,     train-method,        test-method,          exam-method,           net-conf_pi,                            lr,       R(*10), to-best, batch, net_input_format, separator, flash_t,  epsilon, NNNNNNNNNN,                 NNNNN, outer,  inner, self_test, exam_test
parameters_set.append([2043,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.00001,  5,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])
parameters_set.append([2044,   play_dump,           test_MC_result_onego, test_MC_result_onego,  [[1024, 0.2], [256, 0.2]],              0.00001,  1,      0,       200,   [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,     1,     500000,    1000 ])

#### CNN 1 head for comparison
# in code MUST: xx.build_network() BN=y, regular=y ; TestBench(), agent class
#Q                    #id,    train-method,        test-method,          exam-method,            net-conf,                                                                                                                                   lr,         R(*10), to-best, batch,  net_input_format, separator, flash_t,  epsilon, NNNNNNNNNN,                 NNNNN, outer,   inner, self_test, exam_test
parameters_set.append([2045,  play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128, 256], 'kernal_sizes':[2, 2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,1), (1,2), (1,1)]}},        0.0001,     5,      0,       100,    [0, -1, 1,   2],  '#',       False,    0.2,     [],                         0,     50,      1,     500000,    1000 ])
parameters_set.append([2046,  play_dump,           test_MC_result_onego, test_MC_result_onego,  {'input_net':{'conv_filters': [4, 16, 64, 128, 256], 'kernal_sizes':[2, 2, 2, 2, 2], 'strides':[(1,1), (1,2), (2,1), (1,2), (1,1)]}},        0.0001,     5,      0,       100,    [0, -1, 0.5, 1],  '#',       False,    0.2,     [],                         0,     50,      1,     500000,    1000 ])

                       
def save_csv(param_id, his, overwrite, perform_exam):
    if False == perform_exam:
        train_measures2 = his["train2"]  #smaller memory

        ###########################
        # train storing: matrix2 [e, mean(1st batch)]
        train2_to_df = train_measures2
        df_train2 = pd.DataFrame(train2_to_df, columns=['outer', 'covered mean'])

        train2_filename = 'train2-' + str(param_id) + '.csv'
        if True == overwrite:
            df_train2.to_csv(train2_filename, index=False)
        else:
            df_csv_train2 = pd.read_csv(train2_filename)
            df_csv_train2 = df_csv_train2.append(df_train2)
            df_csv_train2.to_csv(train2_filename, index=False)

    exam_measures2  = his["exam"]
    ###########################
    # exam storing
    np_exam_measures2 = np.array(exam_measures2)
    covereds = np.array(np_exam_measures2[:,2].tolist())
    covered = np.mean(covereds).reshape(1,1)
    df_exam = pd.DataFrame(covered, columns=['covered'])
    
    if True == perform_exam:
        #collect the final avg. here the 'overwrite' MUST be False
        exam_filename = 'exam3-' + str(param_id) + '.csv'
    else:
        exam_filename = 'exam2-' + str(param_id) + '.csv'
        
    if True == overwrite:
        df_exam.to_csv(exam_filename, index=False)
    else:
        try:
            df_csv_exam = pd.read_csv(exam_filename)
        except FileNotFoundError as e:
            #here the 'overwrite' MUST be False when first time to write the exam3-xx files
            print("save_csv(): FileNotFoundError ", e)
            print("save_csv(): created file anyway", exam_filename)
            df_exam.to_csv(exam_filename, index=False)
            return
        
        df_csv_exam = df_csv_exam.append(df_exam)
        df_csv_exam.to_csv(exam_filename, index=False)


def save_csv0(param_id, his, overwrite):
    train_measures  = his["train"]
    train_measures2 = his["train2"]  #smaller memory
    train_measures3 = his["train3"]  #CPU
    test_measures   = his["test"]
    exam_measures   = his["exam"]

    
    
    ###########################
    # train storing
    line = []
    train_to_df = []
    for train_meas in train_measures: #a batch
        batch_size = train_meas[0].shape[0]
        
        ydl1 = train_meas[0].tolist()
        ydl2 = train_meas[1].tolist()
        ydl3 = train_meas[2] #is list. .tolist()
        ydl4 = [train_meas[3]]*batch_size
        ydl5 = [train_meas[4]]*batch_size
        ydl6 = [train_meas[5]]*batch_size
        
        ydl7 = [ydl1, ydl2, ydl3, ydl4, ydl5, ydl6]
        ydl8 = np.array(ydl7)
        ydl10 = ydl8[0][:,np.newaxis]
        ydl11 = ydl8[1][:,np.newaxis]
        ydl12 = ydl8[2][:,np.newaxis]
        ydl13 = ydl8[3][:,np.newaxis]
        ydl14 = ydl8[4][:,np.newaxis]
        ydl15 = ydl8[5][:,np.newaxis]
        
        ydl9 = np.concatenate((ydl10, ydl11, ydl12, ydl13, ydl14, ydl15), axis=1)
        line = ydl9.tolist() #best, result, covered

        train_to_df.extend(line)
        line = []

    df_train = pd.DataFrame(train_to_df, columns=['expect best', 'last epoch', 'covered', 'convergency', 'loss mean', 'last loss'])
    #df_train = pd.DataFrame(train_measures, columns=['expect best', 'last epoch', 'covered', 'convergency', 'loss mean', 'last loss'])

    ###########################
    # train storing: matrix2 [e, mean(1st batch)]
    train2_to_df = train_measures2
    df_train2 = pd.DataFrame(train2_to_df, columns=['outer', 'covered mean'])
    
    ###########################
    # train storing: matrix2 [e, mean(1st batch)]
    train3_to_df = train_measures3
    df_train3 = pd.DataFrame(train3_to_df, columns=['CPU1', 'CPU2', 'CPU3', 'CPU4', 'CPU5', 'CPU6', 'CPU7', 'CPU8', 'CPU9', 'CPUa', 'PERF1', 'PERF2', 'PERF3', 'PERF4', 'PERF5', 'PERF6', 'PERF7', 'PERF8', 'PERF9', 'PERFa', 'STEP1', 'STEP2', 'STEP3', 'STEP4', 'STEP5', 'STEP6', 'STEP7', 'STEP8', 'STEP9', 'STEPa' ])

    
    ###########################
    # test storing
    line = []
    test_to_df = []
    for test_meas in test_measures: #a batch
        batch_size = test_meas[0].shape[0]
        
        ydl1 = test_meas[0].tolist()
        ydl2 = test_meas[1].tolist()
        ydl3 = test_meas[2] #is list. .tolist()
        ydl4 = [ydl1, ydl2, ydl3]
        ydl5 = np.array(ydl4)
        ydl6 = ydl5[0][:,np.newaxis]
        ydl7 = ydl5[1][:,np.newaxis]
        ydl8 = ydl5[2][:,np.newaxis]
        ydl9 = np.concatenate((ydl6,ydl7,ydl8), axis=1)
        line = ydl9.tolist() #best, result, covered
        

        test_meas_3 = test_meas[3]
        ydl1 = test_meas_3[0].tolist()
        ydl2 = test_meas_3[1] #.tolist()
        ydl3 = np.round(test_meas_3[2], 7)
        ydl3 = ydl3.tolist()
        if (test_meas_3[3].shape[1]) > 1: #[255] as dummy
            ydl4 = test_meas_3[3].tolist()
            ydl5 = test_meas_3[4] #.tolist()
            ydl6 = test_meas_3[6] #.tolist()
        else:
            ydl4 = [[[],[],[],[],[],[],[],[]]]*batch_size
            ydl5 = [[[],[],[],[],[],[],[],[]]]*batch_size
            ydl6 = [[[],[],[],[],[],[],[],[]]]*batch_size
            
        if (test_meas_3[5].shape[1]) > 1: #[255] as dummy
            ydl7 = np.round(test_meas_3[5], 7)
            ydl7 = ydl7.tolist()
        else:
            ydl7 = [[[],[],[],[],[],[],[],[]]]*batch_size
            
        ydl8 = [ydl1, ydl2, ydl3, ydl4, ydl5, ydl6, ydl7 ]
        ydl19 = np.array(ydl8)
        ydl10 = ydl19[0][:,np.newaxis]
        ydl11 = ydl19[1][:,np.newaxis]
        ydl12 = ydl19[2][:,np.newaxis]
        ydl13 = ydl19[3][:,np.newaxis]
        ydl14 = ydl19[4][:,np.newaxis]
        ydl15 = ydl19[5][:,np.newaxis]
        ydl16 = ydl19[6][:,np.newaxis]
        ydl17 = np.concatenate((ydl10, ydl11, ydl12, ydl13, ydl14, ydl15, ydl16), axis=1)
        ydl18 = np.concatenate((ydl9, ydl17), axis=1)
        line = ydl18.tolist() #best, result, covered

        test_to_df.extend(line)
        line = []
        
    df_test = pd.DataFrame(test_to_df, columns=['expect best', 'result', 'covered', '1st 8 cards', '1st covered 18', '1st 8 possibilty', 'last 8 cards', 'last cover 18', 'last cover 13', 'last 8 possibility'])

    ###########################
    # exam storing
    line = []
    exam_to_df = []
    for exam_meas in exam_measures:
        batch_size = exam_meas[0].shape[0]
        ydl1 = exam_meas[0].tolist()
        ydl2 = exam_meas[1].tolist()
        ydl3 = exam_meas[2] #is list. .tolist()
        ydl4 = [ydl1, ydl2, ydl3]
        ydl5 = np.array(ydl4)
        ydl6 = ydl5[0][:,np.newaxis]
        ydl7 = ydl5[1][:,np.newaxis]
        ydl8 = ydl5[2][:,np.newaxis]
        ydl9 = np.concatenate((ydl6,ydl7,ydl8), axis=1)
        line = ydl9.tolist() #best, result, covered

        exam_meas_3 = exam_meas[3]
        ydl1 = exam_meas_3[0].tolist()
        ydl2 = exam_meas_3[1] #.tolist()
        ydl3 = np.round(exam_meas_3[2], 7)
        ydl3 = ydl3.tolist()
        if (exam_meas_3[3].shape[1]) > 1: #[255] as dummy
            ydl4 = exam_meas_3[3].tolist()
            ydl5 = exam_meas_3[4] #.tolist()
            ydl6 = exam_meas_3[6] #.tolist()
        else:
            ydl4 = [[[],[],[],[],[],[],[],[]]]*batch_size
            ydl5 = [[[],[],[],[],[],[],[],[]]]*batch_size
            ydl6 = [[[],[],[],[],[],[],[],[]]]*batch_size
            
        if (exam_meas_3[5].shape[1]) > 1:  #[255] as dummy
            ydl7 = np.round(exam_meas_3[5], 7)
            ydl7 = ydl7.tolist()
        else:
            ydl7 = [[[],[],[],[],[],[],[],[]]]*batch_size
            
        ydl8 = [ydl1, ydl2, ydl3, ydl4, ydl5, ydl6, ydl7 ]
        ydl19 = np.array(ydl8)
        ydl10 = ydl19[0][:,np.newaxis]
        ydl11 = ydl19[1][:,np.newaxis]
        ydl12 = ydl19[2][:,np.newaxis]
        ydl13 = ydl19[3][:,np.newaxis]
        ydl14 = ydl19[4][:,np.newaxis]
        ydl15 = ydl19[5][:,np.newaxis]
        ydl16 = ydl19[6][:,np.newaxis]
        ydl17 = np.concatenate((ydl10, ydl11, ydl12, ydl13, ydl14, ydl15, ydl16), axis=1)
        ydl18 = np.concatenate((ydl9, ydl17), axis=1)
        line = ydl18.tolist() #best, result, covered

        exam_to_df.extend(line)
        line = []

    df_exam = pd.DataFrame(exam_to_df, columns=['expect best', 'result', 'covered', '1st 8 cards', '1st covered 18', '1st 8 possibilty', 'last 8 cards', 'last cover 18', 'last cover 13', 'last 8 possibility'])



    train_filename = 'train-' + str(param_id) + '.csv'
    train2_filename = 'train2-' + str(param_id) + '.csv'
    train3_filename = 'cpu-' + str(param_id) + '.csv'
    test_filename = 'test-' + str(param_id) + '.csv'
    exam_filename = 'exam-' + str(param_id) + '.csv'
    
    if True == overwrite:
        df_train.to_csv(train_filename, index=False)
        df_train2.to_csv(train2_filename, index=False)
        df_train3.to_csv(train3_filename, index=False)
        df_test.to_csv(test_filename, index=False)
        df_exam.to_csv(exam_filename, index=False)
    else:
        df_csv_train = pd.read_csv(train_filename)
        df_csv_train2 = pd.read_csv(train2_filename)
        df_csv_train3 = pd.read_csv(train3_filename)
        df_csv_test = pd.read_csv(test_filename)
        df_csv_exam = pd.read_csv(exam_filename)
        print("df_csv_t/t, df_csv_exam shape ", df_csv_test.shape, df_csv_exam.shape)
        df_csv_train = df_csv_train.append(df_train)
        df_csv_train2 = df_csv_train2.append(df_train2)
        df_csv_train3 = df_csv_train3.append(df_train3)
        df_csv_test = df_csv_test.append(df_test)
        df_csv_exam = df_csv_exam.append(df_exam)
        df_csv_train.to_csv(train_filename, index=False)
        df_csv_train2.to_csv(train2_filename, index=False)
        df_csv_train3.to_csv(train3_filename, index=False)
        df_csv_test.to_csv(test_filename, index=False)
        df_csv_exam.to_csv(exam_filename, index=False)

def common_h5(parameters, selected_p_set, reload, overwrite, net0_list=0, seed=13, perform_exam=False, enable_GPU=False):
        his = {}
        q_init_test_bench = TestBench(parameters, reload0=reload, net0_list=net0_list, seed=seed)
        print("ID q_init_test_bench ", parameters[0], id(q_init_test_bench))
        
        self_test = parameters[17] * (10 if True == enable_GPU else 1)  #not used in testbench
        exam_test = parameters[18]
        
        np.random.seed(seed)  #keep same random sequence
        if False == perform_exam:
            #rd.seed(seed) 
            matrix, matrix2 = q_init_test_bench.train(parameters[0], episodes=self_test, seed=seed)
            #ce v5.0 removes perf_xx
            #print("CPU cost1: ", parameters[0], ticks_d_1_diff, ticks_d_2_diff, ticks_d_3_diff, ticks_d_4_diff, ticks_d_5_diff)
            #print("CPU cost2: ", parameters[0], ticks_d_6_diff, ticks_d_7_diff, ticks_d_8_diff, ticks_d_9_diff, ticks_d_a_diff)
            #print("CPU cost3: ", parameters[0], ce.perf_diff_1, ce.perf_diff_2, ce.perf_diff_3, ce.perf_diff_4, ce.perf_diff_5)
            #print("CPU cost4: ", parameters[0], ce.perf_diff_6, ce.perf_diff_7, ce.perf_diff_8, ce.perf_diff_9, ce.perf_diff_a)
            #print("CPU cost5: ", parameters[0], ce.step2_diff_1, ce.step2_diff_2, ce.step2_diff_3, ce.step2_diff_4, ce.step2_diff_5)
            #print("CPU cost6: ", parameters[0], ce.step2_diff_6, ce.step2_diff_7, ce.step2_diff_8, ce.step2_diff_9, ce.step2_diff_a)
            his["train"] = matrix
            his["train2"] = matrix2
            his["train3"] = []
            '''
            [[ticks_d_1_diff, ticks_d_2_diff, ticks_d_3_diff, ticks_d_4_diff, ticks_d_5_diff,
                              ticks_d_6_diff, ticks_d_7_diff, ticks_d_8_diff, ticks_d_9_diff, ticks_d_a_diff,
                              ce.perf_diff_1, ce.perf_diff_2, ce.perf_diff_3, ce.perf_diff_4, ce.perf_diff_5,
                              ce.perf_diff_6, ce.perf_diff_7, ce.perf_diff_8, ce.perf_diff_9, ce.perf_diff_a,
                              ce.step2_diff_1, ce.step2_diff_2, ce.step2_diff_3, ce.step2_diff_4, ce.step2_diff_5,
                              ce.step2_diff_6, ce.step2_diff_7, ce.step2_diff_8, ce.step2_diff_9, ce.step2_diff_a]]
            '''
            np.random.seed(seed)  #keep same random sequence
            #rd.seed(seed) #test set=train set
            matrix = q_init_test_bench.test(parameters[0], episodes=self_test)
            his["test"] = matrix
    
        #if perform_exam == True, come to here directly
        #no re-seed() needed in exam. generate diff sequence
        np.random.seed(179)  #use diff random sequence
        #rd.seed(179) 
        matrix = q_init_test_bench.exam(parameters[0], episodes=exam_test)
        his["exam"] = matrix
        
        #measurements[parameters[0]] = his
        save_csv(parameters[0], his, overwrite, perform_exam)
        
       
#initial
def initial_h5(selected_p_set, net0_list=0, seed=13, perform_exam=False, enable_GPU=False):
    #global ce.perf_diff_1, ce.perf_diff_2, ce.perf_diff_3, ce.perf_diff_4, ce.perf_diff_5, ce.perf_diff_6, ce.perf_diff_7, ce.perf_diff_8, ce.perf_diff_9
    
    print("initial H5 creation\ninitial H5 creation\ninitial H5 creation\ninitial H5 creation\n perform_exam only: ", perform_exam)
    for i, parameters in enumerate(parameters_set):
        if parameters[0] not in selected_p_set:
            continue;
        #init: perform_exam MUST be False
        common_h5(parameters, selected_p_set, False, True, net0_list=net0_list, seed=seed, perform_exam=False, enable_GPU=enable_GPU)

#pause
#continuous
def resume_h5(selected_p_set, net0_list=0, seed=13, perform_exam=False, enable_GPU=False):
    print("resume H5 ... \nresume H5 ... \nresume H5 ... \nresume H5 ... \nresume H5 ... \n perform_exam only: ", perform_exam)
    #dont keep same sequence with init
    seed = int(np.random.random_sample() * 1000000) % 65535
    for i, parameters in enumerate(parameters_set):
        if parameters[0] not in selected_p_set:
            continue;
            
        common_h5(parameters, selected_p_set, True, False, net0_list=net0_list, seed=seed, perform_exam=perform_exam, enable_GPU=enable_GPU)
    

def run_child(h5_func_f,selected_p_set, cpu, net0_list, seed=13, perform_exam=False):
    proc = psutil.Process()  # get self pid
    print("PID: ", proc.pid)
    
    aff = proc.cpu_affinity()
    print("Affinity before: ", aff)
    
    proc.cpu_affinity(cpu)
    aff = proc.cpu_affinity()
    print("Affinity after:", aff)

    print("run_child: net0 id ", id(net0_list))
    print("run_child ", net0_list[0][0][0][0], net0_list[0][0][0][1], 
                        net0_list[0][2][50][50], net0_list[0][4][126][2])
    h5_func_f(selected_p_set, net0_list, seed, perform_exam=perform_exam)
    

def main(argv):
    print("argv ", argv)
    #spawn()
    #return

    multi_proces = 1
    cpu_back_start = 7
    seed_start = 13
    from_to = 1
    perform_exam = False
    enable_GPU = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   #it(environment) doesn't work in main(). it is so late. it takes effects at very beginning of the file before any code(such as main()) starts
    
    try:
        opts, args = getopt.getopt(argv,"r:m:c:s:p:g:t:")
    except getopt.GetoptError:
        print("wrong input")
        return;

    try:
        for opt, arg in opts:
            print("arg ",opt, arg)
            if opt == '-r':
                if arg == 'init' :
                    h5_func_f = initial_h5
                elif arg == 'resume' :
                    h5_func_f = resume_h5
                elif arg == 'exam' :
                    h5_func_f = resume_h5
                    perform_exam = True
                else:
                    print("wrong -r input", opt, arg)
                    return;

            if opt == '-m': #multi process: 0,1-27
                multi_proces = int(arg)

            if opt == '-c': #7 or 27
                cpu_back_start = int(arg)

            if opt == '-s':
                seed_start = int(arg)

            if opt == '-t':
                from_to = int(arg)
            
            if opt == '-p':
                selected_p_set0 = arg.split(',')
                selected_p_set = [int(c) for c in selected_p_set0]

            if opt == '-g':  #actually, -g doesn't work. enable/disable the GPU, have to do so before main during imported pkgs start up
                gpu_id = int(arg)
                GPU_prepare(gpu_id)  #only GPU-0 is supported
                enable_GPU = True
                
    except  ValueError:
            print("wrong input", opt, arg)
            return

    print("input set ", multi_proces, cpu_back_start, seed_start, selected_p_set, enable_GPU)
    n_cpus = psutil.cpu_count()  #0-27
    print("total CPUs ", n_cpus)
    
    if from_to > 1:
        selected_p_set_v = selected_p_set[0]
        for i in range(selected_p_set_v, selected_p_set_v + from_to,1):
            print("YDL: bunlde run: from ", selected_p_set_v, " now: ", i)
            h5_func_f([i], perform_exam=perform_exam, enable_GPU=enable_GPU)
        pass
    
    elif multi_proces > 1 :
        #'''
        #set_start_method('fork') #spawn')
        
        data_mgr = Manager()
        net0_lock1 = data_mgr.Lock() #需要整体lock,不是分项
        #net0_lock2 = data_mgr.Lock()
    
        #manual setup the shape
        net0_shape_1 = [(54,54), (54,), (54,256), (256,), (256,54), (54,)]
        net4_shape_1 = [(54,54), (54,), (54,1024), (1024,), (1024,256), (256,), (256,54), (54,)]
        net13_shape_1 = [(54,54), (54,), (54,1024), (1024,), (1024,256), (256,), (256,54), (54,)]
        net13_shape_2021 = [(54, 54), (54,), (54, 4096), (4096,), (4096, 512), (512,), (512, 128), (128,), (128, 54), (54,)]
        net13_shape_2022 = [(54, 54), (54,), (54, 4096), (4096,), (4096, 2048), (2048,), (2048, 512), (512,), (512, 54), (54,)]
        net13_shape_2023 = [(54, 54), (54,), (54, 4096), (4096,), (4096, 2048), (2048,), (2048, 512), (512,), (512, 54), (54,)]
    
        net0_shape_1 = net13_shape_2021 #net13_shape_1
        #net0_shape_2 = net13_shape_2
        ydl = [np.zeros(net0_shape_1[i]).tolist() for i in range(len(net0_shape_1))]
        
        net0_list1 = data_mgr.list()
        net0_list1.append(ydl)
        net0_list1.append(net0_lock1)
        net0_list1.append(net0_shape_1) #read only
        print("main: net0 id ", id(net0_list1))
        print("main ", net0_list1[0][0][0][0], net0_list1[0][0][0][1], 
                       net0_list1[0][2][50][50], net0_list1[0][4][126][2])
        
    
        procs = []
        #did not verify the multiple params in mp mode
        #cpu_back_start = 7 #27
        #seed_start = 13
        for i in range(multi_proces):
            cpu = [cpu_back_start-i]
            seed = seed_start+i
            p = Process(target=run_child, args=(h5_func_f, selected_p_set, cpu, net0_list1, seed, perform_exam, enable_GPU))
            p.start()
            procs.append(p)
            time.sleep(1)

        for p in procs:
            p.join()
            print('joined')
        #'''
    
    
        '''
        with Pool() as pool:
            # noinspection PyProtectedMember
            workers: int = pool._processes
            print(f"Running pool with {workers} workers")
    
            for i in range(workers):
                pool.apply_async(child, (i,))
    
            # Wait for children to finnish
            pool.close()
            pool.join()
            
        return
        '''
    
        ''' #multi thread. GIL lock makes the mt meaningless
        selected_p_set = [0]
        run = Thread(target=h5_func_f, args=(selected_p_set,))
        run.start()
        selected_p_set = [1]
        run = Thread(target=h5_func_f, args=(selected_p_set,))
        run.start()
        selected_p_set = [2]
        run = Thread(target=h5_func_f, args=(selected_p_set,))
        run.start()
        '''
    
        '''#concurrent futures. same CPU result with thread
        ex = futures.ThreadPoolExecutor(max_workers=3)
        model_types = [[0], [1], [2]]
        results = ex.map(h5_func_f, model_types) #返回值是generator 生成器    
        '''
    
        #multiple processes,. works in LINUX only
        ''' #process
        with Manager() as manager:
            lock=Lock()
            l = manager.list(range(5)) #l是共享对象。生成一个列表，在多个进程中传递和共享
            p_list = []    #存放进程对象
    
            selected_p_set = [0]
            p = Process(target=h5_func_f, args=(selected_p_set,))
            p.start()
            p_list.append(p)
    
            selected_p_set = [1]
            p = Process(target=h5_func_f, args=(selected_p_set,))
            p.start()
            p_list.append(p)
    
            selected_p_set = [2]
            p = Process(target=h5_func_f, args=(selected_p_set,))
            p.start()
            p_list.append(p)
            
            for res in p_list:
                res.join()  #??????
        '''
    
        ''' #pool
        num_workers = 3
        model_types = [[0], [1], [2]]
        pool = Pool(num_workers)
        
        pool.map(h5_func_f, model_types)
        '''
    else: #mp>1
        h5_func_f(selected_p_set, perform_exam=perform_exam, enable_GPU=enable_GPU)
    
   
    return

# -r init -m 3 -c 7 -p 4  #3 CPUs starting from CPU#7 backward
# -r init -p 13   #single CPU
# -r exam -p 7   #single CPU, set by taskset
# -r resume -g 0 -p 6   #enable GPU 0
# -r init -t 10 -p 13   #single CPU, from p13 -> p22, total 10 runs

if __name__ == "__main__":
    t0 = time.localtime()
    
    #measurements test tmp
    def summarize_csvs():
        p2_list = [2007, 2008, 2010, 2011, 2016, 2019,
                   2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039,
                   2040, 2041, 2042, 20210]
        p3_list = [2007, 2008, 2010, 2011, 2016, 2019,
                   2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 
                   2040, 2041, 2042, 20210]
        p2_len = len(p2_list)
        p3_len = len(p3_list)
        
        df_csv_covers2 = pd.DataFrame([[-1, -1, -1]], columns=['fileid', 'train2', 'exam2'])
        for file_id in p2_list:
            train2_filename = 'train2-' + str(file_id) + '.csv'
            exam2_filename = 'exam2-' + str(file_id) + '.csv'
            
            np_csv_train2 = pd.read_csv(train2_filename).values
            train2_len = len(np_csv_train2)
            np_file_ids = np.array([file_id]*train2_len)
            np_csv_train2_covered = np_csv_train2[:,1]
            NA = np.array([-1]*train2_len)
            np_csv_covered = np.concatenate((np_file_ids[:, np.newaxis], np_csv_train2_covered[:, np.newaxis], NA[:, np.newaxis]), axis=1)
            
            np_csv_exam2 = pd.read_csv(exam2_filename).values #shape(1,1)
            exam2_len = len(np_csv_exam2)
            np_file_ids = np.array([file_id]*exam2_len)
            NA = np.array([-1]*exam2_len)
            np_csv_exam = np.concatenate((np_file_ids[:, np.newaxis], NA[:, np.newaxis], np_csv_exam2), axis=1)
            
            df_csv_train = pd.DataFrame(np_csv_covered, columns=['fileid', 'train2', 'exam2'])
            df_csv_exam = pd.DataFrame(np_csv_exam, columns=['fileid', 'train2', 'exam2'])
            df_csv_covers2 = df_csv_covers2.append(df_csv_train, ignore_index=True)
            df_csv_covers2 = df_csv_covers2.append(df_csv_exam, ignore_index=True)
        
        df_csv_covers2 = df_csv_covers2.drop(0)
        filename = 'covers2_' + str(p2_list[0]) + '_' + str(p2_len) + '.csv'
        df_csv_covers2.to_csv(filename)


        df_csv_covers3 = pd.DataFrame([[-1, -1]], columns=['fileid', 'exam3 avg'])
        for file_id in p3_list:
            exam_filename = 'exam3-' + str(file_id) + '.csv'
            
            np_csv_exam3 = pd.read_csv(exam_filename).values
            np_exam3_mean = np.array([np.mean(np_csv_exam3)])
            np_file_ids = np.array([file_id])
            np_csv_exam = np.concatenate((np_file_ids[:, np.newaxis], np_exam3_mean[:, np.newaxis]), axis=1)
            
            df_csv_exam = pd.DataFrame(np_csv_exam, columns=['fileid', 'exam3 avg'])
            df_csv_covers3 = df_csv_covers3.append(df_csv_exam, ignore_index=True)

        df_csv_covers3 = df_csv_covers3.drop(0)
        filename = 'covers3_' + str(p3_list[0]) + '_' + str(p3_len) + '.csv'
        df_csv_covers3.to_csv(filename)

    
    #summarize_csvs()
    #measurements test tmp
    ydl_1 = time.time()
    ydl_2 = time.time()
    print(ydl_1, ydl_2)
    
    
    main(sys.argv[1:])
    print(time.strftime("%Y-%m-%d_%H-%M-%S", t0))
    print(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))