import time
import numpy as np
import tensorflow.compat.v2 as tf2
from tensorflow import keras

import play_agent_CNN_MC_q_5_2 as CNN_base
import play_agent_DNN_TD_q_5_2 as TD_base
import game_verify_online_5_2 as game_verify
from traj_replayer_5_2 import TrajectoryReplayer



class PlayAgentCNN_TD_Expected_q(CNN_base.PlayAgentCNN_MC_q, TD_base.PlayAgentDNN_TD_Expected_q):
    #memory leak increase rate < 0.6G/hour (*27 CPU parallel)
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [16, 32], 'kernal_sizes':[3, 2], 'strides':[(1,2), (1,2)]}, 'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior', 'TD'], fraud=5, residual=False):

        #print("PlayAgentCNN_TD_Expected_q: ", self.__class__.mro())        
        CNN_base.PlayAgentCNN_MC_q.__init__(self, hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, 
                                            net0_list=net0_list, name=name, fraud=fraud, residual=residual)

        self.replayer = TrajectoryReplayer(2e+8, (5,54), (5,54))  #2e+9 = 99G/27CPU=3.7G, 3.7G/2~=1.9G. set the limit to 2G
        self.qt_net_refresh_limit = 5
        self.qt_net_refresh_cnt = 0


    #no additional processing. easy for debug
    def decide(self, state3s_batch, available_mask_batch, train=True):
        return CNN_base.PlayAgentCNN_MC_q.decide(self, state3s_batch=state3s_batch, available_mask_batch=available_mask_batch, train=train)

    def learning_multi_games(self):
        TD_base.PlayAgentDNN_TD_Expected_q.learning_multi_games(self)
    
    def learning_single_game(self, state3s_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=0):  #TBD: don't know how to use 'behaviors' in NN, is it a factor to 'learning rate'?
        TD_base.PlayAgentDNN_TD_Expected_q.learning_single_game(self, state3s_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=behaviors_batch)

    def shapes(self, learning_states, learning_next_states):
        n_shape = learning_next_states.shape
        s_shape = learning_states.shape
        if 0 == s_shape[0]:
            next_state_shape = (0,)
            state_shape = (0,)
        else:
            next_state_shape = (n_shape[0], n_shape[1], n_shape[2], 1)
            state_shape = (s_shape[0], s_shape[1], s_shape[2], 1)
        return state_shape, next_state_shape
    
class PlayAgentCNN_TD_Expected_q_Behavior(PlayAgentCNN_TD_Expected_q): #, CNN_base.PlayAgentCNN_MC_q_Behavior):
    #TBD: any diff to epsilon = 1.0? how rho works in MC DNN q value?
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [16, 32], 'kernal_sizes':[3, 2], 'strides':[(1,2), (1,2)]}, 'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior', 'TD'], fraud=5):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        
    def decide(self, state3s_batch, available_mask_batch, train=True):
        return self.decide_b_pre(state3s_batch, available_mask_batch, PlayAgentCNN_TD_Expected_q_Behavior, train=train)
