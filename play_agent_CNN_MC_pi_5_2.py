import time
import numpy as np
import tensorflow.compat.v2 as tf2
from tensorflow import keras

import play_agent_CNN_base_5_2 as CNN_base
import play_agent_DNN_MC_pi_5_2 as DNN_pi_base
import game_verify_online_5_2 as game_verify
import CNN_network_utils_5_2 as CNN_net

class PlayAgentRes_MC_pi(CNN_base.PlayAgentCNN_base):
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [32], 'kernal_sizes':[3], 'strides':[(1,3)]}, 'residual_net':{'conv_filters':[[32, 32],]*2, 'kernal_sizes':[[2,2], [2,2]], 'strides':[[(1,1), (1,1)],]*2}, 'policy_net':{'conv_filter': 64, 'kernal_size':2, 'stride':(1,3)}, 'v_net':{'conv_filter': 64, 'kernal_size':3, 'stride':(1,3)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior'], fraud=5):

        CNN_base.PlayAgentCNN_base.__init__(self, learning_rate, epsilon=epsilon, gamma=gamma, net0_list=net0_list, name=name)

        self.filename_pi_v = filename_e
        self.filename_dummy = filename_t
        self.fraud = fraud
        #self.net0_list_agent = net0_list #never used
        if self.net0_list != 0:
            print("PlayAgentCNN_MC: net0 id + init: ", id(net0_list), net0_list[0][0][0,1], net0_list[0][2][50,50], net0_list[0][4][126,2])
        
        if ( reload == True ):
            self.policy_v_net = self.load_model(self.filename_pi_v)
            self.primay_net = self.policy_v_net
            self.policy_v_net.summary()
            self.first_net0_sync()  #PR#32 to be verified
        else:                
            '''
            if 4 == fraud:
                input_shape=(4, 54*2, 1)
            else:
                input_shape=(5, 54, 1)
            '''
            input_shape = CNN_net.reshape_CNN_network(self.fraud)
            input_net_config = hidden_layers['input_net']
            residual_net_config = hidden_layers['residual_net']
            policy_net_config = hidden_layers['policy_net']
            v_net_config = hidden_layers['v_net']

            ###############
            # net=q(s,a)
            ###############
            self.policy_v_net = CNN_net.build_Res_network(input_shape, input_net_config, residual_net_config, policy_net_config, v_net_config, learning_rate)
            self.primay_net = self.policy_v_net

            self.first_net0_sync()

            ###############
            # net=q(s,a)
            ###############
            '''
            if net0_list != 0 :
                print("PlayAgentCNN_MC: net0 id1 + init: ", id(net0_list), net0_list[0][0][0,1], net0_list[0][2][50,50], net0_list[0][4][126,2])

                if (7 == net0_list[0][0][0,1] and 3 == net0_list[0][2][50,50] and 7 == net0_list[0][4][126,2]):
                    #first access the shared net weights. after deepcopy, the value of the 2 weights must change to non-zero
                    print("PlayAgentDNN_MC: first copy to net0")
                    self.sync_acquire()
                    self.sync_local_to_mp_net0(self.qe_net, net0_list)
                    self.sync_release()
                else:
                    print("PlayAgentDNN_MC: second+ copy to local")
                    self.sync_acquire()
                    self.sync_mp_net0_to_local(net0_list, self.qe_net)
                    self.sync_release()
            ''' 
            #self.qt_net.set_weights(self.qe_net.get_weights())
            
        return


    def decide(self, state3s_batch, available_mask_batch, train=True):
        return self.decide_pi(state3s_batch[:,:,:,np.newaxis], available_mask_batch, train=train, piv_net=True)
    
    def learning_multi_games(self):
        return
    
    def learning_single_game(self, state3s_batch, actions_batch, rewards_batch, behaviors_batch=0):  #TBD: don't know how to use 'behaviors' in NN, is it a factor to 'learning rate'?
        #shape: round(=11)-player*batch-info(5,54) or (4,108)
        if 4 == self.fraud:
            state3s_batch2 = state3s_batch.reshape(-1, 4, 54*2, 1)
        else:
            state3s_batch2 = state3s_batch.reshape(-1, 5, 54, 1)

        ### verify
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_15, state3s_batch, state3s_batch2, self.fraud)
        history = self.learning_single_game_pi(state3s_batch2, actions_batch, rewards_batch, behaviors_batch=behaviors_batch, piv_net=True)

        return history


    def save_models(self):
        super().save_model(self.policy_v_net, self.filename_pi_v)


class PlayAgentCNN_MC_pi(PlayAgentRes_MC_pi):
    #idencial to Res_pi(). using config to isolate CNN and Res
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [32], 'kernal_sizes':[3], 'strides':[(1,3)]}, 'residual_net':{'conv_filters':[[32, 32],]*2, 'kernal_sizes':[[2,2], [2,2]], 'strides':[[(1,1), (1,1)],]*2}, 'policy_net':{'conv_filter': 64, 'kernal_size':2, 'stride':(1,3)}, 'v_net':{'conv_filter': 64, 'kernal_size':3, 'stride':(1,3)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior'], fraud=5):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, net0_list=net0_list, name=name, fraud=fraud)


class PlayAgentRes_MC_pi_Behavior(PlayAgentRes_MC_pi):
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [32], 'kernal_sizes':[3], 'strides':[(1,3)]}, 'residual_net':{'conv_filters':[[32, 32],]*2, 'kernal_sizes':[[2,2], [2,2]], 'strides':[[(1,1), (1,1)],]*2}, 'policy_net':{'conv_filter': 64, 'kernal_size':2, 'stride':(1,3)}, 'v_net':{'conv_filter': 64, 'kernal_size':3, 'stride':(1,3)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior'], fraud=5):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        
    def decide(self, state3s_batch, available_mask_batch, train=True):
        return self.decide_b_pre(state3s_batch, available_mask_batch, PlayAgentRes_MC_pi_Behavior, train=train)
        '''
        if True == train:
            action_oindex, action_b = self.decide_b(state3s_batch, available_mask_batch)
        else:
            batch_size = state3s_batch.shape[0]
            action_oindex = PlayAgentRes_MC_pi.decide(self, state3s_batch, available_mask_batch, train=False)
            action_b = np.array([1]*batch_size)
            
        return action_oindex, action_b
        '''    
    
    def learning_multi_games(self):
        return
    
    def learning_single_game(self, state3s_batch, actions_batch, rewards_batch, behaviors_batch=0):  #TBD: don't know how to use 'behaviors' in NN, is it a factor to 'learning rate'?
        history = super().learning_single_game(state3s_batch, actions_batch, rewards_batch, behaviors_batch=behaviors_batch)
        return history

class PlayAgentCNN_MC_pi_Behavior(PlayAgentRes_MC_pi_Behavior):
    #idencial to Res_pi_b(). using config to isolate CNN and Res
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [32], 'kernal_sizes':[3], 'strides':[(1,3)]}, 'residual_net':{'conv_filters':[[32, 32],]*2, 'kernal_sizes':[[2,2], [2,2]], 'strides':[[(1,1), (1,1)],]*2}, 'policy_net':{'conv_filter': 64, 'kernal_size':2, 'stride':(1,3)}, 'v_net':{'conv_filter': 64, 'kernal_size':3, 'stride':(1,3)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior'], fraud=5):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)

    def decide(self, state3s_batch, available_mask_batch, train=True):
        return  super().decide(state3s_batch, available_mask_batch, train=train)


class PlayAgentRes_MC_pi_acc(PlayAgentRes_MC_pi, DNN_pi_base.PlayAgentDNN_MC_pi_aac):
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [32], 'kernal_sizes':[3], 'strides':[(1,3)]}, 'residual_net':{'conv_filters':[[32, 32],]*2, 'kernal_sizes':[[2,2], [2,2]], 'strides':[[(1,1), (1,1)],]*2}, 'policy_net':{'conv_filter': 64, 'kernal_size':2, 'stride':(1,3)}, 'v_net':{'conv_filter': 64, 'kernal_size':3, 'stride':(1,3)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior', 'AC'], fraud=5):

        PlayAgentRes_MC_pi.__init__(self, hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        #super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)

        discount = [ gamma ** i for i in range(11)]
        self.discount = np.array(discount)

    def shapes(self, learning_states, learning_next_states):  #invoked by learning_single_game() from parent class
        s_shape = learning_states.shape
        n_shape = learning_next_states.shape
        if 0 == s_shape[0]:
            state_shape = (0,)
            next_state_shape = (0,)
        else:
            state_shape = (s_shape[1], s_shape[2], s_shape[3], 1)
            next_state_shape = (n_shape[1], n_shape[2], n_shape[3], 1)
        return state_shape, next_state_shape

    def learning_single_game(self, states_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=0):
        DNN_pi_base.PlayAgentDNN_MC_pi_aac.learning_single_game(self, states_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch=behaviors_batch, piv_net=True)


    def decide(self, state3s_batch, available_mask_batch, train=True):
        return PlayAgentRes_MC_pi.decide(self, state3s_batch, available_mask_batch, train=train)





class PlayAgentRes_MC_pi_acc_Behavior(PlayAgentRes_MC_pi_acc): #, PlayAgentRes_MC_pi_Behavior):
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [32], 'kernal_sizes':[3], 'strides':[(1,3)]}, 'residual_net':{'conv_filters':[[32, 32],]*2, 'kernal_sizes':[[2,2], [2,2]], 'strides':[[(1,1), (1,1)],]*2}, 'policy_net':{'conv_filter': 64, 'kernal_size':2, 'stride':(1,3)}, 'v_net':{'conv_filter': 64, 'kernal_size':3, 'stride':(1,3)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior', 'AC'], fraud=5):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        
    def decide(self, state3s_batch, available_mask_batch, train=True):
        return self.decide_b_pre(state3s_batch, available_mask_batch, PlayAgentRes_MC_pi_acc_Behavior, train=train)
        #return PlayAgentRes_MC_pi_Behavior.decide(self, state3s_batch, available_mask_batch, train=train)




#do nothing beside add 'fraud' to name. easy for debuging
class PlayAgentRes_MC_pi_fraud(PlayAgentRes_MC_pi):
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [32], 'kernal_sizes':[3], 'strides':[(1,3)]}, 'residual_net':{'conv_filters':[[32, 32],]*2, 'kernal_sizes':[[2,2], [2,2]], 'strides':[[(1,1), (1,1)],]*2}, 'policy_net':{'conv_filter': 64, 'kernal_size':2, 'stride':(1,3)}, 'v_net':{'conv_filter': 64, 'kernal_size':3, 'stride':(1,3)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior', 'fraud'], fraud=4):
        super().__init__(hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)

    def decide(self, state3s_batch, available_mask_batch, train=True):
        return super().decide(state3s_batch, available_mask_batch, train=train)
    
    def learning_multi_games(self):
        return
    
    def learning_single_game(self, state3s_batch, actions_batch, rewards_batch, behaviors_batch=0):  #TBD: don't know how to use 'behaviors' in NN, is it a factor to 'learning rate'?
        history = super().learning_single_game(state3s_batch, actions_batch, rewards_batch, behaviors_batch=behaviors_batch)
        return history


class PlayAgentRes_MC_pi_Behavior_fraud(PlayAgentRes_MC_pi_Behavior):  #PlayAgentRes_MC_pi_fraud, 
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [32], 'kernal_sizes':[3], 'strides':[(1,3)]}, 'residual_net':{'conv_filters':[[32, 32],]*2, 'kernal_sizes':[[2,2], [2,2]], 'strides':[[(1,1), (1,1)],]*2}, 'policy_net':{'conv_filter': 64, 'kernal_size':2, 'stride':(1,3)}, 'v_net':{'conv_filter': 64, 'kernal_size':3, 'stride':(1,3)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior', 'fraud'], fraud=4):
        super().__init__(hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        
    def decide(self, state3s_batch, available_mask_batch, train=True):
        return super().decide(state3s_batch, available_mask_batch, train=train)

    def learning_multi_games(self):
        return
    
    def learning_single_game(self, state3s_batch, actions_batch, rewards_batch, behaviors_batch=0):  #TBD: don't know how to use 'behaviors' in NN, is it a factor to 'learning rate'?
        history = super().learning_single_game(state3s_batch, actions_batch, rewards_batch, behaviors_batch=behaviors_batch)
        return history




