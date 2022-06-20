import time
import numpy as np
import tensorflow.compat.v2 as tf2
from tensorflow import keras

import play_agent_CNN_base_5_2 as CNN_base
import game_verify_online_5_2 as game_verify
import play_guess_CNN_5_2 as guess
import CNN_network_utils_5_2 as CNN_net

class PlayAgentCNN_MC_q(CNN_base.PlayAgentCNN_base):
    #has to add default to hidden_layers due to UT name checking
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [16, 32], 'kernal_sizes':[3, 2], 'strides':[(1,2), (1,2)]}, 'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior'], fraud=5, residual=False):
        
        CNN_base.PlayAgentCNN_base.__init__(self, learning_rate, epsilon=epsilon, gamma=gamma, net0_list=net0_list, name=name)

        self.filename_e = filename_e
        self.filename_t = filename_t
        self.fraud = fraud
        #self.net0_list_agent = net0_list #never used
        if self.net0_list != 0:
            print("PlayAgentCNN_MC: net0 id + init: ", id(net0_list), net0_list[0][0][0,1], net0_list[0][2][50,50], net0_list[0][4][126,2])
        
        if ( reload == True ):
            self.qe_net = self.load_model(filename_e)
            self.primay_net = self.qe_net
            self.qt_net = self.load_model(filename_t)
            self.secondary_net = self.qt_net
            self.qe_net.summary()
            self.qt_net.summary()
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
            output_net_config = hidden_layers['output_net']
            
            conv_filters_in = input_net_config['conv_filters']
            kernal_sizes_in = input_net_config['kernal_sizes']
            strides_in      = input_net_config['strides']
            
            conv_filter_out = output_net_config['conv_filter']
            kernal_size_out = output_net_config['kernal_size']
            stride_out      = output_net_config['stride']

            if True == residual:
                residual_net_config = hidden_layers['residual_net']
            else:
                residual_net_config = {}

            ###############
            # net=q(s,a)
            ###############
            self.qe_net = CNN_net.build_CNN_network(input_shape, 
                                                    conv_filters_in, kernal_sizes_in, strides_in, 
                                                    conv_filter_out, kernal_size_out, stride_out,
                                                    residual_net_config, learning_rate)
            self.primay_net = self.qe_net

            self.qt_net = CNN_net.build_CNN_network(input_shape,
                                                    conv_filters_in, kernal_sizes_in, strides_in, 
                                                    conv_filter_out, kernal_size_out, stride_out,
                                                    residual_net_config, learning_rate)
            self.qt_net.set_weights(self.qe_net.get_weights())
            self.secondary_net = self.qt_net

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
            
        return

    def decide(self, state3s_batch, available_mask_batch, train=True):
        return self.decide_q(state3s_batch[:,:,:,np.newaxis], available_mask_batch, train=train)
    
    def learning_multi_games(self):
        return
    
    def learning_single_game(self, state3s_batch, actions_batch, rewards_batch, behaviors_batch=0):  #TBD: don't know how to use 'behaviors' in NN, is it a factor to 'learning rate'?
        #shape: round(=11)-player*batch-info        
        #state3s_batch2 = state3s_batch.reshape(-1, 4, 54*2, 1)
        if 4 == self.fraud:
            state3s_batch2 = state3s_batch.reshape(-1, 4, 54*2, 1)
            state3s_batch3 = state3s_batch2[:,:,0:54,:].reshape(-1, 4, 54)
        else:
            state3s_batch2 = state3s_batch.reshape(-1, 5, 54, 1)
            state3s_batch3 = state3s_batch2.reshape(-1, 5, 54)
        bitmasks54 = np.where(state3s_batch3[:,0,:]>0, 1, 0)

        ### verify
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_16, state3s_batch, state3s_batch2, self.fraud, bitmasks54)
        history = self.learning_single_game_q(state3s_batch2, actions_batch, rewards_batch, bitmasks54, behaviors_batch=behaviors_batch)
        return history


    def save_models(self):
        super().save_model(self.qe_net, self.filename_e)
        super().save_model(self.qt_net, self.filename_t)


class PlayAgentRes_MC_q(PlayAgentCNN_MC_q):
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [32], 'kernal_sizes':[3], 'strides':[(1,3)]}, 'residual_net':{'conv_filters':[[32, 32],]*2, 'kernal_sizes':[[2,2], [2,2]], 'strides':[[(1,1), (1,1)],]*2}, 'output_net':{'conv_filter': 64, 'kernal_size':3, 'stride':(1,3)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior'], fraud=5):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud, residual=True)





class PlayAgentCNN_MC_q_Behavior(PlayAgentCNN_MC_q):
    #TBD: any diff to epsilon = 1.0? how rho works in MC DNN q value?
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [16, 32], 'kernal_sizes':[3, 2], 'strides':[(1,2), (1,2)]}, 'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior'], fraud=5, residual=False):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud, residual=residual)
        #PlayAgentCNN_MC_q
        
    def decide(self, state3s_batch, available_mask_batch, train=True):
        return self.decide_b_pre(state3s_batch, available_mask_batch, PlayAgentCNN_MC_q_Behavior, train=train)
        '''
        batch_size = state3s_batch.shape[0]
        
        if True == train:
            action_oindex, action_b = self.decide_b(state3s_batch, available_mask_batch)
        else:
            action_oindex = PlayAgentCNN_MC_q.decide(self, state3s_batch, available_mask_batch, train=False)
            action_b = np.array([1]*batch_size)
        return action_oindex, action_b
        '''
        
class PlayAgentRes_MC_q_Behavior(PlayAgentCNN_MC_q_Behavior):
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [32], 'kernal_sizes':[3], 'strides':[(1,3)]}, 'residual_net':{'conv_filters':[[32, 32],]*2, 'kernal_sizes':[[2,2], [2,2]], 'strides':[[(1,1), (1,1)],]*2}, 'output_net':{'conv_filter': 64, 'kernal_size':3, 'stride':(1,3)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior'], fraud=5):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud, residual=True)

    def decide(self, state3s_batch, available_mask_batch, train=True):
        return super().decide(state3s_batch, available_mask_batch, train=train)



class PlayAgentCNN_MC_q_guess(PlayAgentCNN_MC_q):
    #TBD: any diff to epsilon = 1.0? how rho works in MC DNN q value?
    def __init__(self, hidden_layers=[{'input_net':{'conv_filters': [16, 32], 'kernal_sizes':[3, 2], 'strides':[(1,2), (1,2)]}, 'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},
                                      {'input_net':{'conv_filters': [16, 32], 'kernal_sizes':[3, 2], 'strides':[(1,2), (1,2)]}, 'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}}],
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior', 'guess'], fraud=4, residual=False):
        super().__init__(hidden_layers=hidden_layers[0], filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud, residual=residual)

        if '' != filename_e: #in UT, the init() has no input parameters. then skip here.
            filename_e_items = filename_e.split('.') #'./results/play_e_2_3600.h5'. filename_e_items[0]=''
            filename_g = '.' + filename_e_items[1] + '_gc.' + filename_e_items[2]
            self.agent_guess = guess.PlayGuess_CNN(hidden_layers=hidden_layers[1], filename_g=filename_g, learning_rate=learning_rate, reload=reload) #learning_amount=10 for debug #default learning_amount=2000

    def decide(self, state3s_batch, available_mask_batch, guess_env_callback, train=True):
        #state3s_batch shape = (batch-info) or (batch(=1)-info)
        
        #input=(batch-info(5,54)). invoke integrity() out of guess() since guess() doesn't check the shape of input state
        unknown_cards = self.agent_guess.integrity(state3s_batch)
        player_card_lens = self.agent_guess.players_inhand_length(state3s_batch, guess_env_callback)
        
        #input=(batch-info(5,54,1)). output=(batch-info(3,54))
        state3s_batch1 = state3s_batch[:,:,:,np.newaxis]
        guessed_cards = self.agent_guess.guess(state3s_batch1, unknown_cards, player_card_lens, guess_env_callback)
        
        new_state3s_batch = self.agent_guess.new_state3_translate(state3s_batch, guessed_cards)

        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_20, state3s_batch, guessed_cards, new_state3s_batch)
        
        return super().decide(new_state3s_batch, available_mask_batch, train=train), new_state3s_batch
    
    def learning_single_round(self, state3s_batch, state3s_fraud_batch, discard_batch):
        # guess net learning only
        state3s_batch1 = state3s_batch[:,:,:,np.newaxis]
        # state3s_batch shape = (batch-info(5-54-1)), state3s_fraud_batch=(batch0-info(4-108))
        history = self.agent_guess.guess_learning_single_round(state3s_batch1, state3s_fraud_batch, discard_batch)
        return history
      
    def learning_single_game(self, state3s_batch, actions_batch, rewards_batch, behaviors_batch=0):
        history = super().learning_single_game(state3s_batch, actions_batch, rewards_batch, behaviors_batch=behaviors_batch) #combine state3s_batch_guess0[] and guessed_state3s[] to new_state3s_batch[]. it and rewards[] for fit(), the player/batch is match
        return history
    
    def learning_multi_games(self):
        #dump buffered records in guess
        history = self.agent_guess.flush_learning()
        return history
        
    def save_models(self):
        super().save_models()
        self.agent_guess.save_model()


class PlayAgentCNN_MC_q_guess_Behavior(PlayAgentCNN_MC_q_guess):
    def __init__(self, hidden_layers=[{'input_net':{'conv_filters': [16, 32], 'kernal_sizes':[3, 2], 'strides':[(1,2), (1,2)]}, 'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}},
                                      {'input_net':{'conv_filters': [16, 32], 'kernal_sizes':[3, 2], 'strides':[(1,2), (1,2)]}, 'output_net':{'conv_filter': 64,  'kernal_size':2, 'stride':(1,1)}}],
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior', 'guess'], fraud=4, residual=False):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        

    def decide(self, state3s_batch, available_mask_batch, guess_env_callback, train=True):
        batch_size = state3s_batch.shape[0]
        

        if True == train:
            action_oindex, action_b = self.decide_b(state3s_batch, available_mask_batch)

            #whatever 'behavior' or not, the new state3 should be updated.it is waste but algin to other agents behaviors
            unknown_cards = self.agent_guess.integrity(state3s_batch)
            player_card_lens = self.agent_guess.players_inhand_length(state3s_batch, guess_env_callback)
            state3s_batch1 = state3s_batch[:,:,:,np.newaxis]
            guessed_cards = self.agent_guess.guess(state3s_batch1, unknown_cards, player_card_lens, guess_env_callback)
            new_state3s_batch = self.agent_guess.new_state3_translate(state3s_batch, guessed_cards)

        else:
            action_oindex, new_state3s_batch = super().decide(state3s_batch, available_mask_batch, guess_env_callback, train=False)
            action_b = np.array([1]*batch_size)
        
        return action_oindex, action_b, new_state3s_batch


#do nothing beside add 'fraud' to name. easy for understanding class
class PlayAgentCNN_MC_q_fraud(PlayAgentCNN_MC_q):
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [16, 32], 'kernal_sizes':[3, 2], 'strides':[(1,2), (1,2)]}, 'output_net':{'conv_filter': 64, 'kernal_size':2, 'stride':(1,1)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior', 'fraud'], fraud=4):
        PlayAgentCNN_MC_q.__init__(self, hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        return

    def decide(self, state3s_batch, available_mask_batch, train=True):
        return super().decide(state3s_batch, available_mask_batch, train=train)
    
    def learning_multi_games(self):
        return
    
    def learning_single_game(self, state3s_batch, actions_batch, rewards_batch, behaviors_batch=0):  #TBD: don't know how to use 'behaviors' in NN, is it a factor to 'learning rate'?
        history = super().learning_single_game(state3s_batch, actions_batch, rewards_batch, behaviors_batch=behaviors_batch)
        return history



class PlayAgentCNN_MC_q_Behavior_fraud(PlayAgentCNN_MC_q_Behavior): #PlayAgentCNN_MC_q_fraud
    #TBD: any diff to epsilon = 1.0? how rho works in MC DNN q value?
    def __init__(self, hidden_layers={'input_net':{'conv_filters': [16, 32], 'kernal_sizes':[3, 2], 'strides':[(1,2), (1,2)]}, 'output_net':{'conv_filter': 64, 'kernal_size':2, 'stride':(1,1)}},
                 filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior', 'fraud'], fraud=4):
        super().__init__(hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)

        
    def decide(self, state3s_batch, available_mask_batch, train=True):
        return super().decide(state3s_batch, available_mask_batch, train=train)
