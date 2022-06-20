import time
import numpy as np
import tensorflow.compat.v2 as tf2

import play_agent_DNN_base_5_2 as DNN_base
import game_verify_online_5_2 as game_verify
import play_guess_DNN_5_2 as guess
import DNN_network_utils_5_2 as DNN_net


class PlayAgentDNN_MC_q(DNN_base.PlayAgentDNNBase):
    #memory leak increase rate < 0.6G/hour (*27 CPU parallel)
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior'], fraud=5):
        super().__init__(learning_rate, epsilon=epsilon, gamma=gamma, net0_list=net0_list, name=name)

        self.filename_e = filename_e
        self.filename_t = filename_t
        self.fraud = fraud
        #self.net0_list_agent = net0_list #never used
        if self.net0_list != 0:
            print("PlayAgentDNN_MC: net0 id + init: ", id(net0_list), net0_list[0][0][0,1], net0_list[0][2][50,50], net0_list[0][4][126,2])
        
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
                input_size=4*54*2
            else:
                input_size=5*54
            input_shape=(input_size,)
            '''
            input_size, input_shape = DNN_net.reshape_DNN_network(self.fraud)
            output_size=54
            activation=tf2.nn.relu
            loss=tf2.losses.mse
            output_activation=None
            ###############
            # net=q(s,a)
            ###############
            self.qe_net = DNN_net.build_network(input_size, input_shape, hidden_layers, output_size,
                                                activation, loss, output_activation, learning_rate)
            self.primay_net = self.qe_net
            #qt_net not used yet in MC q
            self.qt_net = DNN_net.build_network(input_size, input_shape, hidden_layers, output_size,
                                                activation, loss, output_activation, learning_rate)
            self.secondary_net = self.qt_net
            self.first_net0_sync()
            '''
            if net0_list != 0 :
                print("PlayAgentDNN_MC: net0 id1 + init: ", id(net0_list), net0_list[0][0][0,1], net0_list[0][2][50,50], net0_list[0][4][126,2])

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
            self.qt_net.set_weights(self.qe_net.get_weights())
            
        return

    def decide(self, state3s_batch, available_mask_batch, train=True):
        batch_size = state3s_batch.shape[0]
        return self.decide_q(state3s_batch.reshape(batch_size,-1), available_mask_batch, train=train)
    
    def learning_multi_games(self):
        return
    
    def learning_single_game(self, state3s_batch, actions_batch, rewards_batch, behaviors_batch=0):  #TBD: don't know how to use 'behaviors' in NN, is it a factor to 'learning rate'?
        #input state3: shape: round(=11)-player*batch-info(5, 54), or (4, 54*2)
        #action, reward shape: round(=11)-player*batch-info(0). no change in this method
        #state3s_batch2 = state3s_batch.reshape(-1, self.fraud*54)
        if 4 == self.fraud:
            state3s_batch2 = state3s_batch.reshape(-1, 4*54*2)
            state3s_batch3 = state3s_batch.reshape(-1, 4, 54*2)
            state3s_batch3 = state3s_batch3[:,:,0:54]
        else:
            state3s_batch2 = state3s_batch.reshape(-1, 5*54)
            state3s_batch3 = state3s_batch.reshape(-1, 5, 54)
        bitmasks54 = np.where(state3s_batch3[:,0,:]>0, 1, 0)

 
        ### verify
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_9, state3s_batch, state3s_batch2, self.fraud, bitmasks54)
        history = self.learning_single_game_q(state3s_batch2, actions_batch, rewards_batch, bitmasks54, behaviors_batch=behaviors_batch)

        return history


    def save_models(self):
        super().save_model(self.qe_net, self.filename_e)
        super().save_model(self.qt_net, self.filename_t)



class PlayAgentDNN_MC_q_Behavior(PlayAgentDNN_MC_q):
    #TBD: any diff to epsilon = 1.0? how rho works in MC DNN q value?
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior'], fraud=5):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        
    def decide(self, state3s_batch, available_mask_batch, train=True):
        return self.decide_b_pre(state3s_batch, available_mask_batch, PlayAgentDNN_MC_q_Behavior, train=train)
    

class PlayAgentDNN_MC_q_Top3_base(PlayAgentDNN_MC_q):
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list)
    
    def build_top3_possibility(self):
        print("build_top3_possibility(): in virtual base. WRONG !!!")
        return 0 #virtual
    
    def decide(self, state3s_batch, available_mask_batch, train=True):
        candidates = 3 #3
        batch_size = state3s_batch.shape[0]

        targets, targets0 = self.decide_54_q(state3s_batch.reshape(batch_size, -1), available_mask_batch)
        actions_sorted_oindex = np.argsort(-targets, axis=1)  #(-):bigger -> smaller
        actions_max_oindex = actions_sorted_oindex[:,0:candidates]  # 3 would > available_len
        
        batch_n0 = np.arange(0,batch_size).reshape(1,-1)  #batch_size
        batch_n = np.repeat(batch_n0, candidates, axis=0).T #max available len
        
        actions_max_targets0 = targets[batch_n.reshape(-1), actions_max_oindex.reshape(-1)]
        actions_max_targets = actions_max_targets0.reshape(batch_size, -1)
        
        p_actions_max_targets = self.build_top3_possibility(actions_max_targets)  #virtual implementation. actions_max_targets may include -inf
        
        ##### verify
        available_lens0 = np.sum(available_mask_batch, axis=1)[:,np.newaxis]
        available_lens1 = np.hstack((available_lens0, np.array([candidates]*batch_size)[:,np.newaxis]))
        available_lens = np.min(available_lens1, axis=1) #[:,np.newaxis] #shape = (batch_size, 1)
        available_p_index0 = np.where(available_lens >= 2)
        available_p_index = available_p_index0[0]
        #available_p = p_actions_max_targets[available_p_index]
        c1 = p_actions_max_targets[available_p_index, available_lens[available_p_index]-1]
        c2 = p_actions_max_targets[available_p_index, available_lens[available_p_index]-2]
        c = (c2<c1)
        if c.any():
            print("Top3 decide: p[5] < p[6] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        action_oindex = []
        for oindex, p_action in zip(actions_max_oindex, p_actions_max_targets):
            try:
                action_oindex0 = np.random.choice(oindex, p=p_action)
            except ValueError: 
                print("PlayAgentDNN_MC_q_Top3_base: probabilities contain NaN. p_action: ", p_action, oindex)
                action_oindex0 = np.random.choice(oindex)
            action_oindex.append(action_oindex0)
        
        return np.array(action_oindex)

class PlayAgentDNN_MC_q_Top3_Softmax(PlayAgentDNN_MC_q_Top3_base):
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list)
        
    def build_top3_possibility(self, actions_max_targtes):
        #input matrix could inlcude float(-inf). but exp(-inf)=0, does not impact the softmax result
        exp_actions_max_targets = np.exp(actions_max_targtes)
        p_actions_max_targets = exp_actions_max_targets/np.sum(exp_actions_max_targets, axis=1)[:,np.newaxis]
        return p_actions_max_targets
    
class PlayAgentDNN_MC_q_Top3_Uniform(PlayAgentDNN_MC_q_Top3_base):
    #linear possibility. actually, select 2 from 3. the ppossible of 3rd = 0
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list)
        
    def build_top3_possibility(self, actions_max_targtes):
        #option 2: who is faster
        actions_max_targtes0 = np.where(actions_max_targtes==float("-inf"), float('inf'), actions_max_targtes)
        min_targtes = np.min(actions_max_targtes0, axis=1)[:,np.newaxis] #would optm. targets has sorted outside the method
        actions_max_targtes_positive0 = actions_max_targtes0 - min_targtes + 2.718281828459e-4 #TBD: plus a small 'e' to keep the last q(from 0)
        #-inf ==> 0 again after -min
        actions_max_targtes_positive = np.where(actions_max_targtes_positive0==float('inf'), 0, actions_max_targtes_positive0)
        p_actions_max_targets = actions_max_targtes_positive/np.sum(actions_max_targtes_positive, axis=1)[:,np.newaxis]
        return p_actions_max_targets




class PlayAgentDNN_MC_q_guess(PlayAgentDNN_MC_q):
    #TBD: any diff to epsilon = 1.0? how rho works in MC DNN q value?
    def __init__(self, hidden_layers=[[[64, 0.2],[16, 0.2]], [[64, 0.2],[16, 0.2]]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior', 'guess'], fraud=4):
        super().__init__(hidden_layers=hidden_layers[0], filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        
        if '' != filename_e: #in UT, the init() has no input parameters. then skip here.
            filename_e_items = filename_e.split('.') #'./results/play_e_2_3600.h5'. filename_e_items[0]=''
            filename_g = '.' + filename_e_items[1] + '_gd.' + filename_e_items[2]
            self.agent_guess = guess.PlayGuess_DNN(hidden_layers=hidden_layers[1], filename_g=filename_g, learning_rate=learning_rate, reload=reload)  #learning_amount=20*batch
            print("YDL guess made in DNN_MC_q: ", filename_g)


    def decide(self, state3s_batch, available_mask_batch, guess_env_callback, train=True):
        #state3s_batch shape = (batch-info) or (batch(=1)-info)
        batch = state3s_batch.shape[0]
        
        #input=(batch-info(5,54)). invoke integrity() out of guess() since guess() doesn't check the shape of input state
        unknown_cards = self.agent_guess.integrity(state3s_batch)
        player_card_lens = self.agent_guess.players_inhand_length(state3s_batch, guess_env_callback)
        
        state3s_batch1 = state3s_batch.reshape(batch, -1)
        #input=(batch-info(5*54)). output=(batch-info(3,54))
        guessed_cards = self.agent_guess.guess(state3s_batch1, unknown_cards, player_card_lens, guess_env_callback)
        
        #new_state3s_batch shape=(batch-info(4-108)). translate to fraud shape
        new_state3s_batch = self.agent_guess.new_state3_translate(state3s_batch, guessed_cards)

        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_20, state3s_batch, guessed_cards, new_state3s_batch)
        
        return super().decide(new_state3s_batch, available_mask_batch, train=train), new_state3s_batch

    
    def learning_single_round(self, state3s_batch, state3s_fraud_batch, discard_batch):
        # guess net learning only
        #state3s_batch shape = (batch-info(5-54)), state3s_fraud_batch=(batch0-info(4-108))
        history = self.agent_guess.guess_learning_single_round(state3s_batch, state3s_fraud_batch, discard_batch)
        return history

     
    def learning_single_game(self, state3s_batch, actions_batch, rewards_batch, behaviors_batch=0):
        history = super().learning_single_game(state3s_batch, actions_batch, rewards_batch, behaviors_batch=behaviors_batch) #combine state3s_batch_guess0[] and guessed_state3s[] to new_state3s_batch[]. it and rewards[] for fit(), the player/batch is match
        return history

    def save_models(self):
        super().save_models()
        self.agent_guess.save_model()




class PlayAgentDNN_MC_q_guess_Behavior(PlayAgentDNN_MC_q_guess):
    def __init__(self, hidden_layers=[[[64, 0.2],[16, 0.2]], [[64, 0.2],[16, 0.2]]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior', 'guess'], fraud=4):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        
    def decide(self, state3s_batch, available_mask_batch, guess_env_callback, train=True):
        #guess need a env_callback(). doesn't use decide_b_pre()
        batch_size = state3s_batch.shape[0]

        if True == train:
            action_oindex, action_b = self.decide_b(state3s_batch, available_mask_batch)

            #whatever 'behavior' or not, the new state3 should be updated. it is waste but algin to other agents behaviors
            unknown_cards = self.agent_guess.integrity(state3s_batch)
            player_card_lens = self.agent_guess.players_inhand_length(state3s_batch, guess_env_callback)
            state3s_batch1 = state3s_batch.reshape(batch_size, -1)
            guessed_cards = self.agent_guess.guess(state3s_batch1, unknown_cards, player_card_lens, guess_env_callback)
            new_state3s_batch = self.agent_guess.new_state3_translate(state3s_batch, guessed_cards)
        else:
            action_oindex, new_state3s_batch = super().decide(state3s_batch, available_mask_batch, guess_env_callback, train=False)
            action_b = np.array([1]*batch_size)
        
        return action_oindex, action_b, new_state3s_batch


#the only difference in agent is fraud=4 rather than 5
class PlayAgentDNN_MC_q_fraud(PlayAgentDNN_MC_q):
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['non-behavior', 'fraud'], fraud=4):
        super().__init__(hidden_layers, filename_e, filename_t, learning_rate, epsilon, gamma, reload, net0_list, name=name, fraud=fraud)
        return

    #add explict method with identical code: easy to understand the class object in debug
    def decide(self, state3s_fraud_batch, available_mask_batch, train=True):
        return super().decide(state3s_fraud_batch, available_mask_batch, train=train)
    
    def learning_multi_games(self):
        return super().learning_multi_games()
    
    def learning_single_game(self, state3s_fraud_batch, actions_batch, rewards_batch, behaviors_batch=0):
        #shape: round(=11)-player*batch-info
        return super().learning_single_game(state3s_fraud_batch, actions_batch, rewards_batch, behaviors_batch=behaviors_batch)

            

class PlayAgentDNN_MC_q_Behavior_fraud(PlayAgentDNN_MC_q_Behavior): #PlayAgentDNN_MC_q_fraud):
    #TBD: any diff to epsilon = 1.0? how rho works in MC DNN q value?
    def __init__(self, hidden_layers=[[512, 0.3], [128, 0.3]], filename_e='', filename_t='', learning_rate=0.00001, epsilon=0.2, gamma=0.2, reload=False, net0_list=0, name=['behavior', 'fraud'], fraud=4):
        super().__init__(hidden_layers=hidden_layers, filename_e=filename_e, filename_t=filename_t, learning_rate=learning_rate, epsilon=epsilon, gamma=gamma, reload=reload, net0_list=net0_list, name=name, fraud=fraud)
        
    #add explict method with identical code: easy to understand the class object in debug
    def decide(self, state3s_fraud_batch, available_mask_batch, train=True):
        return super().decide(state3s_fraud_batch, available_mask_batch, train=train)
        
        
       