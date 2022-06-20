#v5.0: split from game_5.py
#v5.1: suport fraud
#v5.2: guess added


import time
import copy
import numpy as np

import deal_cards_5_2 as dc
from game_verify_online_5_2 import CHKIDs

#######################################
# Limitations
# 1. build_decide_mask(): if NONE user added in leading(as first round), all cardsets must be NONE
#######################################

class PlayAgentProxy:
    #make some adaption for various agents, such as behavior, trajectory tranaltion, unified action IF, absorb env  ...
    def __init__(self, agent, checkpoints, guess_callback):
        #state3 = numpy.shape(5,54)
        #=inhand_oh[1,54] + played_oh[4,54]
        self.agent = agent
        self.checkpoints = checkpoints
        self.guess_callback = guess_callback
        #print("PlayAgentProxy init checkpoint id: ", id(checkpoints), checkpoints.batch_size)
        return
    
    def get_agent_state_shape(self): 
        if 'fraud' in self.agent.name:
            shape = (4, 54*2)
        elif 'guess' in self.agent.name:
            shape = (4, 54*2)
        else:
            shape = (5, 54)
        return shape
    
    
    def build_decide_mask(self, envs, state3s_batch, leading_trump_batch, leading_suit_batch):
        #reformat the input as np and combatible to scalar value
        np_state3s_batch = np.array(state3s_batch)
        if np_state3s_batch.shape == (5,54):
            batch_size = 1  # player by player, loop for batch
            #np_envs = np.array(envs).reshape(batch_size)
            np_state3s_batch = np_state3s_batch[np.newaxis,:]
            np_leading_trump_batch = np.array(leading_trump_batch).reshape(batch_size)
            np_leading_suit_batch = np.array(leading_suit_batch).reshape(batch_size)
        else: # all players, loop for batch
            #already been np array
            batch_size = np_state3s_batch.shape[0]
            #np_envs = np.array(envs)
            np_leading_trump_batch = np.array(leading_trump_batch)
            np_leading_suit_batch = np.array(leading_suit_batch)

        full_mask_batch = np.zeros((batch_size, 54), dtype=np.int)
        
        #check NONE suit, = first player in a round
        full_list = list(range(batch_size))
        none_list = np.where(np_leading_suit_batch == dc.CardSuits.NONE)[0]
        full_mask_batch[full_list] = (np_state3s_batch[full_list,0]>0)
        
        if len(full_list) == len(none_list):
            pass  #limitation: MUST be all are first player
        else:
            run_list = list(set(full_list) - set(none_list)) #limitation: MUST be full_list = run_list. otherwise=PR#34
            
            np_full_suits_batch = np.full((len(run_list), 54), dc.CardSuits.NONE)
            available_oindex0 = np.where(np_state3s_batch[run_list, 0]>0)
            available_suits_batch0 = np.array(dc.oindex_2_suit(available_oindex0[1])) 
            np_full_suits_batch[available_oindex0[0], available_oindex0[1]] = available_suits_batch0
            leading_suit_batch54 = np.repeat(np_leading_suit_batch.reshape(-1,1), 54, axis=1)
            np_full_suits_batch = (np_full_suits_batch[run_list]==leading_suit_batch54) #所有牌，和leading_suit相等

            fmt3_batch = np.array([envs.net_input_format[3]])
            fmt3_batch54 = np.repeat(fmt3_batch.reshape(-1,1), 54, axis=1)
            np_full_trumps_batch = (np_state3s_batch[run_list,0]==fmt3_batch54) #所有牌，是否为主
            
            np_full_suits_batch0 = np_full_suits_batch.astype(np.int) - np_full_trumps_batch.astype(np.int)
            np_full_suits_batch = np.where(np_full_suits_batch0>0, True, False) #true/false matrix. 除掉主2. full_suit里只有非主normal
            #print("proxy decide: leading suit+ input trump", np_leading_suit_batch, np_leading_trump_batch)
            #print("proxy decide: full suit+trump", np_full_suits_batch[run_list], np_full_trumps_batch[run_list])
            #print("proxy decide: state3[0]", np_state3s_batch[run_list,0])
            
            #select *54 bitmap from correct suit or trump
            c1 = (np_leading_trump_batch == fmt3_batch).reshape(-1,1)
            leading_normal_player = np.where(c1==False)[0]    #normal in leading

            c2 = np.any(np_full_trumps_batch, axis=1).reshape(-1,1)
            c3 = np.concatenate((c1, c2), axis=1)
            c4 = np.all(c3, axis=1)
            leading_trump_player_yes = np.where(c4==True)[0]  #trump in leading and who has trump 
            c5 = np.concatenate(((c4==False).reshape(-1,1),(c1==True).reshape(-1,1)),axis=1)
            c6 = np.all(c5==True, axis=1)
            leading_trump_player_no = np.where(c6==True)[0]  #trump in leading and who has NOT trump
            
            c7 = np.any(np_full_suits_batch[leading_normal_player], axis=1)
            leading_nomral_player_yes0 = np.where(c7==True)[0]  #normal in leading and who has leading suit 
            leading_normal_player_no0 = np.where(c7==False)[0]  #normal in leading and who has NOT leading suit
            leading_nomral_player_yes = leading_normal_player[leading_nomral_player_yes0]
            leading_normal_player_no = leading_normal_player[leading_normal_player_no0]
            
            #assign
            #list: leading_trump_player_yes <== np_full_trumps_batch
            full_mask_batch[leading_trump_player_yes] = np_full_trumps_batch[leading_trump_player_yes]
            full_mask_batch[leading_nomral_player_yes] = np_full_suits_batch[leading_nomral_player_yes]
            #list: leading_trump_player_no <== full_mask_batch[], already assigned
            #list: leading_normal_player_no <== full_mask_batch[], already assigned
            #print("proxy decide: full mask ", full_mask_batch)
            
            #verify
            _ = self.checkpoints.checkpoints_entry(CHKIDs.CHKID_2, leading_nomral_player_yes, leading_normal_player_no, leading_trump_player_yes, leading_trump_player_no, full_list)
            '''
            full_set = set(leading_nomral_player_yes) | set(leading_normal_player_no) | set(leading_trump_player_yes) | set(leading_trump_player_no)
            full_len = len(leading_nomral_player_yes) + len(leading_normal_player_no) + len(leading_trump_player_yes) + len(leading_trump_player_no)
            if full_set == set(full_list) and full_len == len(full_list):
                #print("proxy: decide: mask PASS !!!")
                pass
            else:
                print("proxy: decide: mask FAIL !!!")
            '''
        return full_mask_batch, np_state3s_batch, batch_size
        
    def decide(self, envs, state3s_batch, leading_trump_batch, leading_suit_batch, train=True, player_state3s_fraud=np.array([0])):
        _ = self.checkpoints.checkpoints_entry(CHKIDs.CHKID_3, state3s_batch, player_state3s_fraud)
        
        #for fraud only, all_player_state3s is extended here; np_state3s_batch[newaxis] in build_decide_mask()
        if 2 == len(player_state3s_fraud.shape):  #(4,54*2) or (n,4,54*2)
            player_state3s_fraud = player_state3s_fraud[np.newaxis,:,:] # => (1,4,54*2)
        
        ###########################
        # create batch*54 bitmaps(suit and trump) as available playing cards per player in a batch
        ###########################
        full_mask_batch, np_state3s_batch, batch_size = self.build_decide_mask(envs, state3s_batch, leading_trump_batch, leading_suit_batch)
        
        ###########################
        # input available mask to decide()
        ###########################
        if 'guess' in self.agent.name:  #input must not fraud
            if 'behavior' in self.agent.name:
                oindex_batch, b, fraud_state3s = self.agent.decide(np_state3s_batch, full_mask_batch, self.guess_callback, train=train)
            else:
                b = np.array([1]*batch_size) #b=dummy for behavior
                oindex_batch, fraud_state3s = self.agent.decide(np_state3s_batch, full_mask_batch, self.guess_callback, train=train)
            np_state3s_batch = fraud_state3s
        else:
            if 'fraud' in self.agent.name:
                np_state3s_batch = player_state3s_fraud
                
            if 'behavior' in self.agent.name:
                oindex_batch, b = self.agent.decide(np_state3s_batch, full_mask_batch, train=train)
            else:
                b = np.array([1]*batch_size) #b=dummy for behavior
                oindex_batch = self.agent.decide(np_state3s_batch, full_mask_batch, train=train)
            
        #suit_batch = dc.oindex_2_suit(oindex_batch)
        ##whatever fraud or not, below trump_batch is correct. shape=(n,5,54) or (n,4,54), dim[1]==0 is current player
        #trump_batch = np_state3s_batch[range(batch_size), 0, oindex_batch]

        return np_state3s_batch, oindex_batch, b

    def learning_single_game(self, state3s_batch, actions_batch, rewards_batch, behaviors_batch, next_states_batch, player_x=dc.Players.NONE):
        #proxy per player: such as S, E, N, W
        #shape: round(=11)-player*batch-info

        ### verify: state3s history
        _ = self.checkpoints.checkpoints_entry(CHKIDs.CHKID_8_3, state3s_batch, actions_batch, rewards_batch, player_x)

        if 'TD' in self.agent.name:
            self.agent.copy_qt()  #why need this? TBD
            if 'behavior' in self.agent.name:  #if 0 == behaviors_batch :
                self.agent.learning_single_game(state3s_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch)
            else:
                self.agent.learning_single_game(state3s_batch, actions_batch, rewards_batch, next_states_batch)
            return
        
        if 'AC' in self.agent.name:
            if 'behavior' in self.agent.name:  #if 0 == behaviors_batch :
                self.agent.learning_single_game(state3s_batch, actions_batch, rewards_batch, next_states_batch, behaviors_batch)
            else:
                self.agent.learning_single_game(state3s_batch, actions_batch, rewards_batch, next_states_batch)
            return
            
        if 'guess' in self.agent.name:  #if 0 == behaviors_batch :
            if 'behavior' in self.agent.name:  #if 0 == behaviors_batch :
                self.agent.learning_single_game(state3s_batch, actions_batch, rewards_batch, behaviors_batch)
            else:
                self.agent.learning_single_game(state3s_batch, actions_batch, rewards_batch)
        else:
            if 'behavior' in self.agent.name:  #if 0 == behaviors_batch :
                self.agent.learning_single_game(state3s_batch, actions_batch, rewards_batch, behaviors_batch)
            else:
                self.agent.learning_single_game(state3s_batch, actions_batch, rewards_batch)
        return 
    
    def learning_single_round(self, state3s_batch, state3s_fraud_batch, discard_batch=0):
        #shape=player-batch-info
        if 'guess' in self.agent.name:
            self.agent.learning_single_round(state3s_batch, state3s_fraud_batch, discard_batch)
        return

    def learning_multi_games(self):
        if 'TD' in self.agent.name:
            self.agent.learning_multi_games()
        if 'guess' in self.agent.name:
            self.agent.learning_multi_games()
        return 
