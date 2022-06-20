#v4.2: update for TD serise implementations
#v5.0: align to env v5.0
#v5.1: support fraud
#      reshape fraud to (4, 54*2), but online checking MISSED!
#v5.2  refactory state(state3, state3_fraud, and else)

import numpy as np
import copy
from enum import IntEnum

#import game_verify_CHKID_5_1 as CHKIDs

import deal_cards_5_2 as dc

####################################################
#  ONLINE verification
#
####################################################
class CHKIDs(IntEnum):
    CHKID_0       = 0
    CHKID_1       = 1
    CHKID_2       = 2
    CHKID_3       = 3
    CHKID_4       = 4
    CHKID_5       = 5
    CHKID_6       = 6
    CHKID_7_1     = 7
    CHKID_7_2     = 8
    CHKID_8_1     = 9
    CHKID_8_2     = 10
    CHKID_8_3     = 11
    CHKID_9       = 12
    CHKID_10_1    = 13
    CHKID_10_2    = 14
    CHKID_11      = 15
    CHKID_12      = 16
    CHKID_13      = 17
    CHKID_14      = 18
    CHKID_15      = 19
    CHKID_16      = 20
    CHKID_17      = 21
    CHKID_18      = 22
    CHKID_19      = 23
    CHKID_20      = 24
    CHKID_21      = 25
    CHKID_22      = 26
    CHKID_23      = 27
    CHKID_24      = 28
    CHKID_25      = 29
    CHKID_26      = 30


class Game54_Online_Checkpoints:
    def __init__(self, batch_size=0, disable=False):
        self.batch_size = batch_size
        #could disable some in checkpoints table
        #'''
        self.allow_checkpoints = [0,                  CHKIDs.CHKID_1,     CHKIDs.CHKID_2,      CHKIDs.CHKID_3,     CHKIDs.CHKID_4,      #
                                  CHKIDs.CHKID_5,     CHKIDs.CHKID_6,     CHKIDs.CHKID_7_1,    CHKIDs.CHKID_7_2,   CHKIDs.CHKID_8_1,    #
                                  CHKIDs.CHKID_8_2,   CHKIDs.CHKID_8_3,   CHKIDs.CHKID_9,      CHKIDs.CHKID_10_1,  CHKIDs.CHKID_10_2,   #
                                  CHKIDs.CHKID_11,    CHKIDs.CHKID_12,    CHKIDs.CHKID_13,     CHKIDs.CHKID_14,    CHKIDs.CHKID_15,     #
                                  CHKIDs.CHKID_16,    CHKIDs.CHKID_17,    CHKIDs.CHKID_18,     CHKIDs.CHKID_19,    CHKIDs.CHKID_20,     #
                                  CHKIDs.CHKID_21,    CHKIDs.CHKID_22,    CHKIDs.CHKID_23,     CHKIDs.CHKID_24,    CHKIDs.CHKID_25, 
                                  CHKIDs.CHKID_26 ]
        #'''
        if True == disable:
            self.allow_checkpoints = []
            
        self.check_entries = {CHKIDs.CHKID_1: self.check_state3s_fraud_rolling,
                              CHKIDs.CHKID_2: self.check_proxy_leading_bitmap,
                              CHKIDs.CHKID_3: self.check_proxy_input_state3_rolling,
                              CHKIDs.CHKID_4: self.check_round_reward,
                              CHKIDs.CHKID_5: self.check_round_infinity,
                              CHKIDs.CHKID_6: self.check_proxy_learning_single_fraud_state3,
                              CHKIDs.CHKID_7_1: self.check_state3s_in_round_old_state3,
                              CHKIDs.CHKID_7_2: self.check_state3s_in_round_next_state3,
                              CHKIDs.CHKID_8_1: self.check_state3s_history_in_1_round,
                              CHKIDs.CHKID_8_2: self.check_state3s_history_after_1_round,
                              CHKIDs.CHKID_8_3: self.check_state3s_history_proxy_learning_single,
                              CHKIDs.CHKID_9    : self.check_state3s_agent_MCq_state3_reshape,
                              CHKIDs.CHKID_10_1 : self.check_env_saver_store,
                              CHKIDs.CHKID_10_2 : self.check_env_saver_reload,
                              CHKIDs.CHKID_11   : self.check_state3s_agent_MCq_state3_G_A_reshape,
                              CHKIDs.CHKID_12   : self.check_state3s_agent_MCpi_state3_G_A_reshape,
                              CHKIDs.CHKID_13   : self.check_states_storage_reshape,                              
                              CHKIDs.CHKID_14   : self.check_state3s_vs_player_cards,
                              CHKIDs.CHKID_15   : self.check_state3s_agent_CNNq_state3_reshape,
                              CHKIDs.CHKID_16   : self.check_state3s_agent_CNNq_state3_reshape_with_bitmasks54,
                              CHKIDs.CHKID_17   : self.check_cards_integrity,
                              CHKIDs.CHKID_18   : self.check_state3s_reshape_in_guess,
                              CHKIDs.CHKID_19   : self.check_softmax,
                              CHKIDs.CHKID_20   : self.check_guess_format,
                              CHKIDs.CHKID_21   : self.check_learning_unique_action,
                              CHKIDs.CHKID_22   : self.check_reshape_game_single,
                              CHKIDs.CHKID_23   : self.check_guess_trump,
                              CHKIDs.CHKID_24   : self.check_guess_calc_discard,
                              CHKIDs.CHKID_25   : self.check_guess_calc_length,
                              CHKIDs.CHKID_26   : self.check_guess_calc_unknow_bits
                              }
        
        ###################################
        #  data buffered to be checked
        ###################################
        self.state3s_in_round_history = []  #CHK7
        self.state3s_history = {dc.Players.SOUTH: [],
                                dc.Players.EAST:  [],
                                dc.Players.NORTH: [],
                                dc.Players.WEST:  []  }     #CHK8
        self.actions_history = {dc.Players.SOUTH: [],
                                dc.Players.EAST:  [],
                                dc.Players.NORTH: [],
                                dc.Players.WEST:  []  }     #CHK8
        self.rewards_history = {dc.Players.SOUTH: [],
                                dc.Players.EAST:  [],
                                dc.Players.NORTH: [],
                                dc.Players.WEST:  []  }     #CHK8
        self.player_seq = []          #CHK8

        self.full_env_saver_cmp = []  #CHK10
        self.decide_state3 = []       #CHK13
        self.decide_player_x = []     #CHK13
        self.flag_4in1 = 0            #CHK13
        return

    def checkpoints_entry(self, check_id, *args):
        if check_id in self.allow_checkpoints:
            return self.check_entries[check_id](*args)
        else:
            return
        
    def check_state3s_fraud_rolling(self, np_all_player_state3s_fraud, players_x, card_envs_state3s):
        batch = len(players_x)
        env_fraud = np.concatenate((card_envs_state3s[:,:,0,:],  card_envs_state3s[:,:,1,:]), axis=2)
        
        c0 = (env_fraud[0, players_x[0]-1] == np_all_player_state3s_fraud[0, 0])
        c1 = (env_fraud[batch-1, players_x[batch-1]-1] == np_all_player_state3s_fraud[batch-1, 0])
        
        c2 = (env_fraud[0, (players_x[0]+2-1)%4] == np_all_player_state3s_fraud[0, 2])
        c3 = (env_fraud[batch-1, (players_x[batch-1]+3-1)%4] == np_all_player_state3s_fraud[batch-1, 3])
        c = c0.all() and c1.all() and c2.all() and c3.all()
        if not c:
            print("check_state3s_fraud_rolling failed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!. id=", CHKIDs.CHKID_1)
        
        return c

    def check_proxy_leading_bitmap(self, leading_nomral_player_yes, leading_normal_player_no, leading_trump_player_yes, leading_trump_player_no, full_list):
        full_set = set(leading_nomral_player_yes) | set(leading_normal_player_no) | set(leading_trump_player_yes) | set(leading_trump_player_no)
        full_len = len(leading_nomral_player_yes) + len(leading_normal_player_no) + len(leading_trump_player_yes) + len(leading_trump_player_no)
        c = (full_set == set(full_list) and full_len == len(full_list))
        if not c:
            print("check_proxy_leading_bitmap: mask FAIL !!!!!!!!!!!!!!!!!!!!!!!. id=", CHKIDs.CHKID_2)
        return c

    def check_proxy_input_state3_rolling(self, state3s_batch, all_player_state3s):
        shape = all_player_state3s.shape
        c = False
        if tuple([1]) == shape:
            c = True
        elif len(shape) == 2: #(4,54*2), batch=1
            c = (state3s_batch[0] == all_player_state3s[0, 0:54]).all()
        elif len(shape) > 2 and shape[0] == self.batch_size:  #(n, 4, 54*2)
            c = (state3s_batch[:,0] == all_player_state3s[:, 0, 0:54]).all()
        
        if not c:
            print("check_proxy_input_state3_rolling FAIL !!!!!!!!!!!!!!!!!!!!!!!. id=", CHKIDs.CHKID_3)
        return c
            
    def check_round_reward(self, rewards, reward_dict, player_sequences):
        pass

    def check_round_infinity(self, state_dict, action_dict, reward_dict, b_dict, next_state_dict):
        c = True
        for player in [dc.Players.SOUTH, dc.Players.EAST, dc.Players.NORTH, dc.Players.WEST]:
            c1 = np.where(state_dict[player] == -float('inf'))[0]
            c2 = np.where(action_dict[player] == -float('inf'))[0]
            c3 = np.where(reward_dict[player] == -float('inf'))[0]
            c4 = np.where(b_dict[player] == -float('inf'))[0]
            c5 = np.where(next_state_dict[player] == -float('inf'))[0]
            c = c and ((not c1) and (not c2) and (not c3) and (not c4) and (not c5))

        if not c:
            print("check_round_infinity FAIL !!!!!!!!!!!!!!!!!!!!!!!. id=", CHKIDs.CHKID_5)
        return c
            
    
    def check_proxy_learning_single_fraud_state3(self, state3s_batch_fraud, all_state3s):
        #shape state3s_batch_fraud: 11-4-batch-4-54*2
        #shape state3s_batch: 11-4-batch-5-54
        batch = all_state3s.shape[2]
        
        c1 = (state3s_batch_fraud[:, 2, 0, 0, 0:54] == all_state3s[:, 2, 0, 0])
        c2 = (state3s_batch_fraud[:, 2, 0, 3, 0:54] == all_state3s[:, 1, 0, 0])
        c10 = (state3s_batch_fraud[:, 2, 0, 0, 54:] == all_state3s[:, 2, 0, 1])
        c11 = (state3s_batch_fraud[:, 2, 0, 3, 54:] == all_state3s[:, 1, 0, 1])
        c9 = True
        if batch >= 1:
            c3 = (state3s_batch_fraud[:, 1, batch-1, 0, 0:54] == all_state3s[:, 1, batch-1, 0])
            c4 = (state3s_batch_fraud[:, 1, batch-1, 2, 0:54] == all_state3s[:, 3, batch-1, 0])
            c5 = (state3s_batch_fraud[:, 2, batch-1, 1, 0:54] == all_state3s[:, 3, batch-1, 0])
            c6 = (state3s_batch_fraud[:, 2, batch-1, 2, 0:54] == all_state3s[:, 0, batch-1, 0])
            c7 = (state3s_batch_fraud[:, 0, batch-1, 3, 0:54] == all_state3s[:, 3, batch-1, 0])
            c8 = (state3s_batch_fraud[:, 3, batch-1, 2, 0:54] == all_state3s[:, 1, batch-1, 0])
            c12 = (state3s_batch_fraud[:, 1, batch-1, 0, 54:] == all_state3s[:, 1, batch-1, 1])
            c13 = (state3s_batch_fraud[:, 1, batch-1, 2, 54:] == all_state3s[:, 3, batch-1, 1])
            c14 = (state3s_batch_fraud[:, 2, batch-1, 1, 54:] == all_state3s[:, 3, batch-1, 1])
            c15 = (state3s_batch_fraud[:, 2, batch-1, 2, 54:] == all_state3s[:, 0, batch-1, 1])
            c16 = (state3s_batch_fraud[:, 0, batch-1, 3, 54:] == all_state3s[:, 3, batch-1, 1])
            c17 = (state3s_batch_fraud[:, 3, batch-1, 2, 54:] == all_state3s[:, 1, batch-1, 1])
            
            c9 = (c3.all() and c4.all() and c5.all() and c6.all() and c7.all() and c8.all())
            c9 = c9 and (c12.all() and c13.all() and c14.all() and c15.all() and c16.all() and c17.all())
        
        c = c1.all() and c2.all() and c10.all() and c11.all() and c9
        if not c:
            print("check_proxy_learning_single_fraud_state3 FAIL!!!!!!!!!!!!!!!!!! id=", CHKIDs.CHKID_6)
        return c

    def check_state3s_in_round_old_state3(self, old_state3s, old_state3s_fraud):
        self.state3s_in_round_history = copy.deepcopy(old_state3s)
        self.state3s_fraud_in_round_history = copy.deepcopy(old_state3s_fraud)
        c = True
        return c

    def check_state3s_in_round_next_state3(self, actions, next_all_state3s, next_all_state3s_fraud, players_x):
        #state in one round. both new and old, the players seq is same
        old_state3s = self.state3s_in_round_history
        old_state3s_fraud = self.state3s_fraud_in_round_history
        batch = actions.shape[0]
        action_54 = np.eye(54)[actions]
        
        ######  state3s fraud ##########
        old_inhand_oindexs = old_state3s_fraud[np.arange(batch), 0, actions]
        old_played_oindexs = old_state3s_fraud[np.arange(batch), 0, actions+54]
        new_inhand_oindexs = next_all_state3s_fraud[np.arange(batch), 0, actions]
        new_played_oindexs = next_all_state3s_fraud[np.arange(batch), 0, actions+54]
        c6 = (old_inhand_oindexs > 0) 
        c7 = (old_played_oindexs == 0) 
        c8 = (new_inhand_oindexs == 0) 
        c9 = (new_played_oindexs > 0) 
        c10 = c6.all() and c7.all() and c8.all() and c9.all()
        
        ######  state3s ################
        #inhand
        old_inhand = old_state3s[np.arange(batch), players_x-1, 0, :]
        diff = old_inhand * action_54
        new_inhand = next_all_state3s[np.arange(batch), players_x-1, 0, :]
        c1 = ( (new_inhand + diff) == old_inhand )
        
        #played
        #shape=(n, 4, 5, 54): 如果S played， S应该出现在其他player，dim=（5）的第几个位置
                           #S  E  N  W
        offset = np.array([[1, 4, 3, 2],  #S played
                           [2, 1, 4, 3],  #E played
                           [3, 2, 1, 4],  #N played
                           [4, 3, 2, 1]   #W played
                          ])
        '''
          played  dim(5)		
                S	1	 S
                	2	 E
                	3	 N
                	4	 W
                		
                E	1	 E
                	2	 N
                	3	 W
                	4	 S
                		
                N	1	 N
                	2	 W
                	3	 S
                	4	 E
                		
                		
                W	1	 W
                	2	 S
                	3	 E
                	4	 N
        '''
        #played position
        cardset_4 = np.arange(batch)[:,np.newaxis].repeat(4, axis=1)
        players_n = np.arange(4)[np.newaxis, :].repeat(batch, axis=0)
        actions_4 = actions[:,np.newaxis].repeat(4, axis=1)
        inhand_index = offset[players_x - 1]
        old_played = old_state3s[cardset_4.reshape(-1), players_n.reshape(-1), inhand_index.reshape(-1)].reshape(batch, 4, 54)
        new_played = next_all_state3s[cardset_4.reshape(-1), players_n.reshape(-1), inhand_index.reshape(-1)].reshape(batch, 4, 54)
        diff_0 = new_played - old_played
        diff_1 = np.where(diff_0 != 0)
        c2 = (cardset_4.reshape(-1) == diff_1[0])
        c3 = (players_n.reshape(-1) == diff_1[1])
        c4 = (actions_4.reshape(-1) == diff_1[2])
        
        ''' # optmed as above. below is simple to understand
        old_played1 = []
        new_played1 = []
        for i, player in enumerate(players_x):
            old_played0 = next_state3s[i, np.arange(4), offset[player-1], :]
            old_played1.append(old_played0)

            new_played0 = next_state3s[i, np.arange(4), offset[player-1], :]
            new_played1.append(new_played0)
        
        old_played = np.array(old_played1)
        new_played = np.array(new_played1)
        '''
        #played fmt
        oindexes = old_state3s[np.arange(batch), players_x-1, 0, actions]
        oindexes_4 = oindexes[:, np.newaxis].repeat(4, axis=1)
        c5 = (diff_0[diff_1[0], diff_1[1], diff_1[2]] == oindexes_4.reshape(-1))
        
        c = c10 and c1.all() and c2.all() and c3.all() and c4.all() and c5.all()
        if not c:
            print("check_state3s_in_round_next_state3 FAIL!!!!!!!!!!!!!!!!!! id=", CHKIDs.CHKID_7_2)
        return c
        
    def check_state3s_history_in_1_round(self, state_dict, action_dict, reward_dict, player_sequences):
        #collect orignal state3s. shape=[n, 4, 5, 54]. 4 players complete 1 round
        for player in [dc.Players.SOUTH, dc.Players.EAST, dc.Players.NORTH, dc.Players.WEST]:
            self.state3s_history[player].append(state_dict[player])
            self.actions_history[player].append(action_dict[player])
            self.rewards_history[player].append(reward_dict[player])
        self.player_seq.append(player_sequences)
        c = True
        return c
        
    def check_state3s_history_after_1_round(self, game_state3s, rounds):
        #some additional info collected
        c = True
        return c

    def check_state3s_history_proxy_learning_single(self, state3s, actions, rewards, player_x):
        #state3s shape=(rounds(11)-player*batch-info)
        game_state3s = []
        game_actions = []
        game_rewards = []
        c3 = np.array([])
        
        info_shape = self.state3s_history[dc.Players.SOUTH][0].shape #whatever a player and a round
        
        if player_x == dc.Players.NONE:
            #4in1. shape = (11-4-n-info)
            for round_id in range(11):
                game_state3s_r = []
                game_actions_r = []
                game_rewards_r = []
                for player in [dc.Players.SOUTH, dc.Players.EAST, dc.Players.NORTH, dc.Players.WEST]:
                    #state3s_history[player_x], shape=(11(list)-1-n-info)
                    game_state3s_r.append(np.squeeze(np.array(self.state3s_history[player][round_id]), axis=0))
                    game_actions_r.append(np.squeeze(np.array(self.actions_history[player][round_id]), axis=0))
                    game_rewards_r.append(np.squeeze(np.array(self.rewards_history[player][round_id]), axis=0))
                
                #list shape=(4-n-info)
                game_state3s.append(game_state3s_r)
                game_actions.append(game_actions_r)
                game_rewards.append(game_rewards_r)
            
            #list shape=(11-4-n-info) reshape to (11-4*n-info)
            game_state3s = np.array(game_state3s).reshape((11, -1, info_shape[2], info_shape[3]), order='C')
            game_actions = np.array(game_actions).reshape((11, -1), order='C')
            game_rewards = np.array(game_rewards).reshape((11, -1), order='C')
            
        else: #1by1
            #shape = (11-n-info)
            game_state3s = np.squeeze(np.array(self.state3s_history[player_x]), axis=1)
            game_actions = np.squeeze(np.array(self.actions_history[player_x]), axis=1)
            game_rewards = np.squeeze(np.array(self.rewards_history[player_x]), axis=1)

        c1 = (game_state3s == state3s)
        c2 = (game_actions == actions)
        #c3 = (game_rewards == rewards) #can't compare 'reward' since game result add additional value to it
            
        if dc.Players.NONE == player_x or dc.Players.WEST == player_x:  #4in1 or (1by1 last=W)
            self.player_seq = []
            for player in [dc.Players.SOUTH, dc.Players.EAST, dc.Players.NORTH, dc.Players.WEST]:
                self.state3s_history[player] = []
                self.actions_history[player] = []
                self.rewards_history[player] = []

        
        c = c1.all() and c2.all() and c3.all() # and c4 #rewards_1 has updates from game result, it must not be equal to rewards_0
        if not c:
            print("check_state3s_history_proxy_learning_single FAIL!!!!!!!!!!!!!!!!!! id=", CHKIDs.CHKID_8_3)
        return c
                              
                              
    def check_state3s_agent_MCq_state3_G_A_reshape(self, state3s_batch, state3s_batch2, Gs_batch, Gs2, actions_batch, actions_batch2):
        #state3 DNN shape: round(=11)*player*batch-info(5*54)
        #state3 CNN shape: round(=11)*player*batch-info(5-54-1)
        #state3_2 shape:   round(=11)*player*batch-info(5-54)
        #action, Gs shape: round(=11)*player*batch-info(1)
        #action2, Gs2 shape: round(=11)-player*batch-info(1)
        state3s_shape = state3s_batch.shape
        batch_size2 = state3s_batch.shape[0]
        batch_size = int(batch_size2/11)
        Gs2 = np.array(Gs2).reshape(11, -1)   #remove dim=1
        
        if len(state3s_shape) == 2 : #DNN
            c1 = (state3s_batch2[0, 0, :] == state3s_batch[0, 0:54])
            c2 = (state3s_batch2[0, 1, :] == state3s_batch[0, 54*1:54*2])
            c3 = (state3s_batch2[7, 0, :] == state3s_batch[7, 0:54])
            c4 = (state3s_batch2[9, 3, :] == state3s_batch[9, 54*3:54*4])
            c5 = (state3s_batch2[batch_size2-1,2,:] == state3s_batch[batch_size2-1,54*2:54*3])
            if 33 <= batch_size2:
                c6 = (state3s_batch2[30,3,:] == state3s_batch[30, 54*3:54*4])
            else:
                c6 = (np.arange(2) == np.arange(2)) # bool_ True
            #no fraud, then no checking to [4], only [0~3]
            
            c = (c1.all() and c2.all() and c3.all() and c4.all() and c5.all() and c6.all())
            
        elif len(state3s_shape) == 4 : #CNN. (54) == (54,1)
            c1 = (state3s_batch2[0, 0, :] == np.squeeze(state3s_batch[0, 0], axis=1))
            c2 = (state3s_batch2[0, 1, :] == np.squeeze(state3s_batch[0, 1], axis=1))
            c3 = (state3s_batch2[7, 0, :] == np.squeeze(state3s_batch[7, 0], axis=1))
            c4 = (state3s_batch2[9, 3, :] == np.squeeze(state3s_batch[9, 3], axis=1))
            c5 = (state3s_batch2[batch_size2-1,2,:] == np.squeeze(state3s_batch[batch_size2-1,2], axis=1))
            if 33 <= batch_size2:
                c6 = (state3s_batch2[30,3,:] == np.squeeze(state3s_batch[30, 3], axis=1))
            else:
                c6 = (np.arange(2) == np.arange(2)) # bool_ True
            #no fraud, then no checking to [4], only [0~3]
            
            c = (c1.all() and c2.all() and c3.all() and c4.all() and c5.all() and c6.all())
        

        #action, G: checking position is same to state3 in check_state3s_agent_MCq_state3_reshape()
        if 3 <= batch_size:
            c10 = (Gs_batch[0] == Gs2[0, 0])
            c11 = (Gs_batch[2*batch_size+1] == Gs2[2, 1])
            c12 = (Gs_batch[7*batch_size+2] == Gs2[7, 2])
        else:
            c10 = (Gs_batch[0] == Gs2[0, 0])
            c11 = (Gs_batch[3*batch_size+0] == Gs2[3, 0])
            c12 = (Gs_batch[8*batch_size+0] == Gs2[8, 0])
        c = c and (c10.all() and c11.all() and c12.all())


        if 3 <= batch_size:
            c15 = (actions_batch[0] == actions_batch2[0, 0])
            c16 = (actions_batch[2*batch_size+1] == actions_batch2[2, 1])
            c17 = (actions_batch[7*batch_size+2] == actions_batch2[7, 2])
        else:
            c15 = (actions_batch[0] == actions_batch2[0, 0])
            c16 = (actions_batch[3*batch_size+0] == actions_batch2[3, 0])
            c17 = (actions_batch[8*batch_size+0] == actions_batch2[8, 0])
        c = c and (c17.all() and c15.all() and c16.all())
            
        if not c:
            print("check_state3s_agent_MCq_state3_G_A_reshape FAIL!!!!!!!!!!!!!!!!!! id=", CHKIDs.CHKID_11)
        return c

    def check_state3s_agent_MCq_state3_reshape(self, state3s_batch, state3s_batch2, fraud, bitmasks54):
        #state3 shape: round(=11)-player*batch-info(5-54)
        #state3_2 shape: round(=11)*player*batch-info(5*54)
        #ignore fraud, only check[0~3]
        batch_size = state3s_batch.shape[1]
        if 4 == fraud :
            info_shape = (4, 54*2)
        else:
            info_shape = (5, 54)
            
        # state3: checking position is same to action, G in check_state3s_agent_MCq_state3_G_A_reshape()
        if 3 <= batch_size:
            c1 = (state3s_batch2[0, info_shape[1]:info_shape[1]*2] == state3s_batch[0, 0, 1, :])
            c2 = (state3s_batch2[2*batch_size+1, info_shape[1]*2:info_shape[1]*3] == state3s_batch[2, 1, 2, :])
            c3 = (state3s_batch2[7*batch_size+2, info_shape[1]*3:info_shape[1]*4] == state3s_batch[7, 2, 3, :])
            c4 = (bitmasks54[9] == np.where(state3s_batch2[9, 0:54]>0, 1, 0))
            c5 = (bitmasks54[19] == np.where(state3s_batch2[19, 0:54]>0, 1, 0))
            c6 = (bitmasks54[29] == np.where(state3s_batch2[29, 0:54]>0, 1, 0))
        else:
            c1 = (state3s_batch2[0, info_shape[1]:info_shape[1]*2] == state3s_batch[0, 0, 1, :])
            c2 = (state3s_batch2[3*batch_size+0, info_shape[1]*2:info_shape[1]*3] == state3s_batch[3, 0, 2, :])
            c3 = (state3s_batch2[8*batch_size+0, info_shape[1]*2:info_shape[1]*3] == state3s_batch[8, 0, 2, :])
            c4 = (bitmasks54[1] == np.where(state3s_batch2[1, 0:54]>0, 1, 0))
            c5 = (bitmasks54[9] == np.where(state3s_batch2[9, 0:54]>0, 1, 0))
            c6 = np.array([])
        #no fraud, then no checking to [4], only [0~3]
        
        c = (c1.all() and c2.all() and c3.all() and c4.all() and c5.all() and c6.all())
        if not c:
            print("check_state3s_agent_MCq_state3_reshape FAIL!!!!!!!!!!!!!!!!!! id=", CHKIDs.CHKID_9)
        return c
    
    def check_state3s_agent_MCpi_state3_G_A_reshape(self, state3s_batch, state3s_batch2, Gs_batch, Gs2,  Gs_gamma_batch, Gs_gamma2, actions_batch, actions_batch2):
        #state3 shape: round(=11)*player*batch-info(5*54)
        #state3_2 shape: round(=11)*player*batch-info(5-54)
        #action, Gs shape: round(=11)*player*batch-info(1)
        #action2, Gs2 shape: round(=11)-player*batch-info(1)
        batch_size2 = state3s_batch.shape[0]
        batch_size = int(batch_size2/11)
        Gs_gamma2 = np.array(Gs_gamma2).reshape(11, -1)
        
        c1 = self.check_state3s_agent_MCq_state3_G_A_reshape(state3s_batch, state3s_batch2, Gs_batch, Gs2, actions_batch, actions_batch2)
        #Gs_gamma: checking position is same to state3, G, A in above check_state3s_agent_MCq_state3_G_A_reshape()
        if 3 <= batch_size:
            c10 = (Gs_gamma_batch[0] == Gs_gamma2[0, 0])
            c11 = (Gs_gamma_batch[2*batch_size+1] == Gs_gamma2[2, 1])
            c12 = (Gs_gamma_batch[7*batch_size+2] == Gs_gamma2[7, 2])
        else:
            c10 = (Gs_gamma_batch[0] == Gs_gamma2[0, 0])
            c11 = (Gs_gamma_batch[3*batch_size+0] == Gs_gamma2[3, 0])
            c12 = (Gs_gamma_batch[8*batch_size+0] == Gs_gamma2[8, 0])

        c = c1 and (c10.all() and c11.all() and c12.all())
        if not c:
            print("check_state3s_agent_MCpi_state3_G_A_reshape FAIL!!!!!!!!!!!!!!!!!! id=", CHKIDs.CHKID_12)
        return c

    def check_env_saver_store(self, np_start_player_cards, np_trumps, oindex_pattern, np_cards):
        si = copy.deepcopy(np_start_player_cards)  #[0]
        self.full_env_saver_cmp.append(si)
        si = copy.deepcopy(np_trumps)   #[1]
        self.full_env_saver_cmp.append(si)
        si = copy.deepcopy(np.array(oindex_pattern))   #[2]
        self.full_env_saver_cmp.append(si)
        si = copy.deepcopy(np_cards)   #[3]
        self.full_env_saver_cmp.append(si)        

    def check_env_saver_reload(self, np_start_player_cards, np_trumps, oindex_pattern, np_cards):
        if len(self.full_env_saver_cmp) == 0:
            return True

        si = copy.deepcopy(self.full_env_saver_cmp[0])
        c0 = (np_start_player_cards == si)
        si = copy.deepcopy(self.full_env_saver_cmp[1])
        c1 = True
        for np_trump, si_1 in zip(np_trumps, si):
            c1 = c1 and (np_trump == si_1).all()   #variable length list[]
        si = copy.deepcopy(self.full_env_saver_cmp[2])
        c2 = (oindex_pattern == si)
        si = copy.deepcopy(self.full_env_saver_cmp[3])
        c3 = (np_cards == si)
        
        c = c0.all() and c1 and c3.all() and c2.all()
        if not c:
            print("check_env_saver_reload FAIL!!!!!!!!!!!!!!!!!! id=", CHKIDs.CHKID_10_2)
        self.full_env_saver_cmp = []
        return c

    def check_states_storage_reshape(self, np_next_states):
        pass
        
    def check_state3s_vs_player_cards(self, new_state3s, players_cards, input_format):
        #shape state3: (n,4,5,54)
        #shape _cards[0]: (3,18,8)
        #shape _cards[1~3]: (3,3,12,8)
        state3s = copy.deepcopy(new_state3s)
        batch_size = state3s.shape[0]
        
        c= True
        c1 = np.array([])   #c1.all() = True
                         #batch  who
        checking_items = [[9,    dc.Players.WEST],
                          [2,    dc.Players.EAST ],
                          [1,    dc.Players.SOUTH],
                          [0,    dc.Players.NORTH]]
        
        for batch_id, player in checking_items:
            if batch_id >= batch_size:
                continue
            
            state3_inhand = state3s[batch_id, player-1, 0]
            state3_played = state3s[batch_id, player-1, 1]
            if dc.Players.SOUTH == player:
                player_card = players_cards[0][batch_id]
                index18_discard_yes = (player_card[:, dc.COL_DISCARDED] == True)
                oindex18 = player_card[:, 0]  #(12)
                #discard in inhand
                oindex54_discard = oindex18[index18_discard_yes]
                c1 = (state3_inhand[oindex54_discard] == input_format[1]) #discard

                #remove discard, reshape to (12)
                oindex12 = oindex18[np.logical_not(index18_discard_yes)]
                player_card = player_card[np.logical_not(index18_discard_yes)]
            else:
                player_card = players_cards[1][batch_id, player-2]
                oindex54_discard = []
            
            oindex12 = player_card[:, dc.COL_OINDEX]
            index12_trump_yes   = (player_card[:, dc.COL_TRUMP] == True)  #(12)
            index12_trump_no    = np.logical_not(index12_trump_yes)  #(12)
            index12_played_yes  = (player_card[:, dc.COL_PLAYED] == True)
            index12_played_no   = np.logical_not(index12_played_yes)

            #trump in inhand
            index12_trump_inhand = np.logical_and(index12_played_no, index12_trump_yes)
            oindex54_trump_inhand = oindex12[index12_trump_inhand]
            c2 = (state3_inhand[oindex54_trump_inhand] == input_format[3])
            #regular in inhand
            index12_regular_inhand = np.logical_and(index12_played_no, index12_trump_no)
            oindex54_regular_inhand = oindex12[index12_regular_inhand]
            c3 = (state3_inhand[oindex54_regular_inhand] == input_format[2])
            #played in inhand
            oindex54_played_inhand = oindex12[index12_played_yes]
            c4 = (state3_inhand[oindex54_played_inhand] == input_format[0])
            #never inhand
            state3_inhand[oindex54_discard] = 0
            state3_inhand[oindex54_trump_inhand] = 0
            state3_inhand[oindex54_regular_inhand] = 0
            c5 = (state3_inhand == input_format[0])
            
            #trump in played
            index12_trump_played = np.logical_and(index12_played_yes, index12_trump_yes)
            oindex54_trump_played = oindex12[index12_trump_played]
            c6 = (state3_played[oindex54_trump_played] == input_format[3])
            #regular in played
            index12_regular_played = np.logical_and(index12_played_yes, index12_trump_no)
            oindex54_regular_played = oindex12[index12_regular_played]
            c7 = (state3_played[oindex54_regular_played] == input_format[2])
            #not played in played
            index12_played_no
            oindex54_not_played = oindex12[index12_played_no]
            c8 = (state3_played[oindex54_not_played] == input_format[0])
            #never has in played
            state3_played[oindex12] = 0
            c9 = (state3_played == input_format[0])

            c = (c and c1.all() and c2.all() and c3.all() and c4.all() and c5.all() and c6.all() and c7.all() and c8.all() and c9.all())
        
        if not c:
            print("check_state3s_vs_player_cards failed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!. id=", CHKIDs.CHKID_14)
        return c

    def check_state3s_agent_CNNq_state3_reshape(self, state3s_batch, state3s_batch2, fraud):
        #state3   shape: round(=11)-player*batch-info(5,54) or (4,108)
        #state3_2 shape: round(=11)*player*batch-info(5,54,1) or (4,54*2,1)
        batch_size = state3s_batch.shape[1]
        if 4 == fraud :
            info_shape = (4, 54*2, 1)
        else:
            info_shape = (5, 54, 1)
        
        if 3 <= batch_size:
            c1 = (state3s_batch2[0, info_shape[1]:info_shape[1]*2, 0] == state3s_batch[0, 0, 1])
            c2 = (state3s_batch2[2*batch_size+1, info_shape[1]*2:info_shape[1]*3, 0] == state3s_batch[2, 1, 2])
            c3 = (state3s_batch2[7*batch_size+2, info_shape[1]*3:info_shape[1]*4, 0] == state3s_batch[7, 2, 3])
        else:
            c1 = (state3s_batch2[0, info_shape[1]:info_shape[1]*2, 0] == state3s_batch[0, 0, 1])
            c2 = (state3s_batch2[3*batch_size+0, info_shape[1]*2:info_shape[1]*3, 0] == state3s_batch[3, 0, 2])
            c3 = (state3s_batch2[8*batch_size+0, info_shape[1]*2:info_shape[1]*3, 0] == state3s_batch[8, 0, 2])
        #no fraud, then no checking to [4], only [0~3]
            
        c = (c1.all() and c2.all() and c3.all())
        if not c:
            print("check_state3s_agent_CNNq_state3_reshape FAIL!!!!!!!!!!!!!!!!!! id=", CHKIDs.CHKID_15)
        return c
    

    def check_state3s_agent_CNNq_state3_reshape_with_bitmasks54(self, state3s_batch, state3s_batch2, fraud, bitmasks54):
        c1 = self.check_state3s_agent_CNNq_state3_reshape(state3s_batch, state3s_batch2, fraud)
        batch_size = state3s_batch.shape[1]
            
        # state3: checking position is same to action, G in check_state3s_agent_MCq_state3_G_A_reshape()
        if 3 <= batch_size:
            c4 = (bitmasks54[ 9] == np.where(state3s_batch2[ 9, 0, 0:54, 0]>0, 1, 0))
            c5 = (bitmasks54[19] == np.where(state3s_batch2[19, 0, 0:54, 0]>0, 1, 0))
            c6 = (bitmasks54[29] == np.where(state3s_batch2[29, 0, 0:54, 0]>0, 1, 0))
        else:
            c4 = (bitmasks54[1] == np.where(state3s_batch2[1, 0, 0:54, 0]>0, 1, 0))
            c5 = (bitmasks54[9] == np.where(state3s_batch2[9, 0, 0:54, 0]>0, 1, 0))
            c6 = np.array([])
        
        c = (c1 and c4.all() and c5.all() and c6.all())
        if not c:
            print("check_state3s_agent_CNNq_state3_reshape_with_bitmasks54 FAIL!!!!!!!!!!!!!!!!!! id=", CHKIDs.CHKID_16)
        return c

    def check_cards_integrity(self, known_cards):
        verify_all_0 = np.where(known_cards==1, 0, known_cards)
        c = (verify_all_0 > 0)
        if c.any():
            print("check_cards_integrity: FAIL. id=", CHKIDs.CHKID_17)
        return c
        
    def check_state3s_reshape_in_guess(self, guess_cards0, guess_cards1):
        guess_shape = guess_cards0.shape    #(n, 3*54)
        
        c1 = True
        if guess_shape[0] >= 3:
            #c1 = (state3s_batch0[2,3,10] == state3s_batch1[2, 3*54+10])
            c2 = (guess_cards0[2,2*54+10] == guess_cards1[2,2,10])
        else:
            #c1 = (state3s_batch0[0,3,10] == state3s_batch1[0, 3*54+10])
            c2 = (guess_cards0[0,2*54+10] == guess_cards1[0,2,10])
        
        c = c1 and c2
        if not c:
            print("check_state3s_reshape_in_guess: FAIL. id=", CHKIDs.CHKID_18)
        return c            
    
    def check_softmax(self, unknown_cards, guess_cards1, guess_cards2):
        #unknown_cards(n, 1, 54)
        #guess_cards2=(n,3,54) #0~1 pssibility
        err = 1e-5
        guess_shape = guess_cards1.shape #(n,3,54)
        if guess_shape[0] >= 3:
            if True == unknown_cards[2, 0, 10]:
                x0 = guess_cards1[2, :, 10]
                x_max = x0.max()
                x1 = x0 - x_max
                exp_x = np.exp(x1)
                exp_x_sum = np.sum(exp_x)
                softmax_x = exp_x / exp_x_sum
                c1_diff = softmax_x - guess_cards2[2, :, 10]
                c1 = (c1_diff <= err)
                sum_diff = np.abs(1.0 - guess_cards2[2, :, 10].sum())
                c2 = (sum_diff < err)
            else:
                c1 = np.logical_or((0 == guess_cards2[2, :, 10]), np.isnan(guess_cards2[2, :, 10]))
                c2 = True
        else:
            if True == unknown_cards[0, 0, 10]:
                x0 = guess_cards1[0, :, 10]
                x_max = x0.max()
                x1 = x0 - x_max
                exp_x = np.exp(x1)
                exp_x_sum = np.sum(exp_x)
                softmax_x = exp_x / exp_x_sum
                c1_diff = softmax_x - guess_cards2[0, :, 10]
                c1 = (c1_diff <= err)
                sum_diff = np.abs(1.0 - guess_cards2[0, :, 10].sum())
                c2 = (sum_diff < err)
            else:
                c1 = (0 == guess_cards2[0, :, 10])
                c2 = True
        
        
        guess_sum0 = guess_cards2.sum(axis=1)  #value would be 0 or close 1
        v1 = np.logical_and(guess_sum0>(1.0-err), guess_sum0<(1.0+err))
        v2 = (guess_sum0==0)
        v3 = np.isnan(guess_sum0)  # TBD: all = false !!!
        c3 = np.logical_or(v1, v2, v3)
        
        c = c1.all() and c2 and c3.all()
        if not c:
            print("check_softmax: FAIL. id=", CHKIDs.CHKID_19)
            print("check_softmax: FAIL. id=", unknown_cards)
            print("check_softmax: FAIL. id=", guess_cards1)
            print("check_softmax: FAIL. id=", guess_cards2)
        return c            
            
    def check_guess_format(self, state3s_batch, guessed_cards, new_state3s_batch):
        #during decide()
        #state3s_batch(n, 5, 54), 
        #guessed_cards(n, 3 ,54)
        #new_state3s_batch(n, 4, 54*2)
        # new_state3s:       0, 1, 2, ... 53, | 54, 55, ... 106, 107
        # 0: myself:         state3s[0], 0~53 |   state3s[1], 0~53
        # 1: 1st follower:   guessed[0]: 0~53 |   state3s[2], 0~53
        # 2: 2nd follower:   guessed[1]: 0~53 |   state3s[3], 0~53
        # 3: 3rd follower:   guessed[2]: 0~53 |   state3s[4], 0~53
        
        batch_size = state3s_batch.shape[0]
        if batch_size >=3 :
            c1 = (new_state3s_batch[2, 0, 30]  == state3s_batch[2, 0, 30])
            c2 = (new_state3s_batch[2, 0, 100] == state3s_batch[2, 1, 100-54])
            c3 = (new_state3s_batch[2, 1, 30]  == guessed_cards[2, 0, 30])
            c4 = (new_state3s_batch[2, 1, 100] == state3s_batch[2, 2, 100-54])
            c5 = (new_state3s_batch[2, 2, 30]  == guessed_cards[2, 1, 30])
            c6 = (new_state3s_batch[2, 2, 100] == state3s_batch[2, 3, 100-54])
            c7 = (new_state3s_batch[2, 3, 30]  == guessed_cards[2, 2, 30])
            c8 = (new_state3s_batch[2, 3, 100] == state3s_batch[2, 4, 100-54])
        else:
            c1 = (new_state3s_batch[0, 0, 30]  == state3s_batch[0, 0, 30])
            c2 = (new_state3s_batch[0, 0, 100] == state3s_batch[0, 1, 100-54])
            c3 = (new_state3s_batch[0, 1, 30]  == guessed_cards[0, 0, 30])
            c4 = (new_state3s_batch[0, 1, 100] == state3s_batch[0, 2, 100-54])
            c5 = (new_state3s_batch[0, 2, 30]  == guessed_cards[0, 1, 30])
            c6 = (new_state3s_batch[0, 2, 100] == state3s_batch[0, 3, 100-54])
            c7 = (new_state3s_batch[0, 3, 30]  == guessed_cards[0, 2, 30])
            c8 = (new_state3s_batch[0, 3, 100] == state3s_batch[0, 4, 100-54])
        
        c = c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8
        if not c:
            print("check_guess_format: FAIL. id=", CHKIDs.CHKID_20)
        return c            
            
    def check_learning_unique_action(self, oindex):
        c = True
        batch_size0 = oindex.shape[0]
        batch_size1 = oindex.shape[1]
        

        #### opion 1. time cost: option1:2=0.56:2.75. lopp=100000
        for i in range(batch_size1):
            oindex_in_one_cardset = set(oindex[:,i])   #dim=[0]上，必须没有重复oindex
            if len(oindex_in_one_cardset) != batch_size0:
                c = False
                break

        #### opion 2
        '''
        oindex_bitmap = np.zeros((batch_size1, 54))
        oindex_T = oindex.T
        batch_size1_rep_batch_size0 = np.arange(batch_size1)[:,np.newaxis].repeat(batch_size0, axis=1)
        oindex_bitmap[batch_size1_rep_batch_size0.reshape(-1), oindex_T.reshape(-1)] = 1
        oindex_sum = np.sum(oindex_bitmap, axis=1)
        c1 = (oindex_sum == batch_size0)
        c = c1.all()
        '''
        if not c:
            print("check_learning_unique_action: FAIL. id=", CHKIDs.CHKID_21)
        return c            
        
        
    def check_reshape_game_single(self, state3s_batch, state3s_batch1, actions_batch, actions_batch1):
        #state3s_batch1.shape=(round-player-batch-info)
        #state3s_batch.shape=(round-player*batch-info)
        batch = state3s_batch1.shape[2]
        if batch >= 3:
            c1 = (state3s_batch1[5, 1, 2] == state3s_batch[5, 1*batch+2])
            c2 = (state3s_batch1[9, 2, 1] == state3s_batch[9, 2*batch+1])
            c3 = (actions_batch1[5, 1, 2] == actions_batch[5, 1*batch+2])
            c4 = (actions_batch1[9, 2, 1] == actions_batch[9, 2*batch+1])
        else:
            c1 = (state3s_batch1[5, 1, 0] == state3s_batch[5, 1*batch+0])
            c2 = (state3s_batch1[9, 2, 0] == state3s_batch[9, 2*batch+0])
            c3 = (actions_batch1[5, 1, 0] == actions_batch[5, 1*batch+0])
            c4 = (actions_batch1[9, 2, 0] == actions_batch[9, 2*batch+0])

        c = (c1.all() and c2.all() and c3.all() and c4.all())
        if not c:
            print("check_reshape_game_single: FAIL. id=", CHKIDs.CHKID_22)
        return c            
    
    def check_guess_trump(self, guess_cards3, trumps, guess_cards):
        #guess_cards(n,3,54) = guess_cards3(n,3,54) * trumps(1,1,54)

        if guess_cards3.shape[0] >= 30:
            v1 = guess_cards3[20,1] * trumps[0,0]
            v2 = guess_cards3[10,2] * trumps[0,0]
            c1 = (v1 == guess_cards[20,1])
            c2 = (v2 == guess_cards[10,2])
        elif guess_cards3.shape[0] >= 3:
            v1 = guess_cards3[2,1] * trumps[0,0]
            v2 = guess_cards3[1,2] * trumps[0,0]
            c1 = (v1 == guess_cards[2,1])
            c2 = (v2 == guess_cards[1,2])
        else:
            v1 = guess_cards3[0,1] * trumps[0,0]
            v2 = guess_cards3[0,2] * trumps[0,0]
            c1 = (v1 == guess_cards[0,1])
            c2 = (v2 == guess_cards[0,2])
            
        c = (c1.all() and c2.all())
        if not c:
            print("check_guess_trump: FAIL. id=", CHKIDs.CHKID_23)
        return c            
        

    def check_guess_calc_discard(self, guess_cards, guess_discard_oindex, state3s_batch, guess_inhand, guess_discard, unknown_cards, guess_env_callback):
        #################################
        # guessed discard position
        banker_pos = guess_env_callback(2) #banker_pos shape=(<batch,1>,),  =[0,1,2,3]
        fmts = guess_env_callback(3)

        #every oindex happen once
        is_fmt0 = (guess_cards == fmts[0])
        is_fmt1 = (guess_cards == fmts[1])
        is_fmt2 = (guess_cards == fmts[2])
        is_fmt3 = (guess_cards == fmts[3])
        all_fmts = is_fmt0 + is_fmt1 + is_fmt2 + is_fmt3
        c1 = all_fmts.all()

        #discard checking
        current_is_banker = np.where(banker_pos==0, 1, 0)
        current_is_not_banker = np.logical_not(current_is_banker)
        current_is_not_banker_len = np.where(current_is_not_banker == True)
        if current_is_not_banker_len[0].shape[0] > 0:
            #discard oindex position should be fmts[1]
            bankers_guessed_cards = guess_cards[current_is_not_banker, banker_pos[current_is_not_banker]-1]  #current is not banker, guess area里banker的位置
            bankers_guessed_discard = guess_discard_oindex[current_is_not_banker]
            batch_size_rep6 = np.arange(current_is_not_banker_len[0].shape[0])[:,np.newaxis].repeat(6, axis=1)
            c2 = (bankers_guessed_cards[batch_size_rep6.reshape(-1), bankers_guessed_discard.reshape(-1)] == fmts[1])
        else:
            c2 = np.logical_not(is_fmt1.any())
        
        c = c1 and c2.all()
        if not c:
            print("check_guess_calc_discard: FAIL. id=", CHKIDs.CHKID_24)
        return c
        
        

    def check_guess_calc_length(self, guessed_cards_result, state3s_batch, guess_inhand, guess_discard, unknown_cards, players_inhand_length, guess_env_callback):
        # length checking: 
        # input: players_inhand_length
        #            (n,n,n)     ==> (n,n,n):       0 or 3 players played of the round; current is banker
        #                        ==> (n,n,n+6),     0 or 3 players played of the round; banker is at n+6; 
        #                        ==> (n,n+6,n),     0 or 3 players played of the round; banker is at n+6; 
        #                        ==> (n+6,n,n):     0 or 3 players played of the round; banker is at n+6; 
        #
        #            (n+1,n+1,n) ==> (n+1,n+1,n):   1 player played; current is banker
        #                        ==> (n+1,n+1,n+6): 1 player played; banker already played; banker at n+6
        #                        ==> (n+1,n+7,n),   1 player played; banker at n+7
        #                        ==> (n+7,n+1,n),   1 player played; banker at n+7
        #
        #            (n+1,n,n)   ==> (n+1,n,n),     2 players played; current is banker
        #                        ==> (n+1,n,n+6),   2 players played; banker at n+6
        #                        ==> (n+1,n+6,n),   2 players played; banker at n+6
        #                        ==> (n+7,n,n),     2 players played; banker at n+7

        c = False
        batch_size = guessed_cards_result.shape[0]
        
        bankers_yes = guess_env_callback(1, state3s_batch)  #current is banker. banker = True
        bankers_no  = np.logical_not(bankers_yes)
        banker_pos = guess_env_callback(2) #banker_pos shape=(<batch,1>,),  =[0,1,2,3]

        #additional banker position check. does guess_env_callback() work well?
        c1 = (banker_pos[bankers_yes] == 0)
        
        guessed_cards_result_1 = np.where(guessed_cards_result!=0, 1, 0)
        guessed_cards_result_len = np.sum(guessed_cards_result_1, axis=2)

        current_is_banker = (players_inhand_length[bankers_yes] == guessed_cards_result_len[bankers_yes])
        current_is_not_banker_len = players_inhand_length[bankers_no, banker_pos[bankers_no]-1]
        if current_is_not_banker_len.shape[0] == 0:
            c2 = (current_is_banker.shape[0] == batch_size)
            c3 = True
        else:
            current_is_not_banker = ((current_is_not_banker_len + 6) == guessed_cards_result_len[bankers_no, banker_pos[bankers_no]-1])
            c2 = ((current_is_banker.shape[0] + current_is_not_banker.shape[0]) == batch_size)
            c3 = current_is_not_banker.all()

        c = c1.all() and c2 and c3
        if not c:
            print("check_guess_calc_length: FAIL. id=", CHKIDs.CHKID_25)
        return c

    def check_guess_calc_unknow_bits(self, unknown_cards_minus_discard, bankers_yes, unknown_cards_minus_discard0, unknown_cards_minus_discard1, state3s_batch, unknown_cards, guess_discard1, guess_discard_oindex):
        unknown_bits = copy.deepcopy(unknown_cards_minus_discard)
        unknown_bits = np.where(unknown_bits==0, 0, unknown_bits)   #0 and 1 are avaiable value
        unknown_bits = np.where(unknown_bits==1, 0, unknown_bits)
        c1 = np.where(unknown_bits!=0, 1, unknown_bits)
        c = c1.any()
        if c:
            print("check_guess_calc_unknow_bits: FAIL. id=", CHKIDs.CHKID_26)
            print("check_guess_calc_unknow_bits: FAIL. 1", unknown_cards_minus_discard)
            print("check_guess_calc_unknow_bits: FAIL. 2", bankers_yes)
            print("check_guess_calc_unknow_bits: FAIL. 3", unknown_cards_minus_discard0)
            print("check_guess_calc_unknow_bits: FAIL. 4", unknown_cards_minus_discard1)
            print("check_guess_calc_unknow_bits: FAIL. 5", state3s_batch)
            print("check_guess_calc_unknow_bits: FAIL. 6", unknown_cards)
            print("check_guess_calc_unknow_bits: FAIL. 7", guess_discard1)
            print("check_guess_calc_unknow_bits: FAIL. 8", guess_discard_oindex)

        return not c
        
    
checkpoints = Game54_Online_Checkpoints()  #runtime inited by Game::__init__()

