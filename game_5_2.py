#v4.2: update for TD serise implementations. but blocked due to imperfect iformation game theory
#      update .h5 file name, remove gameid(in InputParameters.read() only) for crossing competition. 缺点：init时，容易误删已有的.h5
#v5.0: aglin to env v5.0. parallel envs running optm
#    : support GPU. 
#v5.1: state3s_fraud
#      runtime checkpoints added
#      CNN added
#      fraud shape -> (4, 54*2). added 'played' info. has error. but online checking MISSED!
#v5.2: state refactory for diff-structured(state3, state3_fraud, and coming states)
#      reward alg changed. fmt[-1, 0.5, 1, 2] is obsolete
#      add TD with playback
#      guess class added. train in orund
##################################################
# ToDOs
# 1. state3, state2 in Game{} is useless, remove them
# 2. in CNN Q alg, input would be in range [-1, 1] so that the normalization works. as well as the loss=MSE
#    in DNN Q alg, there is no batchnormalization(). but the predict() result is no big
# 3. DONE. TD with and without fraud mode, replay mode. how to setup the experiecen priority?
# 4. DONE. reshape the network for CNN with similar value scope of axis0 and axis1
# 5. DONE. when init the agents, if the 'agent id' are same, should not create new instance. refer to eisting one 
# 6. guess() alg optimize
# 7. comparison in discard_q: 6 dumps vs. 54 sigmiod
##################################################
import os
import time
import numpy as np
import random as rd
import copy
import psutil
import sys, getopt
from multiprocessing import Process, Manager, set_start_method
import gc
#gc.disable()

import tensorflow as tf
#in root@CentOS: apply all CPUs(28 cores) at ???: 
#in ute@Debian: apply all CPUs(28 cores) at ???: does 'root' impact that? A: no, both root and ute take CPU >20%
#BTW, sudo or su, doesn't make the conda env(. block running python discard_testbench...) A: 'conda init bash' will add conda env to root rc
#CPU mode: 5.6h(GPU-133 28 CPUs with 60%+) vs. 36h(ute 1 CPU=100%)
#TB verify in 'game.py' ????

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
config.gpu_options.per_process_gpu_memory_fraction = 0.95  #default = 100%(actual=~96%)
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
    
    # it is late if perform below lines here. MUST run them before any program code !!
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #实现卡号匹配
    #config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.9
    #tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    return








import deal_cards_5_2 as dc
import cards_env_5_2 as ce
import discard_q_net_keras_5_2 as q_net

import play_agent_DNN_MC_q_5_2 as agent_1
import play_agent_DNN_MC_pi_5_2 as agent_2
import play_agent_DNN_TD_q_5_2 as agent_3
import play_agent_CNN_MC_q_5_2 as agent_6
import play_agent_CNN_MC_pi_5_2 as agent_7
import play_agent_CNN_TD_q_5_2 as agent_11

import game_verify_online_5_2 as game_verify
import game_verify_offline_5_2 as game_verify_off
import config_5_2 as cfg
import meas_5_2 as meas
from play_agent_proxy_5_2 import PlayAgentProxy 



class Game54:
    def __init__(self, batch_size, envs, discard_agent, play_agent0, play_agent1, play_agent2, play_agent3, flag_4in1=True, train=True):
        self.logs = []  #matrix and log collections
        self.batch_size = batch_size
        self.card_envs = envs
        self.discard_agent = discard_agent
        self.state3s = []  #almost not used at all
        self.state2s = []  #almost not used at all
        self.best_discards_oindexes = []
        self.discard_rewards = []
        self.flag_4in1 = flag_4in1
        self.train = train
        self.game_his = []

        print("game init checkpoint id: ", id(game_verify.checkpoints))

        self.play_agent = {}
        self.agent_shape = {}
        self.play_agent[dc.Players.SOUTH] = PlayAgentProxy(play_agent0, game_verify.checkpoints, envs.guess_callback)
        self.play_agent[dc.Players.EAST]  = PlayAgentProxy(play_agent1, game_verify.checkpoints, envs.guess_callback)
        self.play_agent[dc.Players.NORTH] = PlayAgentProxy(play_agent2, game_verify.checkpoints, envs.guess_callback)
        self.play_agent[dc.Players.WEST]  = PlayAgentProxy(play_agent3, game_verify.checkpoints, envs.guess_callback)
        
        for player in [dc.Players.SOUTH, dc.Players.EAST, dc.Players.NORTH, dc.Players.WEST]:
            self.agent_shape[player] = self.play_agent[player].get_agent_state_shape()
        
        print("Game init done")

        return
    
    def deal_card(self, render=False, keep=True):
        #env only support same banker. dc.get_player_cards_by_players() can't convert mixed shape(12, 18)
        bankers = np.array([dc.Players.SOUTH, dc.Players.SOUTH, dc.Players.SOUTH, dc.Players.SOUTH]*(int(self.batch_size/4+1))).reshape(-1)
        state3s, state2s, best_discards_oindexes, _, _ = self.card_envs.reset(bankers[:self.batch_size], keep=keep, render=render)
        
        return state3s, state2s, best_discards_oindexes

        
    def discard(self):
        #YDLs: discarding_oindex, _, _ = self.discard_agent.decide_onego(self.state2s)
        discarding_oindexes, _, _ = self.discard_agent.decide_onego(self.state2s)
        #YDLs： discarding_oindex = discarding_oindex.reshape(-1)
        
        #verify
        #YDLs: set1 = set(self.best_discards_oindex)
        #YDLs: set2 = set(discarding_oindex)
        #YDLs: covered_by_best = len(set1 & set2)
        #covered_by_best = [ len(set(best_discards_oindex) & set(discarding_oindex)) for best_discards_oindex, discarding_oindex in zip(self.best_discards_oindexes, discarding_oindexes)]

        #reward(scalar) and done(scalar) are meaningless in dump-6 alg
        state3s, state2s, reward, done = self.card_envs.discard_step2(discarding_oindexes)
        self.card_envs.discard_done(discarding_oindexes)
        return state3s, state2s, reward, done

    def play_one_round(self, last_winners):
        #support batch envs(card deals) with "list=[...for]" format
        
        round_played_cards = [[] for i in range(self.batch_size)]

        #verify, batch=3. the verify can't be applyed to env.step_play_1card() since env.players_cards[1] hsa default players order=E,N,W
        #last_winners = [dc.Players.SOUTH, dc.Players.NORTH, dc.Players.WEST]
        
        first_positions = np.argmax((self.card_envs.players[:4,np.newaxis]==last_winners), axis=0) #4*1~n => shape(4, n) -> argmax shape(n,)
        
        #self.batch_size as row index
        pos_index_1 = first_positions[:, np.newaxis].repeat(4, axis=1)
        pos_index_2 = np.arange(4)[np.newaxis,:].repeat(self.batch_size, axis=0)
        player_sequences = self.card_envs.players[(pos_index_1 + pos_index_2).reshape(-1)].reshape(self.batch_size, 4)
        players_batch = player_sequences.T
        
        leading_trumps = np.array([0]*self.batch_size)
        leading_suits = np.array([dc.CardSuits.NONE]*self.batch_size)

        #####################
        # state(before), action, b storage: for return only
        # (1,) = 1 round
        #####################
        n_s_shape = (1,) + (self.batch_size,) + self.agent_shape[dc.Players.SOUTH]  #must put ',' following 'batch_size', then the '+' works
        n_e_shape = (1,) + (self.batch_size,) + self.agent_shape[dc.Players.EAST]
        n_n_shape = (1,) + (self.batch_size,) + self.agent_shape[dc.Players.NORTH]
        n_w_shape = (1,) + (self.batch_size,) + self.agent_shape[dc.Players.WEST]
       
        state_dict  = {dc.Players.SOUTH: np.full(n_s_shape, -float('inf'), dtype=np.float16),
                       dc.Players.EAST:  np.full(n_e_shape, -float('inf'), dtype=np.float16),
                       dc.Players.NORTH: np.full(n_n_shape, -float('inf'), dtype=np.float16),
                       dc.Players.WEST:  np.full(n_w_shape, -float('inf'), dtype=np.float16) }
        action_dict = {dc.Players.SOUTH: np.full((1, self.batch_size,), -float('inf'), dtype=np.int8),
                       dc.Players.EAST:  np.full((1, self.batch_size,), -float('inf'), dtype=np.int8),
                       dc.Players.NORTH: np.full((1, self.batch_size,), -float('inf'), dtype=np.int8),
                       dc.Players.WEST:  np.full((1, self.batch_size,), -float('inf'), dtype=np.int8) }
        reward_dict = {dc.Players.SOUTH: np.full((1, self.batch_size,), -float('inf'), dtype=np.float16),
                       dc.Players.EAST:  np.full((1, self.batch_size,), -float('inf'), dtype=np.float16),
                       dc.Players.NORTH: np.full((1, self.batch_size,), -float('inf'), dtype=np.float16),
                       dc.Players.WEST:  np.full((1, self.batch_size,), -float('inf'), dtype=np.float16) }
        b_dict      = {dc.Players.SOUTH: np.full((1, self.batch_size,), -float('inf'), dtype=np.float16),
                       dc.Players.EAST:  np.full((1, self.batch_size,), -float('inf'), dtype=np.float16),
                       dc.Players.NORTH: np.full((1, self.batch_size,), -float('inf'), dtype=np.float16),
                       dc.Players.WEST:  np.full((1, self.batch_size,), -float('inf'), dtype=np.float16) }
        next_state_dict  =  \
                      {dc.Players.SOUTH: np.full(n_s_shape, -float('inf'), dtype=np.float16), #compatible to state3 and state3_fraud
                       dc.Players.EAST:  np.full(n_e_shape, -float('inf'), dtype=np.float16),
                       dc.Players.NORTH: np.full(n_n_shape, -float('inf'), dtype=np.float16),
                       dc.Players.WEST:  np.full(n_w_shape, -float('inf'), dtype=np.float16) }

        #####################
        # play one by one(rather than a loop, easy for debuging)
        #####################
        cardset_index_0 = np.arange(self.batch_size)
        for l, players_x in enumerate(players_batch): #loop=4

            tc81 = time.time()
            playing_state3s_x = self.card_envs.state3s[cardset_index_0, players_x-1]  #as index, should be: playername-1
            np_all_player_state3s_fraud = self.card_envs.build_state3s_fraud(players_x)
            np_discards = self.card_envs.card_discarded_for_guess()

            #guess learning is possible per single round. round_larning()  before decide() due to decide() and step_play()之后的state3/fraud 没有用在这轮的learning中。不如放到decide() 之前
            self.all_players_learning_single_round(playing_state3s_x, np_all_player_state3s_fraud, np_discards, players_x)
            
            ### verify: state3s history in a round
            _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_7_1, self.card_envs.state3s, np_all_player_state3s_fraud)

            #return: oindex, trump, suit, [mask for verify only]
            if True == self.flag_4in1:
                self.card_envs.banker_position_for_guess(players_x)  #full batch
                #all agents are same. just select SOUTH as decide agent 
                real_states, oindexs, bs = self.play_agent[dc.Players.SOUTH].decide(self.card_envs, playing_state3s_x, leading_trumps, leading_suits, self.train, player_state3s_fraud=np_all_player_state3s_fraud )
            else:
                real_states_l = []
                oindexs_l = []
                bs_l = []
                
                #zip(), 如果loop很短(4 players)，zip的开销可能大于收益. 不明显
                #state shape=(n, info). loop=n
                for i, player, playing_state3, leading_trump, leading_suit, np_player_state3 in zip(np.arange(self.batch_size), players_x, playing_state3s_x, leading_trumps, leading_suits, np_all_player_state3s_fraud):
                    self.card_envs.banker_position_for_guess(players_x, i)  #specific cardset in full batch
                    real_state, oindex, b = self.play_agent[player].decide(self.card_envs, playing_state3, leading_trump, leading_suit, self.train, player_state3s_fraud=np_player_state3)
                    real_states_l.append(real_state)
                    oindexs_l.append(oindex)
                    bs_l.append(b)                    
                
                oindexs = np.squeeze(np.array(oindexs_l), axis=1)
                bs = np.squeeze(np.array(bs_l), axis=1)

            tc82 = time.time()
            meas.perfs.game.perf_diff_8 += tc82 - tc81

            tc91 = time.time()
            #whatever fraud or not, below trump_batch is correct. shape=(n,5,54) or (n,4,54), dim[1]==0 is current player
            trumps = playing_state3s_x[np.arange(self.batch_size), 0, oindexs]
            if 0 == l : #leading
                leading_suits = dc.oindex_2_suit(oindexs)
                leading_trumps = trumps
            
            for round_played_card, player, oindex in zip(round_played_cards, players_x, oindexs): #loop batch
                round_played_card.append([player, oindex])
            next_state3s, next_state3s_fraud = self.card_envs.step_play_1card(players_x, trumps, oindexs, in_round_oindex=round_played_cards)

            #state_dict: dict[S, E, N, W] with np(n-info)
            for batch_id, player in enumerate(players_x):
                if True == self.flag_4in1:
                    state_dict[player][0, batch_id] = real_states[batch_id] 
                else:
                    try:
                        state_dict[player][0, batch_id] = np.squeeze(np.array(real_states_l[batch_id]), axis=0)
                    except ValueError as e:
                        print("YDL: dict() ", e)
                        
                action_dict[player][0, batch_id] = oindexs[batch_id]
                b_dict[player][0, batch_id] = bs[batch_id]

                #collect next_statexx. for TD alg
                # next_state3s.shape=(n,4,5,54)
                # next_state3s_fraud.shape=(n,4,54*2). dim[4] starts with 'player'
                if (next_state_dict[player][0, batch_id].shape == next_state3s[batch_id, player-1].shape):
                    next_state_dict[player][0, batch_id] = next_state3s[batch_id, player-1] 
                else:
                    next_state_dict[player][0, batch_id] = next_state3s_fraud[batch_id]

            ### verify: state3s history in round
            _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_7_2,  oindexs, self.card_envs.state3s, self.card_envs.state3s_fraud, players_x)

            tc92 = time.time()
            meas.perfs.game.perf_diff_9 += tc92 - tc91

            #print("play_one_round0, player+oindex ", players_x, oindexs, dc.oindex_2_suit_name(oindexs))
            #print("play_one_round0: first players: trump+leading", leading_trumps, leading_suits)
        
        
        #####################
        # decide the round winner and score
        #####################
        tca1 = time.time()
        round_winners, scores_got = self.card_envs.round_result(round_played_cards)
        #print("play_one_round: round_winner+score :", round_winners, scores_got)
        tca2 = time.time()
        meas.perfs.game.perf_diff_a += tca2 - tca1
        
        ######################
        # decide the reward for every players
        ######################
        tcb1 = time.time()
        rewards = self.card_envs.round_rewards(round_winners, player_sequences, scores_got)
        np_rewards = rewards.T

        #reward_dict: dict[S, E, N, W] with np(n-1)
        for players in players_batch: #players_batch, np_rewards=(4,n)
            for batch_id, player in enumerate(players):
                reward_dict[player][0, batch_id] = np_rewards[player-1, batch_id]

        ### verify: state3s history
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_4, rewards, reward_dict, player_sequences)
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_5, state_dict, action_dict, reward_dict, b_dict, next_state_dict)

        tcb2 = time.time()
        meas.perfs.game.perf_diff_b += tcb2 - tcb1

        ######################
        # collect the round experience
        ######################
        tcc1 = time.time()
        ### verify: state3s history
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_8_1, state_dict, action_dict, reward_dict, player_sequences)

        tcc2 = time.time()
        meas.perfs.game.perf_diff_c += tcc2 - tcc1
        
        #don't learning for single round since there is not the final winner
        #v5.1-:output shape: round(1)-player-batch-info
        #v5.2+:state3s: dict[S, E, N, W] with np(n-info)
        return round_winners, scores_got, state_dict, action_dict, reward_dict, b_dict, next_state_dict

    
    def play_games(self, games, render, keep_batch, print_game_result_cycle):
        #support batch env(card deal) with "for i" format
        print("play game() start")


        for i in range(0, games, self.batch_size):  #how many games will play
            #t, u = meas.read_GPU_memory_usage(0)
            #print("YDL GPU MEM 0 at round i, (total, used): ", i, t, u)

            tc11 = time.time()
            #deal card. keep=True train for single set
            if 0 == i: # "and True == keep_batch" needed?
                #first run, must keep=False
                state3s, state2s, best_discards_oindexes = self.deal_card(keep=False, render=render)
            else:
                state3s, state2s, best_discards_oindexes = self.deal_card(keep=keep_batch, render=render)
            self.state3s, self.state2s, self.best_discards_oindexes = state3s, state2s, best_discards_oindexes
            tc12 = time.time()
            meas.perfs.game.perf_diff_1 += tc12 - tc11
            
            ######################
            # banker discarding 6 cards
            ######################
            tc21 = time.time()
            state3s, state2s, _, _ = self.discard()
            self.state3s, self.state2s = state3s, state2s
            tc22 = time.time()
            meas.perfs.game.perf_diff_2 += tc22 - tc21
            
            ######################
            # init banker state3 after discard
            ######################
            last_winners = self.card_envs.full_poker.np_bankers
            game_state3s = {}
            game_actions = {}
            game_rewards = {}
            game_bs      = {}
            game_next_states = {}
            
            scores = np.zeros((self.batch_size, 4))  #S=0, E=1 etc.. index=player-1

            #t, u = meas.read_GPU_memory_usage(0)
            #print("YDL GPU MEM 1 at round i, (total, used): ", i, t, u)

            ######################
            # play 11 rounds
            ######################
            tc31 = time.time()
            for j in range(11): #const 11+1 cards per player
                round_winners, scores_got, state3s, actions, rewards, bs, next_state = self.play_one_round(last_winners)
                #round learning is meaningless since no final reward got
                #but guess learning is possible
                self.all_players_learning_full_round(state3s, actions, rewards, bs, next_state, j, last_winners)  #learning per round. TD only
                ### verify:
                _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_14, self.card_envs.state3s, self.card_envs.players_cards, self.card_envs.net_input_format)

                ######################
                # store the score and trajectory
                ######################
                scores[np.arange(self.batch_size), round_winners-1] += scores_got
                
                #v5.1- shape: 
                #  state3s:      np(round(1)-player(4)-batch(n)-info)
                #  game_state3s: np(round(11)-player(4)-batch(n)-info)
                #v5.2+ shape:
                #  state3s:          dict[S, E, N, W] with np(n-info)
                #  game_state3s:     dict[S, E, N, W] with np(11-n-info)
                #  actions, rewards: dict[S, E, N, W] with np(n-1)
                if not game_state3s:
                    game_state3s = state3s
                    game_actions = actions
                    game_rewards = rewards
                    game_bs      = bs
                    game_next_states = next_state
                else:
                    for player in self.card_envs.players[0:4]:
                        game_state3s[player] = np.concatenate((game_state3s[player], state3s[player]), axis=0)
                        game_actions[player] = np.concatenate((game_actions[player], actions[player]), axis=0)
                        game_rewards[player] = np.concatenate((game_rewards[player], rewards[player]), axis=0)
                        game_bs[player]      = np.concatenate((game_bs[player], bs[player]), axis=0)
                        game_next_states[player] = np.concatenate((game_next_states[player], next_state[player]), axis=0)

                ### verify: state3s history
                _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_8_2, game_state3s, j)

                ######################
                # update the states
                ######################
                last_winners = round_winners

            ######################
            # last round 12nd. result. no decision needed.
            ######################
            last_round_cards = self.card_envs.collect_card_last_round(last_winners) 
            last_round_winners, last_round_scores_got = self.card_envs.round_result(last_round_cards)
            scores[np.arange(self.batch_size), last_round_winners-1] += last_round_scores_got
            tc32 = time.time()
            meas.perfs.game.perf_diff_3 += tc32 - tc31
            
            ######################
            # game result added
            ######################
            tc41 = time.time()
            game_winners, level_raised, s_sn, s_ew = self.card_envs.game_result(last_round_winners, scores) #rewards for every single playing with final result
            if False == self.train: #measure in demo only
                self.game_his.append([game_winners[:,0], level_raised])
                
            if i % print_game_result_cycle < self.batch_size :
                gbs = gc.collect()
                #print("game-id result i and garbages: ", param_set.game_id, i, game_winners, level_raised, s_sn, s_ew, gbs)
                print("game-id result i and garbages: ", param_set.game_id, i, gbs)

            # winner +=fmt[3], losser -=fmt[3]
            #for winner in game_winners
            game_rewards = self.card_envs.game_rewards(game_winners, game_rewards, level_raised)
            
            tc42 = time.time()
            meas.perfs.game.perf_diff_4 += tc42 - tc41
            
            ######################
            # learning per game
            # shape: round-player-batch-info
            ######################
            #t, u = meas.read_GPU_memory_usage(0)
            #print("YDL GPU MEM 2 at round i, (total, used): ", i, t, u)

            tc51 = time.time()
            self.all_players_learning_single(game_state3s, game_actions, game_rewards, game_bs, game_next_states)  #learning per game
            tc52 = time.time()
            meas.perfs.game.perf_diff_5 += tc52 - tc51

            #t, u = meas.read_GPU_memory_usage(0)
            #print("YDL GPU MEM 3 at round i, (total, used): ", i, t, u)

        ######################
        # learning again as sampling
        ######################
        tc71 = time.time()
        self.all_players_learning_multi() #learning for multiple games
        tc72 = time.time()
        meas.perfs.game.perf_diff_7 += tc72 - tc71
        
        return 
    
    def all_players_learning_single_round(self, playing_state3s_x, np_all_player_state3s_fraud, np_discards, players_x):
        # 'guess' learning from single round(1 player in a round). 
        if False == self.train:
            return

        #playing_state3s_x shape: 1 player only (batch-info)
        if True == self.flag_4in1:
            state3s_batch = playing_state3s_x
            state3s_fraud_batch = np_all_player_state3s_fraud
            discard_batch = np_discards
            #input shape=(batch(=n)-info)
            self.play_agent[dc.Players.SOUTH].learning_single_round(state3s_batch, state3s_fraud_batch, discard_batch)
        else:
            
            for i, player in enumerate(players_x):
                state3s_batch = playing_state3s_x[i][np.newaxis,:,:]
                state3s_fraud_batch = np_all_player_state3s_fraud[i][np.newaxis,:,:]
                discard_batch = np_discards[i][np.newaxis,:]
                #input shape=(batch(=1)-info)
                self.play_agent[player].learning_single_round(state3s_batch, state3s_fraud_batch, discard_batch)
        
        return


    def all_players_learning_full_round(self, round_states, round_actions, round_rewards, round_bs, round_next_state, round_id, last_winners): #, leading_trump, leading_suit): #round_bs not used yet
        # learning from full round(4 player in a round). 
        return


    def all_players_learning_single(self, game_state3s, game_actions, game_rewards, game_bs, game_next_states):
        #learning from single game
        if False == self.train:
            return

        #shape: dict[S,E,N,W](round(11)-batch-info)
        if True == self.flag_4in1:
            state3s_batch0 = np.concatenate((game_state3s[dc.Players.SOUTH][np.newaxis], game_state3s[dc.Players.EAST][np.newaxis], game_state3s[dc.Players.NORTH][np.newaxis], game_state3s[dc.Players.WEST][np.newaxis]), axis=0)
            actions_batch0 = np.concatenate((game_actions[dc.Players.SOUTH][np.newaxis], game_actions[dc.Players.EAST][np.newaxis], game_actions[dc.Players.NORTH][np.newaxis], game_actions[dc.Players.WEST][np.newaxis]), axis=0)
            rewards_batch0 = np.concatenate((game_rewards[dc.Players.SOUTH][np.newaxis], game_rewards[dc.Players.EAST][np.newaxis], game_rewards[dc.Players.NORTH][np.newaxis], game_rewards[dc.Players.WEST][np.newaxis]), axis=0)
            bs_batch0      = np.concatenate((game_bs[dc.Players.SOUTH][np.newaxis], game_bs[dc.Players.EAST][np.newaxis], game_bs[dc.Players.NORTH][np.newaxis], game_bs[dc.Players.WEST][np.newaxis]), axis=0)
            next_states_batch0 = np.concatenate((game_next_states[dc.Players.SOUTH][np.newaxis], game_next_states[dc.Players.EAST][np.newaxis], game_next_states[dc.Players.NORTH][np.newaxis], game_next_states[dc.Players.WEST][np.newaxis]), axis=0)
            state3s_batch1 = np.swapaxes(state3s_batch0, 0, 1)
            actions_batch1 = np.swapaxes(actions_batch0, 0, 1)
            rewards_batch1 = np.swapaxes(rewards_batch0, 0, 1)
            bs_batch1      = np.swapaxes(bs_batch0, 0, 1)
            next_states_batch1 = np.swapaxes(next_states_batch0, 0, 1)

            #shape=(round-player-batch-info)
            input_shape = state3s_batch1.shape
            
            #shape=(round-player*batch-info)
            state3s_batch = state3s_batch1.reshape((input_shape[0], -1, input_shape[3], input_shape[4]))
            actions_batch = actions_batch1.reshape((input_shape[0], -1))
            rewards_batch = rewards_batch1.reshape((input_shape[0], -1))
            bs_batch      = bs_batch1.reshape((input_shape[0], -1))
            next_states_batch = next_states_batch1.reshape((input_shape[0], -1, input_shape[3], input_shape[4]))
            
            _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_22, state3s_batch, state3s_batch1, actions_batch, actions_batch1)
            _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_21, actions_batch)

            #shape=(round-player*batch-info): (player*batch's order=[player0 batch]->[player1 batch]->[player2]->[player3])
            #whatever a player
            self.play_agent[dc.Players.SOUTH].learning_single_game(state3s_batch, actions_batch, rewards_batch, bs_batch, next_states_batch)
        else:
            #shape=(round-1*batch-info)
            for player in self.card_envs.players[0:4]:
                state3s_batch = game_state3s[player]
                actions_batch = game_actions[player]
                rewards_batch = game_rewards[player]
                bs_batch      = game_bs[player]
                next_states_batch   = game_next_states[player]
                
                _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_21, actions_batch)
                self.play_agent[player].learning_single_game(state3s_batch, actions_batch, rewards_batch, bs_batch, next_states_batch, player)


    def all_players_learning_multi(self):
        if False == self.train:
            return

        if True == self.flag_4in1:
            self.play_agent[dc.Players.SOUTH].learning_multi_games() #learning from stored trajectory
        else:
            for player in self.card_envs.players[0:4]:
                self.play_agent[player].learning_multi_games()
        



#unify inputs: 
#1.render
#set at start func default input. reset() transmit it to env
render_in_train = False  #True #
render_in_demo = False
#2.keep
#set at game.play_games(). refer it at loop in deal_card()
keep_batch_same_in_demo = False
#set at env() creation. impact card shuffle() in reset(). keep constant seed just before full_poker.init() and shuffle()
keep_env_same_in_demo = False
#3.print cycle
#set at game.play_games(). print game result cycle
print_game_result_cycle = 2048

class_short_name = { 'MC_q'            : agent_1.PlayAgentDNN_MC_q,
                     'MC_q_b'          : agent_1.PlayAgentDNN_MC_q_Behavior,
                     'MC_q_sm'         : agent_1.PlayAgentDNN_MC_q_Top3_Softmax,
                     'MC_q_avg'        : agent_1.PlayAgentDNN_MC_q_Top3_Uniform,
                     'MC_q_gd'         : agent_1.PlayAgentDNN_MC_q_guess,
                     'MC_q_gd_b'       : agent_1.PlayAgentDNN_MC_q_guess_Behavior,
                     'MC_q_f'          : agent_1.PlayAgentDNN_MC_q_fraud,
                     'MC_q_f_b'        : agent_1.PlayAgentDNN_MC_q_Behavior_fraud,
                     'MC_pi'           : agent_2.PlayAgentDNN_MC_pi,
                     'MC_pi_b'         : agent_2.PlayAgentDNN_MC_pi_Behavior,
                     'MC_pi_acc'       : agent_2.PlayAgentDNN_MC_pi_aac,
                     'MC_pi_acc_b'     : agent_2.PlayAgentDNN_MC_pi_aac_Behavior,
                     'MC_pi_acc_e'     : agent_2.PlayAgentDNN_MC_pi_aac_elibility,
                     'MC_pi_f'         : agent_2.PlayAgentDNN_MC_pi_fraud,
                     'MC_pi_f_b'       : agent_2.PlayAgentDNN_MC_pi_Behavior_fraud,
                     'MC_pi_acc_f'     : agent_2.PlayAgentDNN_MC_pi_aac_fraud,
                     'MC_pi_acc_f_b'   : agent_2.PlayAgentDNN_MC_pi_aac_Behavior_fraud,
                     'MC_pi_acc_e_f'   : agent_2.PlayAgentDNN_MC_pi_aac_elibility_fraud,
                     'TD_q'            : agent_3.PlayAgentDNN_TD_Expected_q,
                     'TD_q_b'          : agent_3.PlayAgentDNN_TD_Expected_q_Behavior,
                     'TD_q_f'          : agent_3.PlayAgentDNN_TD_Expected_q_fraud,
                     'TD_q_f_b'        : agent_3.PlayAgentDNN_TD_Expected_q_Behavior_fraud,
                     'C_MC_q'          : agent_6.PlayAgentCNN_MC_q,
                     'C_MC_q_b'        : agent_6.PlayAgentCNN_MC_q_Behavior, 
                     'C_MC_q_gc'       : agent_6.PlayAgentCNN_MC_q_guess,
                     'C_MC_q_gc_b'     : agent_6.PlayAgentCNN_MC_q_guess_Behavior, 
                     'Res_MC_q'        : agent_6.PlayAgentRes_MC_q,
                     'Res_MC_q_b'      : agent_6.PlayAgentRes_MC_q_Behavior, 
                     'C_MC_q_f'        : agent_6.PlayAgentCNN_MC_q_fraud,
                     'C_MC_q_f_b'      : agent_6.PlayAgentCNN_MC_q_Behavior_fraud,
                     'Res_MC_pi'       : agent_7.PlayAgentRes_MC_pi,
                     'Res_MC_pi_b'     : agent_7.PlayAgentRes_MC_pi_Behavior,
                     'C_MC_pi'         : agent_7.PlayAgentCNN_MC_pi,
                     'C_MC_pi_b'       : agent_7.PlayAgentCNN_MC_pi_Behavior,
                     'Res_MC_pi_acc'   : agent_7.PlayAgentRes_MC_pi_acc,
                     'Res_MC_pi_acc_b' : agent_7.PlayAgentRes_MC_pi_acc_Behavior,
                     'Res_MC_pi_f'     : agent_7.PlayAgentRes_MC_pi_fraud,
                     'Res_MC_pi_f_b'   : agent_7.PlayAgentRes_MC_pi_Behavior_fraud,
                     'C_TD_q'          : agent_11.PlayAgentCNN_TD_Expected_q,
                     'C_TD_q_b'        : agent_11.PlayAgentCNN_TD_Expected_q_Behavior,
                     }

id_game_id = 0
id_game_env_id     = id_game_id+1
id_game_agent_s    = id_game_env_id+1
id_game_agent_e    = id_game_agent_s+1
id_game_agent_n    = id_game_agent_e+1
id_game_agent_w    = id_game_agent_n+1
id_game_batch      = id_game_agent_w+1
id_game_games      = id_game_batch+1
id_game_keep_batch = id_game_games+1
id_game_demos      = id_game_keep_batch+1
id_game_flag_4in1  = id_game_demos+1

id_env_id = 0
id_env_play_reward_format = id_env_id+1
id_env_keep_env           = id_env_play_reward_format+1

id_agent_id = 0
id_agent_name       = id_agent_id+1
id_agent_net_conf   = id_agent_name+1
id_agent_lr         = id_agent_net_conf+1
id_agent_epsilon    = id_agent_lr+1
id_agent_gamma      = id_agent_epsilon+1


class InputParamters():
    def __init__(self):
        self.game_params    = cfg.game_config_sets
        self.env_params     = cfg.env_config_sets
        self.agent_params   = cfg.agent_config_sets
        self.discard_params = 0  #/TBD
        #self.read_params(2)  #TBD: tu share same global var name among multiple processess. the class MUST be "inited DATA" rather than a "run-time assigned BSS"
        return
    
    def read_params(self, selected_p_set_game):
        #############################
        # game parameters
        #############################
        game_parameters = self.game_params[selected_p_set_game]
        print("read param: game: ", selected_p_set_game, game_parameters)
        self.game_id = game_parameters[id_game_id]
        self.env_id = game_parameters[id_game_env_id]
        self.agent_class_s = game_parameters[id_game_agent_s] #class_short_name[]
        self.agent_class_e = game_parameters[id_game_agent_e] #class_short_name[game_parameters[id_game_agent_e]]
        self.agent_class_n = game_parameters[id_game_agent_n] #class_short_name[game_parameters[id_game_agent_n]]
        self.agent_class_w = game_parameters[id_game_agent_w] #class_short_name[game_parameters[id_game_agent_w]]
        self.batch_size = game_parameters[id_game_batch]
        self.games = game_parameters[id_game_games]
        self.keep_batch_same_in_train = game_parameters[id_game_keep_batch]  #True=one-env keep no change in many episodes. diff-env would be depends on param:keep_env_same_in_train
        self.demos = game_parameters[id_game_demos]
        self.for_in_one = game_parameters[id_game_flag_4in1]
    
        #############################
        # env parameters
        #############################
        selected_p_set_env = -1
        for i, params in enumerate(self.env_params):
            if params[id_env_id] == self.env_id:  #id
                selected_p_set_env = i
                break   #i
        if -1 == selected_p_set_env:
            print("worng config param: env id ....................... ", self.env_id)
            return False

        env_parameters = self.env_params[selected_p_set_env]
        print("read param: env: ", selected_p_set_env, env_parameters)
        self.play_reward_template = env_parameters[id_env_play_reward_format]
        self.keep_env_same_in_train = env_parameters[id_env_keep_env] #False=all env have same card set. used in single card deal batching
    
    
        #############################
        # play agent parameters
        #############################
        self.agent_id = []
        self.agent_class = []
        self.net_conf = []
        self.lr = []
        self.epsilon = []
        self.gamma0 = []
        self.play_file_name_e = []
        self.play_file_name_t = []

        if True == self.for_in_one:
            self.agent_class_e = self.agent_class_s
            self.agent_class_n = self.agent_class_s
            self.agent_class_w = self.agent_class_s
        
        for play_agent_id in [self.agent_class_s, self.agent_class_e, self.agent_class_n, self.agent_class_w]:
            selected_p_set_agent = -1
            for i, params in enumerate(self.agent_params):
                if params[id_agent_id] == play_agent_id:  #id
                    selected_p_set_agent = i
                    break   #i
            if -1 == selected_p_set_agent:
                print("worng config param: agent id ........................... ", play_agent_id)
                return False
            
            agent_parameters = self.agent_params[selected_p_set_agent]
            print("read param: agent: ", selected_p_set_agent, agent_parameters)
            self.agent_id.append(play_agent_id)
            self.agent_class.append(class_short_name[agent_parameters[id_agent_name]])
            self.net_conf.append(agent_parameters[id_agent_net_conf]) #TBD: in multi-processes in UNIX! now, it is wrong!!
            self.lr.append(agent_parameters[id_agent_lr])
            self.epsilon.append(agent_parameters[id_agent_epsilon])
            self.gamma0.append(agent_parameters[id_agent_gamma])
            #self.play_file_name_e.append('./results/play_e_' + str(self.game_id) + '_' + str(self.env_id) + '_' + str(play_agent_id) + '.h5')
            #self.play_file_name_t.append('./results/play_t_' + str(self.game_id) + '_' + str(self.env_id) + '_' + str(play_agent_id) + '.h5')
            self.play_file_name_e.append('./results/play_e_' + str(self.env_id) + '_' + str(play_agent_id) + '.h5')
            self.play_file_name_t.append('./results/play_t_' + str(self.env_id) + '_' + str(play_agent_id) + '.h5')

        return True


param_set = InputParamters()
tt0 = time.localtime() #time for training
tt1 = time.localtime()
def init_training(reload0, seed0_per_cpu, render, selected_p_set_game, net0=0, save_agent=True):
    #init training
    global tt0, tt1

    isgc = gc.isenabled() 
    threshold = gc.get_threshold() 
    print("init thred..............................: ", threshold, isgc)
    #gc.set_threshold(int(1e+6), int(1e+6), int(1e+6))
    #threshold = gc.get_threshold()     
    #print("init thred..............................: ", threshold)

    #load all config parameters again. in sub-process, it is needed. sub-process can't reuse same global var name, whose value is updated in main process in realtime rather than 'initial' moment
    if False == param_set.read_params(selected_p_set_game):
        return

    if 0 == param_set.games:
        return
    
    pid = psutil.Process().pid
    print("init_training: pid, net0 id flag0, seed", pid, id(net0), seed0_per_cpu)

    game_verify.checkpoints = game_verify.Game54_Online_Checkpoints(param_set.batch_size, disable=False)

    #############################
    # dicard agent
    #############################
    tc01 = time.time()
    fe = './results/q_dump_e_t1_e1_2019.h5'
    ft = './results/q_dump_t_t1_e1_2019.h5'
    print("init_training: discard files:", fe, ft )
    discard_agent0 = q_net.DiscardAgent_net6_Qmax2([[1024, 0.2], [256, 0.2]], fe, ft, reload=True)  #discard:dump
    
    #############################
    # play agents saving
    #############################
    if True == param_set.for_in_one :
        play_agent0_0 = param_set.agent_class[0](hidden_layers=param_set.net_conf[0], filename_e=param_set.play_file_name_e[0], filename_t=param_set.play_file_name_t[0], learning_rate=param_set.lr[0], epsilon=param_set.epsilon[0], gamma=param_set.gamma0[0], reload=reload0, net0_list=net0) #share net for fourin1 only
        play_agent0_1 = play_agent0_0
        play_agent0_2 = play_agent0_0
        play_agent0_3 = play_agent0_0
    else:
        #TBD: need a dynamic variable naming ..... locals() can't edit the value
        play_agents = {}
        if param_set.agent_id[0] not in play_agents.keys():
            play_agent0_0 = param_set.agent_class[0](hidden_layers=param_set.net_conf[0], filename_e=param_set.play_file_name_e[0], filename_t=param_set.play_file_name_t[0], learning_rate=param_set.lr[0], epsilon=param_set.epsilon[0], gamma=param_set.gamma0[0], reload=reload0, net0_list=0)
            play_agents[param_set.agent_id[0]] = play_agent0_0
        else:
            play_agent0_0 = play_agents[param_set.agent_id[0]]

        if param_set.agent_id[1] not in play_agents.keys():
            play_agent0_1 = param_set.agent_class[1](hidden_layers=param_set.net_conf[1], filename_e=param_set.play_file_name_e[1], filename_t=param_set.play_file_name_t[1], learning_rate=param_set.lr[1], epsilon=param_set.epsilon[1], gamma=param_set.gamma0[1], reload=reload0, net0_list=0)
            play_agents[param_set.agent_id[1]] = play_agent0_1
        else:
            play_agent0_1 = play_agents[param_set.agent_id[1]]

        if param_set.agent_id[2] not in play_agents.keys():
            play_agent0_2 = param_set.agent_class[2](hidden_layers=param_set.net_conf[2], filename_e=param_set.play_file_name_e[2], filename_t=param_set.play_file_name_t[2], learning_rate=param_set.lr[2], epsilon=param_set.epsilon[2], gamma=param_set.gamma0[2], reload=reload0, net0_list=0)
            play_agents[param_set.agent_id[2]] = play_agent0_2
        else:
            play_agent0_2 = play_agents[param_set.agent_id[2]]

        if param_set.agent_id[3] not in play_agents.keys():
            play_agent0_3 = param_set.agent_class[3](hidden_layers=param_set.net_conf[3], filename_e=param_set.play_file_name_e[3], filename_t=param_set.play_file_name_t[3], learning_rate=param_set.lr[3], epsilon=param_set.epsilon[3], gamma=param_set.gamma0[3], reload=reload0, net0_list=0)
            play_agents[param_set.agent_id[3]] = play_agent0_3
        else:
            play_agent0_3 = play_agents[param_set.agent_id[3]]

        '''
        play_agent0_0 = param_set.agent_class[0](hidden_layers=param_set.net_conf[0], filename_e=param_set.play_file_name_e[0], filename_t=param_set.play_file_name_t[0], learning_rate=param_set.lr[0], epsilon=param_set.epsilon[0], gamma=param_set.gamma0[0], reload=reload0, net0_list=0)
        play_agent0_1 = param_set.agent_class[1](hidden_layers=param_set.net_conf[1], filename_e=param_set.play_file_name_e[1], filename_t=param_set.play_file_name_t[1], learning_rate=param_set.lr[1], epsilon=param_set.epsilon[1], gamma=param_set.gamma0[1], reload=reload0, net0_list=0)
        play_agent0_2 = param_set.agent_class[2](hidden_layers=param_set.net_conf[2], filename_e=param_set.play_file_name_e[2], filename_t=param_set.play_file_name_t[2], learning_rate=param_set.lr[2], epsilon=param_set.epsilon[2], gamma=param_set.gamma0[2], reload=reload0, net0_list=0)
        play_agent0_3 = param_set.agent_class[3](hidden_layers=param_set.net_conf[3], filename_e=param_set.play_file_name_e[3], filename_t=param_set.play_file_name_t[3], learning_rate=param_set.lr[3], epsilon=param_set.epsilon[3], gamma=param_set.gamma0[3], reload=reload0, net0_list=0)
        '''
    
    #############################
    # envs
    #############################
    envs = ce.PokerEnvironment_6_1(7, param_set.batch_size, param_set.keep_env_same_in_train, seed0_per_cpu, input_play_rewards=param_set.play_reward_template)

    #############################
    # Game train
    #############################
    np.random.seed(seed0_per_cpu)  #for card shuffle. np is ued for shuffle
    tt0 = time.localtime()
    print("TRAINing start ..................................................", time.strftime("%Y-%m-%d_%H-%M-%S", tt0))
    game0 = Game54(param_set.batch_size, envs, discard_agent0, play_agent0_0, play_agent0_1, play_agent0_2, play_agent0_3, flag_4in1=param_set.for_in_one, train=True)
    tc02 = time.time()
    meas.perfs.game.perf_diff_0 += tc02 - tc01

    _ = game0.play_games(param_set.games, render, param_set.keep_batch_same_in_train, print_game_result_cycle)

    tt1 = time.localtime()
    print("TRAINing finsih..................................................", time.strftime("%Y-%m-%d_%H-%M-%S", tt1))

    #############################
    # saving. don't save agnet .h5 in sanity UT
    #############################
    if True == save_agent:
        if True == param_set.for_in_one :
            play_agent0_0.save_models()
        else:
            play_agent0_0.save_models()
            play_agent0_1.save_models()
            play_agent0_2.save_models()
            play_agent0_3.save_models()
    

    return discard_agent0.echo("hello1")


def demo(render, selected_p_set_game, seed0_per_cpu=13): #param_set0 isnot needed due to demo running in main process
    #demo after training
    print("DEMONSTRATing ..............................................", time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

    #load all config parameters again. in sub-process, it is needed. sub-process can't reuse same global var name, whose value is updated in main process in realtime rather than 'initial' moment
    if False == param_set.read_params(selected_p_set_game):
        return

    if 0 == param_set.demos:
        return

    batch_size = param_set.batch_size #1 #, demo one by one deal set
    
    game_verify.checkpoints = game_verify.Game54_Online_Checkpoints(param_set.batch_size)

    #############################
    # dicard agent
    #############################
    tc01 = time.time()
    discard_agent1 = q_net.DiscardAgent_net6_Qmax2([[1024, 0.2], [256, 0.2]], './results/q_dump_e_t1_e1_2019.h5', './results/q_dump_t_t1_e1_2019.h5', reload=True)  #discard:dump

    #############################
    # play agent
    #############################
    if True == param_set.for_in_one :
        play_agent1_0 = param_set.agent_class[0](hidden_layers=param_set.net_conf[0], filename_e=param_set.play_file_name_e[0], filename_t=param_set.play_file_name_t[0], reload=True)
        play_agent1_1 = play_agent1_0
        play_agent1_2 = play_agent1_0
        play_agent1_3 = play_agent1_0
    else:
        #TBD: need a dynamic variable naming ..... locals() can't edit the value
        play_agents = {}
        if param_set.agent_id[0] not in play_agents.keys():
            play_agent1_0 = param_set.agent_class[0](hidden_layers=param_set.net_conf[0], filename_e=param_set.play_file_name_e[0], filename_t=param_set.play_file_name_t[0], reload=True)
            play_agents[param_set.agent_id[0]] = play_agent1_0
        else:
            play_agent1_0 = play_agents[param_set.agent_id[0]]

        if param_set.agent_id[1] not in play_agents.keys():
            play_agent1_1 = param_set.agent_class[1](hidden_layers=param_set.net_conf[1], filename_e=param_set.play_file_name_e[1], filename_t=param_set.play_file_name_t[1], reload=True)
            play_agents[param_set.agent_id[1]] = play_agent1_1
        else:
            play_agent1_1 = play_agents[param_set.agent_id[1]]

        if param_set.agent_id[2] not in play_agents.keys():
            play_agent1_2 = param_set.agent_class[2](hidden_layers=param_set.net_conf[2], filename_e=param_set.play_file_name_e[2], filename_t=param_set.play_file_name_t[2], reload=True)
            play_agents[param_set.agent_id[2]] = play_agent1_2
        else:
            play_agent1_2 = play_agents[param_set.agent_id[2]]

        if param_set.agent_id[3] not in play_agents.keys():
            play_agent1_3 = param_set.agent_class[3](hidden_layers=param_set.net_conf[3], filename_e=param_set.play_file_name_e[3], filename_t=param_set.play_file_name_t[3], reload=True)
            play_agents[param_set.agent_id[3]] = play_agent1_3
        else:
            play_agent1_3 = play_agents[param_set.agent_id[3]]

        '''
        play_agent1_0 = param_set.agent_class[0](hidden_layers=param_set.net_conf[0], filename_e=param_set.play_file_name_e[0], filename_t=param_set.play_file_name_t[0], reload=True)
        play_agent1_1 = param_set.agent_class[1](hidden_layers=param_set.net_conf[1], filename_e=param_set.play_file_name_e[1], filename_t=param_set.play_file_name_t[1], reload=True)
        play_agent1_2 = param_set.agent_class[2](hidden_layers=param_set.net_conf[2], filename_e=param_set.play_file_name_e[2], filename_t=param_set.play_file_name_t[2], reload=True)
        play_agent1_3 = param_set.agent_class[3](hidden_layers=param_set.net_conf[3], filename_e=param_set.play_file_name_e[3], filename_t=param_set.play_file_name_t[3], reload=True)
        '''

    #############################
    # envs
    #############################
    envs = ce.PokerEnvironment_6_1(8, param_set.batch_size, keep_env_same_in_demo, seed0_per_cpu)

    #############################
    # Game demo
    #############################
    rd.seed(seed0_per_cpu)  #for card shuffle
    game1 = Game54(batch_size, envs, discard_agent1, play_agent1_0, play_agent1_1, play_agent1_2, play_agent1_3, flag_4in1=param_set.for_in_one, train=False)
    tc02 = time.time()
    meas.perfs.game.perf_diff_0 += tc02 - tc01

    game1.play_games(param_set.demos, render, keep_batch_same_in_demo, print_game_result_cycle)
    
    #############################
    # Analyze the win rate
    #############################
    measurement = meas.Measurements(game1.game_his)  #local instance
    win_games_sn, win_games_ew, levels_raised_sn, levels_raised_ew = measurement.win_rate(param_set)
    measurement.write_game_record(param_set, win_games_sn, win_games_ew, levels_raised_sn, levels_raised_ew)
    
    return discard_agent1.echo("hello2")

#self test with testcases
#selected_p_set_game=game config list的连续序号。 selected_p_set3=cmdline里的game ID,不连续
def UT(reload0, seed0_per_cpu, render, selected_p_set_game, test_auto_level, selected_p_set3=0):
    #init training
    print("UT ..................................................", time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    sanity_args = {}
    
    ################################
    # sanity 6 configs. 
    # checking: search for FAIL info in console output from online UT
    # sanity: [810, ~ 8xx]
    ################################
    if 3 == test_auto_level : # go to sanity
        sanity_args = {'start_p_set': 810,   #810
                       'selected_p_set3': selected_p_set3, 
                       'game_config': cfg.game_config_sets,
                       'init_training': init_training,
                       'id_game_id': id_game_id,
                       'render_in_train': render_in_train }
        '''
        selected_p_set = range(810, selected_p_set3+1)   #selected_p_set3 is the last p_set. [810, selected_p_set3]
        for i, params in enumerate(cfg.game_config_sets):
            if params[id_game_id] in selected_p_set:  #id
                selected_p_set2 = i
                seed = 13
                reload = False
                init_training(reload, seed, render_in_train, selected_p_set2, save_agent=False) #don't save agnet .h5
        
        return        
        '''
        
    ################################
    # offline UT MUST: [800, 801, 802]
    ################################

    #load all config parameters again. in sub-process, it is needed. sub-process can't reuse same global var name, whose value is updated in main process in realtime rather than 'initial' moment
    if False == param_set.read_params(selected_p_set_game):
        return

    #############################
    # dicard agent
    #############################
    fe = './results/q_dump_e_t1_e1_2019.h5'
    ft = './results/q_dump_t_t1_e1_2019.h5'
    print("init_training: discard files:", fe, ft)
    discard_agent2 = q_net.DiscardAgent_net6_Qmax2([[1024, 0.2], [256, 0.2]], fe, ft, reload=True)  #discard:dump

    #############################
    # play agents
    #############################
    if True == param_set.for_in_one :
        play_agent2_0 = param_set.agent_class[0](hidden_layers=param_set.net_conf[0], filename_e=param_set.play_file_name_e[0], filename_t=param_set.play_file_name_t[0], learning_rate=param_set.lr[0], epsilon=param_set.epsilon[0], gamma=param_set.gamma0[0], reload=reload0, net0_list=0) #doesn't use share net in UT
        play_agent2_1 = play_agent2_0
        play_agent2_2 = play_agent2_0
        play_agent2_3 = play_agent2_0
    else:
        play_agent2_0 = param_set.agent_class[0](hidden_layers=param_set.net_conf[0], filename_e=param_set.play_file_name_e[0], filename_t=param_set.play_file_name_t[0], learning_rate=param_set.lr[0], epsilon=param_set.epsilon[0], gamma=param_set.gamma0[0], reload=reload0, net0_list=0) # net0)
        play_agent2_1 = param_set.agent_class[1](hidden_layers=param_set.net_conf[1], filename_e=param_set.play_file_name_e[1], filename_t=param_set.play_file_name_t[1], learning_rate=param_set.lr[1], epsilon=param_set.epsilon[1], gamma=param_set.gamma0[1], reload=reload0, net0_list=0) #share net for 4in1 only
        play_agent2_2 = param_set.agent_class[2](hidden_layers=param_set.net_conf[2], filename_e=param_set.play_file_name_e[2], filename_t=param_set.play_file_name_t[2], learning_rate=param_set.lr[2], epsilon=param_set.epsilon[2], gamma=param_set.gamma0[2], reload=reload0, net0_list=0)
        play_agent2_3 = param_set.agent_class[3](hidden_layers=param_set.net_conf[3], filename_e=param_set.play_file_name_e[3], filename_t=param_set.play_file_name_t[3], learning_rate=param_set.lr[3], epsilon=param_set.epsilon[3], gamma=param_set.gamma0[3], reload=reload0, net0_list=0)
    
    #############################
    # envs
    #############################
    envs = ce.PokerEnvironment_6_1(9, param_set.batch_size, param_set.keep_env_same_in_train, seed0_per_cpu, input_play_rewards=param_set.play_reward_template)

    #############################
    # Game train
    #############################
    #stop for random shuffle in test_step_play_1card_massive(). np.random.seed(seed0_per_cpu)  #for card shuffle. np is used for shuffle
    game2 = Game54(param_set.batch_size, envs, discard_agent2, play_agent2_0, play_agent2_1, play_agent2_2, play_agent2_3, flag_4in1=param_set.for_in_one, train=True)

    UT = game_verify_off.Game54_Offline_Verify(game2, test_auto_level, class_short_name)
    fail_TCs = UT.run_all_cases(sanity_args)
    return discard_agent2.echo("hello3")

#################################
#
# prgaram startup and multi-task and shared DNN net (w)
#
#################################
def run_child(func_f, reload, seed, net0_list1, selected_p_set, cpu):
    proc = psutil.Process()  # get self pid
    print("PID: ", proc.pid)
    
    aff = proc.cpu_affinity()
    print("Affinity before: ", aff)
    
    proc.cpu_affinity(cpu)
    aff = proc.cpu_affinity()
    print("Affinity after:", aff)

    if net0_list1 != 0:
        print("run_child: net0 id ", id(net0_list1), net0_list1[0][0][0,1], net0_list1[0][2][50,50], net0_list1[0][4][126,2])
    
    if func_f == init_training:
        a, b = func_f(reload, seed, render_in_train, selected_p_set, net0=net0_list1)
    elif func_f == demo:
        a, b = func_f(render_in_demo, selected_p_set, seed0_per_cpu=seed)
    print("run_child: ", a, b)
    
def auto_get_net0_shape(class_name, net_conf):
    play_agent0 = class_name(hidden_layers=net_conf)
    #play_agent0 = agent_1.PlayAgentDNN_MC()
    nets = play_agent0.get_primay_net().get_weights()
    shapes = [net.shape for net in nets]
    return shapes

def main(argv):
    print("argv ", argv)
#-r init -m 3 -c 7 -s 13 -p 1
#-r init -p 4
#-r init -g 0 -p 4                        #only GPU=0. 'games'*10 if -g
#-r init -m 3 -c 7 -s 13 -t 50 -p 1000    #from p 1000, run 50 configs continuous, on CPU 5,6,7. #can't strictly stick a 'process' to a 'CPU'(in Debian. CentOS works). replaced by 'many_taskset2.py'
#-r resume -m 3 -c 7 -s 13 -p 0
#-r resume -m 3 -c 7 -s 13 -t 50 -p 0     #from p100000, run 50 configs, on CPU 5,6,7
#-r comp -m 3 -c 7 -a 50 -p 100000  #from p100000, run 50 configs, on CPU 5,6,7
#-r test -p 800
#-r test -u 1 -p 800  #massive
#-r test -u 2 -p 800  #auto
#-r test -u 3 -p 824  #sanity: training() for configs [810, selected_p_set3]
    multi_proces = 0
    cpu_back_start = 7
    seed_start = 13
    net0_list1 = 0
    selected_p_set = []
    perform_UT = False
    perform_comp = False
    test_auto_level = 0
    selected_p_set3 = 0
    enable_GPU = False
    from_to = 0

    ####################
    # get command line input params
    ####################
    try:
        opts, args = getopt.getopt(argv,"r:m:c:s:p:u:t:g:")
    except getopt.GetoptError:
        print("wrong input")
        return;

    try:
        for opt, arg in opts:
            print("arg: ",opt, arg)
            if opt == '-r':
                if arg == 'init' :
                    reload = False
                elif arg == 'resume' :
                    reload = True
                elif arg == 'comp' :
                    reload = True
                    perform_comp = True
                elif arg == 'test' :
                    reload = False
                    perform_UT = True
                else:
                    print("wrong -r input", opt, arg)
                    return

            if opt == '-m': #multi process: 0,1-27
                multi_proces = int(arg)

            if opt == '-c': #7 or 27
                cpu_back_start = int(arg)

            if opt == '-s':
                seed_start = int(arg)
            
            if opt == '-u':
                test_auto_level = int(arg)

            if opt == '-t':
                from_to = int(arg)

            if opt == '-g':  #actually, -g doesn't work. enable/disable the GPU, have to do so before main during imported pkgs start up
                gpu_id = int(arg)
                GPU_prepare(gpu_id)  #only GPU-0 is supported
                enable_GPU = True

            if opt == '-p':
                selected_p_set0 = arg.split(',')
                selected_p_set1 = [int(c) for c in selected_p_set0] #support only 1 param set
                selected_p_set = selected_p_set1[0]
                for i, params in enumerate(cfg.game_config_sets):
                    if params[id_game_id] == selected_p_set:  #id
                        selected_p_set2 = i
                        break   #i
                if True == perform_UT and 3 == test_auto_level:
                    #don't search the line ID when sanity
                    selected_p_set3 = selected_p_set
                
    except  ValueError:
        print("wrong input", opt, arg)
        return

    print("input set: multi-porcess + cpu start + seed start + param set + param set id + all games: ", multi_proces, cpu_back_start, seed_start, selected_p_set, selected_p_set2, from_to)

    #load all config parameters into main processor, which can't be propagated to sub-process
    if False == param_set.read_params(selected_p_set2):
        return
    
    #seed_offset = 0 #for single deal set training only
    seed_offset = int(np.random.random_sample() * 1000000) % 65535  #for general training with various seed
        
    #spawn: start from 'target'. fork: start from 'now on'
    set_start_method('spawn', force=True) #fork does not work, multi-processes can't be started
    ##############################
    # multiple processes startup
    ##############################
    if True == perform_UT:
        #so far default running from main processor
        #selected_p_set=cmd line gameID，不连续的. selected_p_set2=list里的序号, 0,1,2,3...连续的
        UT(reload, seed_start, render_in_train, selected_p_set2, test_auto_level, selected_p_set3=selected_p_set3)
        
    elif False == perform_comp:
        if from_to > 0: 
            #init or resume with 'from ... to ...', training() only
            #这里的multiple process是针对多个game,每个CPU run一个单独的game
            cpu = cpu_back_start
            seed = 13
            net0_list1 = 0
            procs = []
            print("YDL bundle training starting ", selected_p_set2, from_to)
    
            for i in range(selected_p_set2, selected_p_set2 + from_to,1):
                if False == param_set.read_params(i):  #redundent reading in main CPU
                    print("YDL: param read fail ", i)
                    return                
                
                if multi_proces > 0 :  #on LINUX
                    cpu_offset = (i - selected_p_set2) % multi_proces
                    cpu = [cpu_back_start-cpu_offset]
                    seed = seed_start+seed_offset
                    p = Process(target=run_child, args=(init_training, reload, seed, net0_list1, i, cpu))
                    print("YDL: game id start, cpu ", i, cpu_offset, cpu)
                    p.start()
                    procs.append(p)
                    if cpu_offset >= (multi_proces-1) or (selected_p_set2 + from_to-1) == i:
                        #waiting for sub-process complete
                        for p in procs:   #CPU pool would help. TBD!!
                            p.join()
                            print('YDL joined ', i)
                        procs = []
                else: #single CPU on windows
                    init_training(reload, seed, render_in_train, selected_p_set2)
            print("YDL: from to exit 'for' ", i)
            
        else:
            print("from_to init/resume without game sections")
            #goto return
         
        
    elif True == perform_comp : #competition with loaded .h5, no training, demo only
        if from_to > 0:
            cpu = cpu_back_start
            seed = 13
            reload = 'Nan'
            net0_list1 = 0
            procs = []
            print("competition starting ", selected_p_set2, from_to)
    
            accum_meas = meas.Measurements(0) #0= dummy his
    
            for i in range(selected_p_set2, selected_p_set2+from_to,1):
                if False == param_set.read_params(i):  #redundent reading in main CPU
                    return                
                accum_meas.add_game_id(param_set.game_id)
                
                if multi_proces > 0 :  #on LINUX
                    cpu_offset = (i - selected_p_set2) % multi_proces
                    cpu = [cpu_back_start-cpu_offset]
                    seed = seed_start+seed_offset
                    p = Process(target=run_child, args=(demo, reload, seed, net0_list1, i, cpu))
                    print("game id start ", i)
                    p.start()
                    procs.append(p)
                    if cpu_offset >= (multi_proces-1) or (selected_p_set2 + from_to-1) == i:
                        #waiting for sub-process complete
                        for p in procs:   #CPU pool would help. TBD!!
                            p.join()
                            print('joined comp')
                        procs = []
                else: #single CPU on windows
                    demo(render_in_demo, i, seed0_per_cpu=13)
            
            accum_meas.assemble_records()  #collect records.csv created by demo()
            accum_meas.analyze_competition_result()
            
        else: # competition == 0
            print("competition without game sections")
            #goto return
        
    elif multi_proces > 0 :  # training with multiple CPUs for single game
        #这里的multiple process是针对一个game,用多个CPU run同一个game.当network大时(>??M bytes)，效果不好,不建议用
        data_mgr = Manager()
        net0_lock1 = data_mgr.Lock() #需要整体lock,不是分项
        #net0_lock2 = data_mgr.Lock()
    
        # auto get the net shape
        if True == param_set.for_in_one and multi_proces > 1:
            #only support 4in1
            net0_shape_1 = auto_get_net0_shape(param_set.agent_class[0], param_set.net_conf[0]) #who to transmit a shared DNN net. SOUTH only
            net0 = [np.zeros(shape) for shape in net0_shape_1]

            net0_list1 = data_mgr.list()
            net0_list1.append(net0)
            net0_list1.append(net0_lock1)
            net0_list1.append(net0_shape_1) #read only
            
            #verify1. works!!!
            a, b, c = 7, 3, 7
            ydl = copy.deepcopy(net0_list1[0])
            ydl[0][0,1] = copy.deepcopy(a)
            ydl[2][50,50] = copy.deepcopy(b)
            ydl[4][126,2] = copy.deepcopy(c)
            net0_list1[0] = copy.deepcopy(ydl)
            
            #verify2: doesn't work
            '''
            a, b, c = 7, 3, 7
            net0_list1[0][0][0,1] = copy.deepcopy(a)
            net0_list1[0][2][50,50] = copy.deepcopy(b)
            net0_list1[0][4][126,2] = copy.deepcopy(c)
            '''
            print("main: net0 id ", id(net0_list1), net0_list1[0][0][0,1], net0_list1[0][2][50,50], net0_list1[0][4][126,2])
        else:
            multi_proces = 1  #it(>1) is meaningless if no share DNN net
            
        procs = []
        n_cpus = psutil.cpu_count()  #0-27
        print("total CPUs ", n_cpus)
        
        #did not verify the multiple params in mp mode
        #cpu_back_start = 7 #27
        #seed_start = 13
        for i in range(multi_proces): #support processes=1
            cpu = [cpu_back_start-i]
            seed = seed_start+seed_offset
            p = Process(target=run_child, args=(init_training, reload, seed, net0_list1, selected_p_set2, cpu))
            p.start()
            procs.append(p)
            time.sleep(2)

        for p in procs:
            p.join()
            print('joined training')
        
        time.sleep(2)
        ##################
        # deonstrate
        ###################
        #run on main processor
        demo(render_in_demo, selected_p_set2, seed0_per_cpu=13)

    else: #mp==0, training with single process. on main processor
        cpu = [cpu_back_start]
        seed = seed_start #+seed_offset, ignore offset in window debug
        '''
        ### alternative, UNIX only
        p = Process(target=run_child, args=(init_training, reload, seed, net0_list1, selected_p_set2, cpu))
        p.start()
        time.sleep(1)

        p.join()
        print('joined 1')
    
        '''
        ### original startup, #window only
        init_training(reload, seed, render_in_train, selected_p_set2)
        #'''    
        ##################
        # deonstrate
        ##################
        demo(render_in_demo, selected_p_set2, seed0_per_cpu=13)

        #CPU measurements available in main processor only
        meas.perfs.performance_report(selected_p_set)  #gameid=900, default disable csv writing

    return


if __name__ == "__main__":
    t0 = time.localtime()
   
    ######################################
    # temp test field
    ######################################
    mm0 = np.arange(144*5)
    mm1 = mm0.reshape((3,6,8,5))
    mm2 = mm1.reshape((3,-1, 5), order='C')
    mm3 = mm1.reshape((3,-1, 5))
    
    #ydl_gpu = meas.read_GPU_memory_usage(0)
    #ydl_gpu.total, ydl_gpu.used
    
    class base:
        def __init__(self):
            print('in base')
            print('out base')
            
    class A(base):
        def __init__(self):
            print('in A')
            super(A, self).__init__()
            print('out A')


    class B(A):
        def __init__(self):
            print('in B')
            super(B, self).__init__()
            print('out B')


    class C(B, A):
        def __init__(self):
            print('in C')
            super(C, self).__init__()
            print('out C')

    ydl = C()

    #os.popen('rm ./results/tmp.txt') #doesn't work in windows
    a1 = np.arange(11000).reshape(11,20,50)
    a2 = a1.swapaxes(1, 2)
    a3 = a1.reshape(-1, 2)
    np.set_printoptions(threshold=sys.maxsize)
    #print(a1)
    

    def zip_i(n):
        # zip loop is faster than i loop. 差距主要来自于对i索引的开销。
        # zip 本身比 range(n)耗时， 但处理中的i寻址的消耗的绝对时间更大。总体看，zip快，但不好量化
        # (inline loop + zip) is faster than (outline loop + zip)
        # (inline loop + range(i) is slower than (outline loop + range(i)) !!
        m = 100
        ydl1 = np.arange(n*m).reshape(-1, m)
        ydl2 = np.arange(1,2*n*m+1,2).reshape(-1, m)
        
        tc11 = time.time()
        ydl3 = [ydl1[i]+ydl2[i] for i in range(n)]
        tc12 = time.time()
        perf1 = tc12 - tc11

        tc31 = time.time()
        for i in range(n):
            ydl3 = ydl1[i]+ydl2[i]
        tc32 = time.time()
        perf3 = tc32 - tc31

        tc21 = time.time()
        ydl4 = [ydl11+ydl21 for ydl11, ydl21 in zip(ydl1, ydl2)]
        tc22 = time.time()
        perf2 = tc22 - tc21

        tc41 = time.time()
        for ydl11, ydl21 in zip(ydl1, ydl2):
            ydl4 = ydl11+ydl21
        tc42 = time.time()
        perf4 = tc42 - tc41

        perf5 = 0
        for i in range(n):
            tc51 = time.time()
            ydl4 = ydl1[i]+ydl2[i]
            tc52 = time.time()
            perf5 += tc52 - tc51
        
        perf6 = 0
        for ydl11, ydl21 in zip(ydl1, ydl2):
            tc61 = time.time()
            #ydl3 = ydl0 + ydl00
            ydl4 = ydl11 + ydl21
            tc62 = time.time()
            perf6 += tc62 - tc61

        return perf1, perf2, perf3, perf4, perf5, perf6
        
    #perf1, perf2, perf3, perf4, perf5, perf6 = zip_i(int(5e+6))
    #print(perf1, perf2, perf3, perf4, perf5, perf6)
    
    
    a1 = np.array([ 4, 5, 6, 1, 2, 3, 7, 86, 9, 15])
    a2 = np.array([ 2, 3, 5, 6, 9])
    ydl = np.s_[0:5:2]
    ydl2 = a1[ydl]

    '''
    import tensorflow.compat.v2 as tf
    from tensorflow import keras

    model = keras.Sequential()
    model.add(keras.layers.Dense(units=2, input_shape=(2,), activation=tf.nn.relu)) #(input_size[0],input_size[1])
    model.add(keras.layers.Dense(units=1, activation=None)) # 输出层
    optimizer = tf.optimizers.Adam(lr=0.001)
    model.compile(loss=tf.losses.mse, optimizer=optimizer) #, metrics=metrics) #['accuracy']), his['acc']
    model.summary()
    ydl = model.get_weights()
    
    #meas.get_size(model)

    session = tf.Session()
    keras.backend.set_session(session)
    keras.backend.clear_session()
    tf.reset_default_graph()
    '''
    # temp test field

    ######################################
    # offical starting ...
    ######################################
    main(sys.argv[1:])
    print("YDL ", time.strftime("%Y-%m-%d_%H-%M-%S", t0))
    print("YDL ", time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

    print("YDL train", time.strftime("%Y-%m-%d_%H-%M-%S", tt0))
    print("YDL train", time.strftime("%Y-%m-%d_%H-%M-%S", tt1))
