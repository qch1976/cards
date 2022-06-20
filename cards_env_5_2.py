#v4 based off v3m
#v4.1: optm the reset and saver
#v4.2: TD serise agent and optm for memory leak
#v5 based off v4.2 and aglin to deal_card v5 to support multiple cardset instances in the env.
#   players_cards[cardsets][4players]; np with 0=S, 1=E, 2=N, 3=W replace dict
#v5.1: state3s_fraud
#      fraud shape -> (4, 54*2). added 'played' info
#v5.2: state refactory as template(state3, state3_fraud, and coming states)
#      env has to store all kinds of state parallel since env support multiple agents
#      TO: state shape=(5,54), (4,54*2), (string)??
##     TO: state stack=(n,4,<state shape>), game (11,4,n,<state shape>)
#      reward alg changed. fmt[-1, 0.5, 1, 2] is obsolete
#      guess callback() added)
######################################
# ToDo list
# 1. init_all_player_cards() in reset. need sorting?
#    A: DONE. sorted once
# 2. is full poker needed? can a state fully present the poker status?
# 3. 4(players)*12(cards)*info numpy array rather than 4(dict)*12/18(cards)*info
#    A: done. split to 18+12*3 in players_cards[]
# 4. sort after assign in full_poker
#    A: NO. refer to 1)
# 5. where(np.array == np.array), does it work? refer to AKQJ_pattern_create()
#    A: understood
# 6. shorten the state1/2/3 time. PlayerView would be replaced by numpy state directly so that save the init state
#    in save/load, state3 takes 3/4 time cost
#              reset false	4 players 	optm nps	set bankers	    card status	AKQJ     	allow score  best discard	state2/3 init	save load
#    10000*10: 5.052054882	2.708721797	0.594206015	0.008118232	5.922522942	1.484858116	0.045517921	 4.786068757	    5.425851901	    18.7640694
#    10000*50: 17.65556153	11.71843457	2.230853319	0.015767495	18.82381193	3.07748119	0.06720225	 19.593889	    25.22135679	    83.82468367
#      note: numpy 44 time faster than df
#    A: player_cards use list rathen than dict. but little contribution to performance since [18,12,12,12] can't be converted to numpy
#    A: done for state2/3 initlization
# 7. if disconnect the reference in optm tables, may save time in load_status()
# 8. PR29.牌太好， in create_best_discards(), if dicard candidates is smaller than or close to 6(>6), the alg may create less than 6 cards for discarding, and while loop error happen with index>18
# 9. only banker=SOUTH. in env.init_all_player_cards(), can't recgnize the players in players_cards[1]. has to assume S=0, E=1, N=2, W=3. 
#    BTW, players_cards[0] regardless of the banker since it is the banker already
# 10. total_weights与total_importancy 合并
# 11. create_best_discard(): search AKQJ first then total score. incase: K is pop from remove list first, then AQJ would not be kept.
######################################

#######################################
# Limitations
# 1. the banker MUST be SOUTH due to self.players_cards[0]/[1]
# 2. round=2 only due to the important coeff is defined as 2 is 'round' only. and discard training are based off round=2
#    NOTE: the round_result() decision is right if 2 is NOT the 'round'
#    NOTE: the guess_callback() depends 'same round(=2)' crossing cardsets
# 3. ther trump is SPADE only due to guess_callback() simply use unique TRUMP set
#######################################
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment=None #default='warn'

import random as rd
import gc

import deal_cards_5_2 as dc
import meas_5_2 as meas
import game_verify_online_5_2 as game_verify

import time
import copy

GC_CYCLE = 2048
# support master=2 only. index by 'name'
card_importancy = np.array([0,      #0-invalid
                            13*1.4, #1
                            1,      #2
                            2,      #3
                            3,      #4
                            4,      #5
                            5,      #6
                            6,      #7
                            7,      #8
                            8,      #9
                            9,      #10
                            10*1.1, #11
                            11*1.2, #12
                            12*1.3, #13
                            15*3,   #BJ
                            16*3    #RJ 
                            ])  #reserve 14 for round trump

master_trump_importancy = { 'master'  : 14*3,
                            'regular' : 14*2 }

player_importancy_scope = [[487.2, 527.7], #max
                           [ 36,   68   ]  #min 
                           ]#12,  18 cards
player_deviation_scope = [ 0.65333, #log(4.501252),  #(1,0,0)~4.501252 (+1), max
                          -2.62468  #log(0.002373)]  #(3,3,3)~0.002373 (-1), min
                          ]

#偏离度， ~3=12/4, 实测结果 x=normal(2.88902, 1.35)的正态分布， log(x)
#每种花色的张数                   normal()        log()
norm_p_entropy_table = { 0: [0.029930572,   1.523884976],
                         1: [0.111021978,   0.954591037],
                         2: [0.237906818,   0.623593111],
                         3: [0.294515939,   0.530891196],
                         4: [0.210627322,   0.676485294],
                         5: [0.087021106,   1.060375403],
                         6: [0.020770095,   1.682561525],
                         7: [0.00286389,    2.543043658],
                         8: [0.000228128,   3.641821804],
                         9: [1.04979e-05,   4.978895961],
                         10: [2.79083e-07,  6.554266131],
                         11: [4.28615e-09,  8.367932312],
                         12: [3.80282e-11,  10.41989451],
                         13: [1.94916e-13,  12.71015271] }

IMPORTENCY_ID = 0
DEVIATION_ID  = 1
PRIORITY_ID   = 2



class PokerEnvironment:
    def update_discard_state3(self, cardsets, bankers, oindexes, fmt): #oindexes: length=6
        cardsets_index = cardsets[:, np.newaxis].repeat(6, axis=1).reshape(-1)
        bankers_index = bankers[:, np.newaxis].repeat(6, axis=1).reshape(-1)-1
        oindexes_index = oindexes.reshape(-1)
        self.state3s[cardsets_index, bankers_index, 0, oindexes_index] = fmt ##discard fmt=format[1]
        return

    def update_played_state3(self, cardsets, players, oindexes, fmt): #player: who play 1 card
        #(n, 0-3, 1-4, oindex) = fmt
        #repeat(4) due to all 4 players need updates
        cardsets_4 = cardsets.repeat(4)
        players_4 = np.array([0,1,2,3]*self.n).reshape(-1)
        oindexes_4 = oindexes.repeat(4)
        fmt_4 = fmt.repeat(4).astype('float16')
        players_offset = self.np_player_sequence_played[players-1].reshape(-1)
        self.state3s[cardsets_4, players_4, players_offset, oindexes_4] = fmt_4 ##discard fmt=format[1]
        return

    def update_inhand_state3(self, cardsets, players, oindexes, fmt): #player: who play 1 card
        #(n, 0, oindex) = fmt
        self.state3s[cardsets, players-1, 0, oindexes] = fmt ##discard fmt=format[1]
        return

    #must be invoked after env.init_all_players()
    def state3_reset2(self, fmt2, fmt3):
        # the dim of 5: [0] self inhand; [1] self played; [2] 下家; [3] 对家; [4] 上家
        self.state3s = np.zeros((self.n, 4, 5, 54))

        #####################
        # fmt2: regular
        #####################
        #player_card[0]=(n, 0-17, oindex)
        oindex = self.np_start_player_cards[:, :, dc.COL_OINDEX].reshape(-1)
        cardsets = np.arange(self.n)[:, np.newaxis].repeat(18, axis=1).reshape(-1)
        #state3=(n, (S), 0, oindex)
        self.state3s[cardsets, 0, 0, oindex] = fmt2
        
        #player_card[1]=(n, (E,N,W), 0-11, oindex)
        oindex = self.players_cards[1][:, :, :, dc.COL_OINDEX].reshape(-1)
        player_enw_0 = np.arange(1,4,1)[:,np.newaxis].repeat(12, axis=1).reshape(1,-1)
        player_enw = player_enw_0.repeat(self.n, axis=0).reshape(-1)
        cardsets = np.arange(self.n)[:, np.newaxis].repeat(12*3, axis=1).reshape(-1)
        #state3=(n, (E,N,W), 0, oindex)
        self.state3s[cardsets, player_enw, 0, oindex] = fmt2 ##discard fmt=format[1]

        ######################
        # fmt3: trump. length is diff can't use matrix numpy directly
        ######################
        trump_index_1 = (self.np_start_player_cards[:,:,dc.COL_TRUMP]==True)
        trump_index_2 = (self.players_cards[1][:,:,:,dc.COL_TRUMP]==True)
        for i in range(self.n):
            banker_trump_oindex = self.np_start_player_cards[i, trump_index_1[i], dc.COL_OINDEX]
            players_trump_oindex_1 = self.players_cards[1][i, 0, trump_index_2[i,0], dc.COL_OINDEX]
            players_trump_oindex_2 = self.players_cards[1][i, 1, trump_index_2[i,1], dc.COL_OINDEX]
            players_trump_oindex_3 = self.players_cards[1][i, 2, trump_index_2[i,2], dc.COL_OINDEX]
            self.state3s[i, 0, 0, banker_trump_oindex] = fmt3
            self.state3s[i, 1, 0, players_trump_oindex_1] = fmt3
            self.state3s[i, 2, 0, players_trump_oindex_2] = fmt3
            self.state3s[i, 3, 0, players_trump_oindex_3] = fmt3

        return

    #must be invoked after env.init_all_players()
    def state3_reset(self, fmt2, fmt3):

        self.state3s = np.zeros((self.n, 4, 5, 54))
        self.banker_pos = np.zeros((self.n,), dtype=int) #for guess only

        #####################
        # fmt2: regular
        #####################
        #player_card[0]=(n, 0-17, oindex)
        oindex = self.np_start_player_cards[:, :, dc.COL_OINDEX].reshape(-1)
        cardsets = np.arange(self.n)[:, np.newaxis].repeat(18, axis=1).reshape(-1)
        #state3=(n, (S), 0, oindex)
        self.state3s[cardsets, 0, 0, oindex] = fmt2
        
        #player_card[1]=(n, (E,N,W), 0-11, oindex)
        oindex = self.players_cards[1][:, :, :, dc.COL_OINDEX].reshape(-1)
        player_enw_0 = np.arange(1,4,1)[:,np.newaxis].repeat(12, axis=1).reshape(1,-1)
        player_enw = player_enw_0.repeat(self.n, axis=0).reshape(-1)
        cardsets = np.arange(self.n)[:, np.newaxis].repeat(12*3, axis=1).reshape(-1)
        #state3=(n, (E,N,W), 0, oindex)
        self.state3s[cardsets, player_enw, 0, oindex] = fmt2 ##discard fmt=format[1]

        ######################
        # fmt3: trump. length is diff can't use matrix numpy directly
        ######################
        trump_mask0 = (self.full_poker.np_cards[:, :, dc.COL_TRUMP] == True)
        #verify: repeat bline by line
        #trump_mask0[0,0] = False
        #trump_mask0[2,0] = False
        #trump_mask0[6,0] = False
        trump_mask = trump_mask0.repeat(4, axis=0).reshape(self.n, 4, 54)
        state3s = self.state3s[:,:,0,:]
        trump_locations = np.logical_and(state3s, trump_mask)
        state3s[trump_locations] = fmt3
        
        return

    def build_state3s_fraud(self, players_x=[]):
        inhand = copy.deepcopy(self.state3s[:, :, 0, :])
        played = copy.deepcopy(self.state3s[:, :, 1, :])
        state3s_fraud = np.concatenate((inhand, played), axis=2) #ordered by S,E,N,W. shape=[n,4,54*2]
        
        if len(players_x) > 0:
            #full players' state3s for fraud. put the current player at the first place
            all_player_state3s = [ np.roll(state3s_fraud[i], -(player-1), axis=0) for i, player in enumerate(players_x) ]
            np_all_player_state3s = np.array(all_player_state3s)
            _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_1, np_all_player_state3s, players_x, self.state3s)
        else:
            np_all_player_state3s = state3s_fraud

        return np_all_player_state3s  #shape=[n,4,54*2]. the player is put at the first position [0] in dim[4]
    
    def __init__(self, env_id, n, keep_env, seed):
        self.full_poker = dc.FullPokers(n)
        self.n = n
        self.states = [] #string state for banker, 'discard' uses it only
        self.cards_status = {}  #trump rating, non-trump rating, uneven of non-trump rating
        #TBD: only support E, N, W. 
        self.players_cards = []  #the 2nd item of 2 arraies, index by playername-2, [(self.n, 18, COL_END), (self.n, 3, 12, COL_END)]
                                 #since SOUTH the only banker, then [0]=S, [1]=[E,N,W]
        #TBD: SOUTH is always the banker!!??
        self.np_start_player_cards = np.zeros((self.n, 18, dc.COL_END)) #array, abstarct from players_cards.  (self.n, 18, COL_END)
        self.render = False
        
        self.state2s = np.array([]) #onehot-like state for banker, 'discard' uses it only
        self.df_saver = pd.DataFrame()
        self.oindex_pattern = np.array([])
        self.allowed_discarded_score = 0
        self.best_discards_oindex = []   #oindex ordered
        self.best_discards_priority_sorted = np.array([])  #pri ordered
        self.time_cost = np.zeros(8, dtype='float')  #for performance measure
        self.oindex_full = []
        self.id = env_id
        self.keep_env = keep_env
        self.seed = seed
        self.err_index_cnt = 0
        self.garbage_cnt = -1

        #version 4+, must start from SOUTH
        self.players = np.array([dc.Players.SOUTH, dc.Players.EAST, dc.Players.NORTH, dc.Players.WEST, dc.Players.SOUTH, dc.Players.EAST, dc.Players.NORTH])
        self.df_players_cards = {}  #start from SOUTH
        #self.state3s = []  #state3{} of class PlayerView

        #4=players(indexed by playername-1; 5=inhand+played; 54=onehot 54
        # the dim of 5: [0] self inhand; [1] self played; [2] 下家; [3] 对家; [4] 上家
        self.state3s = np.zeros((n, 4, 5, 54))  #state3s in numpy. index by 'player name'-1

        self.full_env_saver = []
        #self.full_env_saver_cmp = []
        self.total_weights = np.zeros((self.n, 54))  #for 牌大小比较, 考虑与total_importancy 合并
        
        #state3s(n,4,5,54) offset,      --> pos in dim of [5], [1-4]=played. [0]=inhand
        self.np_player_sequence_played = np.array([[1,4,3,2],     #S  |
                                                   [2,1,4,3],     #E  | 
                                                   [3,2,1,4],     #N  V
                                                   [4,3,2,1]])    #W, dim of [4]=player

        self.state3s_fraud = np.zeros((n, 4, 54*2))  #full and perfect info of 4 players 'inhand', 作弊模式
        self.banker_pos = np.zeros((n,), dtype=int) #for guess only. the banker position (0,1,2,3) in a round
        print("PokerEnvironment init checkpoint id: ", id(game_verify.checkpoints), game_verify.checkpoints.batch_size) #
        
    def save_status(self): #save data just after env.reset()
        self.full_env_saver = []
        #self.full_env_saver_cmp = []
        # what items will be changed after a game? A:
        #  self.full_poker.np_cards: discard, played
        #  self.np_start_player_cards: discard, played
        #  self.state1/2/3: discard, played
        # what items will NOT be changed after a game? A:
        #  self.np_trumps; np_non_trumps; np_discards; np_oindexs; oindex_pattern; oindex_full
        
        si = copy.deepcopy(self.full_poker.np_cards)  #[0]. ~=self.full_poker.cards, sync from df if needed
        self.full_env_saver.append(si)
        si = copy.deepcopy(self.players_cards)  #[1]
        self.full_env_saver.append(si)

        '''
        # ?? deepcopy would not refer to orignal data ???
        si = copy.deepcopy(self.np_start_player_cards) #[2]
        self.full_env_saver.append(si)
        si = copy.deepcopy(self.np_trumps)  #[3]
        self.full_env_saver.append(si)
        si = copy.deepcopy(self.np_non_trumps)  #[4]
        self.full_env_saver.append(si)
        si = copy.deepcopy(self.np_discards)  #[5]
        self.full_env_saver.append(si)
        si = copy.deepcopy(self.np_oindexs) #[6]
        self.full_env_saver.append(si)
        '''
        
        si = copy.deepcopy(self.states)   #[2]
        self.full_env_saver.append(si)
        si = copy.deepcopy(self.state2s)  #[3]
        self.full_env_saver.append(si)
        si = copy.deepcopy(self.state3s)  #[4]
        self.full_env_saver.append(si)
        si = copy.deepcopy(self.state3s_fraud)  #[5]
        self.full_env_saver.append(si)
        
        ### verify: stored data
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_10_1, self.np_start_player_cards, self.np_trumps, self.oindex_pattern, self.full_poker.np_cards)
        #print("ENV save_status checkpoints id: ", id(game_verify.checkpoints), game_verify.checkpoints.batch_size)

        return
    
    def load_status(self):
        self.full_poker.np_cards = copy.deepcopy(self.full_env_saver[0])  #~=self.full_poker.cards, sync from df if needed
        self.players_cards = copy.deepcopy(self.full_env_saver[1])  #~=self.full_poker.cards, sync from df if needed
        #in df: has to generate player cards from full_poker since keep same indexing crossing dfs
        #in numpy: no needs to keep the df index. but would setup the reference linkage
        self.np_start_player_cards = self.players_cards[0] #use player index rathen than player name
        '''
        for i in range(self.n):
            self.np_start_player_cards[i] = self.players_cards[i][self.full_poker.np_bankers[i]-1] #use player index rathen than player name
        '''
        
        trump_index = (self.np_start_player_cards[:,:,dc.COL_TRUMP]==True)
        trumps = [self.np_start_player_cards[i, trump_index[i], :] for i in range(self.n)]
        self.np_trumps = np.array(trumps, dtype=object) #CAN'T refer to self.np_start_xxx
        non_trump_index= ~trump_index
        non_trumps = [self.np_start_player_cards[i, non_trump_index[i], :] for i in range(self.n)]
        self.np_non_trumps = np.array(non_trumps, dtype=object) #CAN'T refer to self.np_start_xxx
        self.np_discards  = self.np_start_player_cards[:,:,dc.COL_DISCARDED] #CAN refer to self.np_start_xxx
        self.np_oindexs = self.np_start_player_cards[:,:,dc.COL_OINDEX]  #CAN refer to self.np_start_xxx

        self.states = copy.deepcopy(self.full_env_saver[2])
        self.state2s = copy.deepcopy(self.full_env_saver[3])
        #self.full_cards_onehot_like = copy.deepcopy(self.full_env_saver[3])
        self.state3s = copy.deepcopy(self.full_env_saver[4])
        self.state3s_fraud = copy.deepcopy(self.full_env_saver[5])
        
        ### verify: read stored data
        _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_10_2, self.np_start_player_cards, self.np_trumps, self.oindex_pattern, self.full_poker.np_cards)
        return
    
    def reset(self, start_players, keep=True, render=False): #keep: stay in same deal for initial debug
        gc.set_debug(gc.DEBUG_UNCOLLECTABLE) #|gc.DEBUG_COLLECTABLE) # | gc.DEBUG_LEAK)
        self.garbage_cnt += 1
        if 0 == self.garbage_cnt % GC_CYCLE:
            ydl0 = gc.garbage
            gbs = gc.collect()
            #se = 0
            se = meas.get_size(self) #block due to verify.checkpoint added. PR#40
            print("reset GC before: collect(), gbg.cnt, gc.gbg, full-size......................: ", gbs, self.garbage_cnt, len(ydl0), se)
            
        #print("PokerEnvironment base reset ID ", self.id)
        self.render = render
        tc11 = time.time()
        ######################
        #  clean cards, shuffle, set trump and round and banker
        ######################
        if True == keep : #stay in same card deal
            #stop in 4.1. _ = self.full_poker.load_cards_status()
            ######################
            #  load all required elements for optm
            ######################
            self.load_status()
            if render == True:
                self.full_poker.render(np.arange(self.n))
            return self.state3s, self.state2s, self.best_discards_oindex, self.np_start_player_cards, self.reward_times_10
        else: # free run
            if True == self.keep_env:
                rd.seed(self.seed) #tmp added for single card set training
            self.full_poker.__init__(self.n)
            self.full_poker.cards_shuffle()
            #create start_player list if diff start player is needed
            self.full_poker.set_trumps_banker(np.array([dc.CardSuits.SPADES]*self.n),
                                              np.array([2]*self.n),  #2 is the only 'round'. importancy is bigger and discard will not discard 2
                                              start_players)
                                     
            self.full_poker.cards_assign_to_player()
            if render == True:
                self.full_poker.render(np.arange(self.n))
            #stop in 4.1. self.full_poker.save_cards_status()

        tc12 = time.time()
        meas.perfs.env.perf_diff_0 += tc12- tc11
        
        ######################
        #  filter all 4 players
        ######################
        tc21 = time.time()
        #split to [18] ans [12,12,12]
        self.players_cards = self.init_all_player_cards()  #isolation. original full_poker can't be refered to by 'player_cards'. it is a copy
        
        ######################
        #  bankder for discarding
        ######################
        #start_player, would be extended to an array if needed.
        #must be loop=n rather than np slicing since list(18,12,12,12) inside
        self.np_start_player_cards = self.players_cards[0]
        '''
        for i in range(self.n):
            self.np_start_player_cards[i] = self.players_cards[i][start_player-1]
            #sort by=["suit", "name"]: optm needed ??
        '''
        tc22 = time.time()
        meas.perfs.env.perf_diff_1 += tc22- tc21

        ######################
        #  optm for start player card bufferring
        #
        # slice as a reference to original data rule
        # 1. ydl = aa[:,:,xx:yy]
        #    ydl[1,2] = yy. it is refernece
        #    ydl = ydl/2. it is not. ydl is replaced as new object
        #    ydl /= 2. it is. 左值必须是原始的引用
        # 2. ydl = aa[:,:,xx:yy]. it is a refernce
        #    ydl1 = ydl. ydl1 is still a reference
        # 3. ydl = aa[:,:,selected_index[]]. it is not a refernce
        ######################
        tc31 = time.time()
        trump_index = (self.np_start_player_cards[:,:,dc.COL_TRUMP]==True)
        trumps = [self.np_start_player_cards[i, trump_index[i], :] for i in range(self.n)]
        self.np_trumps = np.array(trumps, dtype=object) #CAN'T refer to self.np_start_xxx
        non_trump_index= ~trump_index
        non_trumps = [self.np_start_player_cards[i, non_trump_index[i], :] for i in range(self.n)]
        self.np_non_trumps = np.array(non_trumps, dtype=object) #CAN'T refer to self.np_start_xxx
        self.np_discards_xx  = self.np_start_player_cards[:,:,dc.COL_DISCARDED] #"self.np_discards", test no one use it. CAN refer to self.np_start_xxx
        self.np_oindexs = self.np_start_player_cards[:,:,dc.COL_OINDEX]  #CAN refer to self.np_start_xxx
        #gc=0
        tc32 = time.time()
        meas.perfs.env.perf_diff_2 += tc32- tc31

        ######################
        #  initial state by string(banker)
        ######################
        tc41 = time.time()
        #bankers = self.full_poker.np_bankers.reshape(-1,1)
        #temp stop to save debug time
        #self.states = self.full_poker.generate_str_key(np.arange(self.n), bankers)
        #gc=31
        tc42 = time.time()
        meas.perfs.env.perf_diff_3 += tc42- tc41
    
        ######################
        #  rating shape: (n, 4). remove dict
        ######################
        tc51 = time.time()
        self.total_weights = self.static_card_weights()
        
        player_importency, player_deviation, priorities = self.create_cards_status_evaluation()
        self.cards_status[IMPORTENCY_ID] = player_importency  #regularity player_importency, (0,1]
        self.cards_status[DEVIATION_ID]  = player_deviation
        self.cards_status[PRIORITY_ID]   = priorities  #data in memory in correct
        #gc=31
        tc52 = time.time()
        meas.perfs.env.perf_diff_4 += tc52- tc51
        #return
    
        ######################
        # pattern for search AKQJ
        ######################
        tc61 = time.time()
        self.oindex_pattern = self.create_AKQJ_pattern()
        tc62 = time.time()
        meas.perfs.env.perf_diff_5 += tc62- tc61
    
        ######################
        # max score can be discarded based on card status
        ######################
        tc71 = time.time()
        self.allowed_discarded_score = self.create_allowed_score(self.cards_status[IMPORTENCY_ID], self.cards_status[DEVIATION_ID])
        tc72 = time.time()
        meas.perfs.env.perf_diff_6 += tc72- tc71
        
        ######################
        # create best discard cards
        ######################
        tc81 = time.time()
        self.best_discards_oindex = self.create_best_discards()
        #self.best_discards_priority_sorted = self.priority_ordered_best_discards()
        tc82 = time.time()
        meas.perfs.env.perf_diff_7 += tc82 - tc81
        
        ######################
        #  initial state2 by onehot(banker)
        ######################
        tc91 = time.time()
        full_cards_onehot_like = np.zeros((self.n, 54))
        oindex = self.np_start_player_cards[:,:,dc.COL_OINDEX]
        self.oindex_full = oindex
        #print(id(oindex), id(self.oindex_full))
        oindex_x = np.arange(self.n)[:,np.newaxis].repeat(18, axis=1)
        full_cards_onehot_like[oindex_x.reshape(-1), oindex.reshape(-1)] = self.net_input_format[2]
        #select trump
        for i, trump in enumerate(self.np_trumps):
            trump_oindex = trump[:, dc.COL_OINDEX].astype(int) #batch=1, would auto become 'object'
            trump_oindex_x = np.array([i]).repeat(len(trump_oindex))
            full_cards_onehot_like[trump_oindex_x, trump_oindex] = self.net_input_format[3]
            
        self.state2s = full_cards_onehot_like
        #gc=31

        ######################
        #  init state3 by onehot(all 4 players)
        ######################
        self.state3_reset(self.net_input_format[2], self.net_input_format[3])
        self.state3s_fraud = self.build_state3s_fraud() #inside env, S is always at the 1st place in the state3s_fraud[]
        '''
        if not self.state3s:
            for i in range(self.n):
                state3 = {}
                for player in self.players[0:4]:
                    state3[player] = PlayerView(player, self.players_cards[i][player-1], self.players, fmt2=self.net_input_format[2], fmt3=self.net_input_format[3])
                self.state3s.append(state3)
        else:
            for i in range(self.n):
                for player in self.players[0:4]:
                    self.state3s[i][player].reset(player, self.players_cards[i][player-1], self.players, fmt2=self.net_input_format[2], fmt3=self.net_input_format[3])
        '''
        tc92 = time.time()
        meas.perfs.env.perf_diff_8 += tc92 - tc91

        ######################
        #  save all required elements for optm
        ######################
        tca1 = time.time()
        self.save_status()
        #verify
        self.load_status()

        tca2 = time.time()
        meas.perfs.env.perf_diff_9 += tca2- tca1
        
        return self.state3s, self.state2s, self.best_discards_oindex, self.np_start_player_cards, self.reward_times_10

        
    def create_cards_status_evaluation(self):
        ######################
        # importency 
        ######################
        #total_cards = 0  #non trump only

        total_importancy = self.sum_cards_importancy(self.np_start_player_cards)
        #total_importancy = self.sum_cards_importancy2(self.np_trumps)*1.5 #stop: trump or non-trump have diff length in dim2. have to 2 nested loop in sum_cards_importancy2()
        #total_importancy += self.sum_cards_importancy2(self.np_non_trumps)
                
        distance18 = player_importancy_scope[0][1] - player_importancy_scope[1][1]
        regularity18 = total_importancy - player_importancy_scope[1][1]
        total_importancy = regularity18/distance18
        
        ######################
        # deviation. sigmoid()效果不好,x趋近0太多
        ######################
        total_deviation2, non_trump_suits_template, np_non_trumps_len = self.calc_deviation()
        '''
        non_trumps_yes = (self.np_start_player_cards[:,:,dc.COL_TRUMP] == False)
        all_suits0 = self.np_start_player_cards[:,:,dc.COL_SUIT]
        #all non-trump suits. length would be diff crossing multiple cardsets
        all_suits = [all_suits0[i][non_trumps_yes[i]] for i in range(self.n)]
        
        trump_suit = np.repeat(self.full_poker.np_trumps[:,np.newaxis], 4, axis=1)
        regular_suits = np.array([[dc.CardSuits.SPADES, dc.CardSuits.HEARTS, dc.CardSuits.CLUBS, dc.CardSuits.DIAMONDS]] * self.n)
        non_trump_index = ~(regular_suits == trump_suit)
        non_trump_suits_template = regular_suits[non_trump_index].reshape(-1,3)
        
        #每个suit的长度
        suit_lens = []
        for all_suit, non_trump_candidates in zip(all_suits, non_trump_suits_template):
            c1 = (all_suit == non_trump_candidates[0])
            c2 = (all_suit == non_trump_candidates[1])
            c3 = (all_suit == non_trump_candidates[2])
            suit_lens.append([np.sum(c1), np.sum(c2), np.sum(c3)])
        
        np_non_trumps_len = np.array(suit_lens)
        total_deviation0 = np.sqrt(np.sum(np.square((np_non_trumps_len - 2.88902)), axis=1))

        total_deviation1 = total_deviation0/np.sum(np_non_trumps_len, axis=1)/np.sum(np_non_trumps_len, axis=1)  #non trump越长，值越小。 ^2突出了副牌长度的影响. #sum(non-trmp-len)=0, 意味着所有的主都在一人手上，p=10e-30->0
        total_deviation2 = np.log10(total_deviation1)
        '''
        distance18 = player_deviation_scope[0] - player_deviation_scope[1]
        regularity18 = total_deviation2 - player_deviation_scope[1] #min
        total_deviation = regularity18/distance18
        
        ######################
        # discard priority
        ######################
        #sort by 牌的数量, trump->longer->shorter
        sorted_length_index0 = np.repeat(np.arange(self.n)[:,np.newaxis], 3, axis=1).reshape(-1)
        sorted_length_index1 = np.argsort(-np_non_trumps_len, axis=1).reshape(-1)
        non_trump_suits_sorted = non_trump_suits_template[sorted_length_index0, sorted_length_index1].reshape(-1, 3)

        priorities = self.assign_discard_prority_to_card(non_trump_suits_sorted)

        return total_importancy, total_deviation, priorities #dict没有顺序，即使用sorted_suit_lens重建dict，也不行

    def calc_deviation(self):
        non_trumps_yes = (self.np_start_player_cards[:,:,dc.COL_TRUMP] == False)
        all_suits0 = self.np_start_player_cards[:,:,dc.COL_SUIT]
        #all non-trump suits. length would be diff crossing multiple cardsets
        all_suits = [all_suits0[i][non_trumps_yes[i]] for i in range(self.n)]
        
        trump_suit = np.repeat(self.full_poker.np_trumps[:,np.newaxis], 4, axis=1)
        regular_suits = np.array([[dc.CardSuits.SPADES, dc.CardSuits.HEARTS, dc.CardSuits.CLUBS, dc.CardSuits.DIAMONDS]] * self.n)
        non_trump_index = ~(regular_suits == trump_suit)
        non_trump_suits_template = regular_suits[non_trump_index].reshape(-1,3)
        
        #每个suit的长度
        suit_lens = []
        for all_suit, non_trump_candidates in zip(all_suits, non_trump_suits_template):
            c1 = (all_suit == non_trump_candidates[0])
            c2 = (all_suit == non_trump_candidates[1])
            c3 = (all_suit == non_trump_candidates[2])
            suit_lens.append([np.sum(c1), np.sum(c2), np.sum(c3)])
        
        np_non_trumps_len = np.array(suit_lens)
        total_deviation0 = np.sqrt(np.sum(np.square((np_non_trumps_len - 2.88902)), axis=1))

        total_deviation1 = total_deviation0/np.sum(np_non_trumps_len, axis=1)/np.sum(np_non_trumps_len, axis=1)  #non trump越长，值越小。 ^2突出了副牌长度的影响. #sum(non-trmp-len)=0, 意味着所有的主都在一人手上，p=10e-30->0
        total_deviation2 = np.log10(total_deviation1)
        return total_deviation2, non_trump_suits_template, np_non_trumps_len

    #optm: n>=50, faster than loop(in xxx2()) 20+ times
    def assign_discard_prority_to_card(self, non_trump_suits_sorted):
        player_classes = np.zeros((self.n, 18))
        priority_class = 300
        trump_slice = self.np_start_player_cards[:,:,dc.COL_TRUMP]
        name_slice = self.np_start_player_cards[:,:,dc.COL_NAME]
        suit_slice = self.np_start_player_cards[:, :, dc.COL_SUIT]
        
        round_18 = np.repeat(self.full_poker.np_rounds[:,np.newaxis], 18, axis=1)
        suit_18 = np.repeat(self.full_poker.np_trumps[:,np.newaxis], 18, axis=1)
        
        #########################
        # trump part
        #########################
        trump_yes = (trump_slice == True)    #所有的主
        round_trumps_yes = (name_slice == round_18)  #常主
        suit_trump_yes = (suit_slice == suit_18)     #花色主， 有overlap to常主
        suit_round_trump_yes = round_trumps_yes & suit_trump_yes #主2
        trump_no_round_yes = trump_yes & ~round_trumps_yes  #纯花色主，去掉2

        if suit_round_trump_yes.any(): #主2
            suit_round_trump_yes_index = np.where(True==suit_round_trump_yes)
            player_classes[suit_round_trump_yes_index[0], suit_round_trump_yes_index[1]] = master_trump_importancy['master'] - master_trump_importancy['regular'] #'priority_class' and 'regular' will be added back later
        
        if round_trumps_yes.any():  #副2
            round_trumps_yes_index = np.where(True==round_trumps_yes)
            player_classes[round_trumps_yes_index[0], round_trumps_yes_index[1]] += priority_class + master_trump_importancy['regular']
            
        if trump_no_round_yes.any(): #纯花色主
            trump_no_round_yes_index = np.where(True==trump_no_round_yes)
            trump_no_round_yes_name = name_slice[trump_no_round_yes_index]
            player_classes[trump_no_round_yes_index] = priority_class + card_importancy[trump_no_round_yes_name]

        #########################
        # non-trump part
        #########################
        #extend to 3 sorted suits. shape(3, self.n, 18). 
        non_trump_suits_sorted = non_trump_suits_sorted.swapaxes(0,1)
        non_trump_suits_sorted_3_18 = non_trump_suits_sorted[:,:,np.newaxis].repeat(18, axis=2) #shape=(3,n,18). 3 sorted suits, n cardsets, repeat 18 cards in banker
        suit_slice_3_18 = suit_slice[np.newaxis,:,:].repeat(3, axis=0)
        trump_slice_3_18 = trump_slice[np.newaxis,:,:].repeat(3, axis=0)
        name_slice_3_18 = name_slice[np.newaxis,:,:].repeat(3, axis=0)

        suit_in_sorted = (suit_slice_3_18 == non_trump_suits_sorted_3_18)  #按sorted副牌花色选， 有overlap to常主
        trump_inside_suit_yes = (trump_slice_3_18 == False)   #副牌
        suit_index0 = suit_in_sorted & trump_inside_suit_yes 
        suit_index = np.where(suit_index0==True)   #纯花色副牌，去掉2
        if (0,) != suit_index[0].shape:   #可能没有某花色
            suit_oindex_name = name_slice_3_18[suit_index]  #indexed by 3dim suit_index, output to 1 dim
            ydl1 = (2-suit_index[0])*100  #suit_index[0]=0, 1st sorted suit; suit_index[0]=1, 2nd sorted suit; ..
            ydl2 = card_importancy[suit_oindex_name]
            player_classes[suit_index[1], suit_index[2]] = ydl2 + ydl1 #card_importancy[suit_oindex_name]  
            
        return player_classes


    #optm: 3+ times faster than loop with separated trump and non-trump @n>=50
    def sum_cards_importancy(self, np_cards):  #input: all cardset
        total_importancy = np.zeros((self.n))
        rounds = np.repeat(self.full_poker.np_rounds[:,np.newaxis], 18, axis=1)  #name
        trump_suits = np.repeat(self.full_poker.np_trumps[:,np.newaxis], 18, axis=1)  #name

        #trump_yes = np_cards[:,:,dc.COL_TRUMP]
        card_names = np_cards[:,:,dc.COL_NAME]
        round_trump_yes = (card_names == rounds)
        suit_trumps_yes = (np_cards[:,:,dc.COL_SUIT] == trump_suits)
        round_suit_trump_yes = round_trump_yes & suit_trumps_yes

        #master 2
        round_suit_trump_yes_index = np.where(round_suit_trump_yes==True)[0]  #cardset index only
        total_importancy[round_suit_trump_yes_index] += master_trump_importancy['master'] - master_trump_importancy['regular'] #will be added in regular
        round_name = self.full_poker.np_rounds[round_suit_trump_yes_index] ##will be added in normal card
        total_importancy[round_suit_trump_yes_index] -= card_importancy[round_name]
        
        #regular 2
        round_trump_lens = np.sum(round_trump_yes, axis=1)
        round_trump_yes_index = np.where(round_trump_lens > 0)[0]
        total_importancy[round_trump_yes_index] +=  round_trump_lens[round_trump_yes_index] * master_trump_importancy['regular']

        #suit trump
        suit_trumps_yes_index = np.where(suit_trumps_yes==True)
        added_importancy = card_importancy[card_names[suit_trumps_yes_index[0], suit_trumps_yes_index[1]]]
        for i, ii in enumerate(suit_trumps_yes_index[0]) :
            total_importancy[ii] += added_importancy[i]*1.5 #add weight to suit trump comparing to normal. guaranttee: suit A < regular 2
        #total_importancy[suit_trumps_yes_index[0]] += added_importancy # doesn't work. only last card in each cardset is added
        
        #normal and BJ, RJ
        round_trump_and_suit_trump_yes = round_trump_yes | suit_trumps_yes
        trump_no_index = np.where(round_trump_and_suit_trump_yes==False)
        added_importancy = card_importancy[card_names[trump_no_index[0], trump_no_index[1]]]
        for i, ii in enumerate(trump_no_index[0]) :
            total_importancy[ii] += added_importancy[i] 
        
        return total_importancy

    
    def create_AKQJ_pattern(self):
        master_rounds = self.full_poker.np_rounds-1 #round=name; name-1 = oindex in suit 1
        
        #verify
        #master_rounds[[0,1]] = 11
        #master_rounds[[4,6]] = 7
        #master_rounds[[9]] = 0

        ################ pattern in AKQJ1098
        #(4,7) => (10,4,7)
        default_oindex_pattern0 = np.array([0, 12, 11, 10, 9, 8, 7])  #oindex
        default_oindex_pattern1 = np.vstack((default_oindex_pattern0, 
                                             default_oindex_pattern0+13,
                                             default_oindex_pattern0+13*2,
                                             default_oindex_pattern0+13*3))
        default_oindex_pattern = default_oindex_pattern1[np.newaxis,:,:].repeat(self.n, axis=0)
        
        #主2的oindex[4]
        #(4,1)=>(10,4,1)
        master_trump_oindex = np.array([master_rounds, 
                                        master_rounds+13, 
                                        master_rounds+13*2, 
                                        master_rounds+13*3]).T[:,:,np.newaxis]
        index_in_default_oindex_pattern = np.where(default_oindex_pattern==master_trump_oindex)  #where((10,4,7)==(10,4,1)), 这种shape包含的形状，可以做 '==' comparison

        if 0 == len(index_in_default_oindex_pattern[0]):
            #不在前7个大牌中
            oindex_pattern = list(default_oindex_pattern)
        else:
            cardset_in_default_oindex_pattern = list(index_in_default_oindex_pattern[0])
            oindex_pattern = []
            for i, one_pattern in enumerate(default_oindex_pattern):
                #oindex_pattern = np.delete(default_oindex_pattern, np.s_[index_in_pattern:index_in_pattern+1:1], axis=1)  #s_[起点：终点：间隔]
                #oindex_pattern = np.delete(default_oindex_pattern, index_in_default_oindex_pattern0)  #s_[起点：终点：间隔]
                if i in cardset_in_default_oindex_pattern:
                    index_in_pattern = cardset_in_default_oindex_pattern.index(i)
                    oindex_pattern.append(np.delete(one_pattern, index_in_default_oindex_pattern[2][index_in_pattern], axis=1))
                else:
                    oindex_pattern.append(one_pattern)
                    
        return oindex_pattern   #would be a list rather than a numpy array since the length of every item could be diff, (4,6) or (4,7)

    def create_allowed_score(self, cards_status_importency, cards_status_deviation):
        max_score = 0
        coeff0 = cards_status_importency * cards_status_deviation
        coeff0_mean = np.mean(coeff0)
        coeff1  = (coeff0 - coeff0_mean)*1.5 + coeff0_mean #reshape
        coeff = np.where(coeff1<0, 0, coeff1)
        max_score = 100 * coeff + 5  #total 100 score in a full poker, 5 personal compensation
        
        #补偿分太多
        players_score = np.sum(self.np_start_player_cards[:, :, dc.COL_SCORE], axis=1)
        score_50 = (players_score>=50)
        max_score[score_50] += 10  #

        score_70 = (players_score>=70)
        max_score[score_70] += 10  #

        return max_score
    
    def create_best_discards(self): #must be invoked after env.set_trump_banker()
        best_discards_oindex = []

        #columns=['oindex', 'suit', 'name', 'score', 'trumps', 'who', 'played', 'discarded'])
        np_priority_sorted_index = np.argsort(self.cards_status[PRIORITY_ID],axis=1)
        sorted_index_0 = np.arange(self.n)[:,np.newaxis].repeat(18,axis=1).reshape(-1)
        sorted_index_1 = np_priority_sorted_index.reshape(-1)
        np_start_player_cards = self.np_start_player_cards[sorted_index_0, sorted_index_1].reshape(self.n, 18, 8)
        #np_priorities = self.cards_status[PRIORITY_ID][sorted_index_0, sorted_index_1].reshape(self.n, -1)
        #np_remove_flags = np.zeros(np_priorities.shape)
        
        biggest_oindex_pattern = np.array([one_pattern[:,0] for one_pattern in self.oindex_pattern])

        #test temp , [[4, 9, 12+13+13+13], [0, 12, 11, 13, 25, 26], [9+13]
        #np_priorities[:, 0] = np.array([4, 9, 48, 26, 25, 13, 11, 12, 0, 43, 49, 51, 34, 8, 16, 47, 14, 1])
        
        for i, np_start_player_card in enumerate(np_start_player_cards): #loop for al cardsets
            best_discards_seq = [] #store and process the index of the 18 cards
            discards_index = 0  #index to orignal banker's 18 cards. keep inceasing
            
            while True: #loop from the smallest pri to bigger
                best_discards_seq.append(discards_index)
                try: 
                    oindex = np_start_player_card[discards_index, dc.COL_OINDEX]
                except IndexError:  #index 18 is out of bounds for axis 0 with size 18
                    print("IndexError: index 18 is out of bounds for axis 0 with size 18")
                    print(i, best_discards_seq)
                    print(np_start_player_card)
                    print(self.allowed_discarded_score[i])
                    print(biggest_oindex_pattern[i])
                    #exit(2)
                    # temp fix PR#36. random assign the index of discard cards
                    best_discards_seq = np.random.choice(np.arange(18), size=6, replace=False)
                    break  #while, would stop .... later
                    
                in_biggest_pattern_index = np.where(biggest_oindex_pattern[i] == oindex)
                if 0 == len(in_biggest_pattern_index[0]) or np_start_player_card[discards_index, dc.COL_TRUMP] == True:  #trump need not to search pattern
                    #current card is NOT in the 'biggest' list
                    if 6 <= len(best_discards_seq):
                        if np.sum(np_start_player_card[best_discards_seq, dc.COL_SCORE]) > self.allowed_discarded_score[i]:
                            #if total score exceeed the allowed ...
                            for candidate_index in best_discards_seq[::-1]:
                                #remove from higher priority
                                if 0 < np_start_player_card[candidate_index, dc.COL_SCORE]:
                                    best_discards_seq.remove(candidate_index)
                                    if np.sum(np_start_player_card[best_discards_seq, dc.COL_SCORE]) > self.allowed_discarded_score[i]:
                                        continue
                                    else:
                                        break
                        else:
                            break #here, a best oindex set generated. break from 'while'
                        
                else:
                    #the top oindex is one of the biggest cards. pop() some
                    _ = best_discards_seq.pop()  #it is the biggest one
                    for biggest_seq in self.oindex_pattern[i][in_biggest_pattern_index[0], 1:].reshape(-1):
                        
                        try:
                            remove_candidate_seq = best_discards_seq.pop()
                            remove_candidate = np_start_player_card[remove_candidate_seq, dc.COL_OINDEX]
                        except IndexError: #pop from empty list
                            break  #empty list, break
                        
                        if remove_candidate != biggest_seq:
                            best_discards_seq.append(remove_candidate_seq)  #add back
                            break
                discards_index += 1
            best_discards_oindex.append(np_start_player_card[best_discards_seq, dc.COL_OINDEX])

        best_discards_oindex = np.array(best_discards_oindex)
        #best_discards_oindex.sort(axis=1) #need not, argsort() above
        #print(best_discards_oindex)

        return best_discards_oindex

    
    def manual_reward_estimation(self, discarded_card_oindex): # for step with single card
        #[16, 18, 34, 41, 42, 43]  #oindex
        best_discards_oindex = self.best_discards_oindex
        
        if discarded_card_oindex in best_discards_oindex :
            reward = self.reward_times_10 #50
        else:
            reward = self.reward_times_1  #5
        return reward

    def get_best_discards_priority_sorted(self):
        oindex_pri_sorted = self.best_discards_priority_sorted.copy()
        return oindex_pri_sorted

    def init_all_player_cards(self):
        #input seq: [0]=S, [1]=E, [2]=N, [3]=W. from game beginning
        #winner or banker at first place
        players_cards = self.full_poker.get_player_cards_by_players(np.arange(self.n), np.array([self.players[0:4]]*self.n))
        #output: shape(n, 4(S,E,N,W), 54)
        np_players_cards = players_cards #return value is numpy

        bankers_cards = list(np_players_cards[:,0])  #SOUTH is the only banker
        np_bankers_cards = np.array(bankers_cards)

        #np.array(list[:, 1:4]) dowsn't work. the shape is (self.n, 3) rather than (self.n, 3, 12, 8)
        np_non_bankers_cards_1 = np.array(list(np_players_cards[:,1]))[:,np.newaxis,:,:]
        np_non_bankers_cards_2 = np.array(list(np_players_cards[:,2]))[:,np.newaxis,:,:]
        np_non_bankers_cards_3 = np.array(list(np_players_cards[:,3]))[:,np.newaxis,:,:]
        np_non_bankers_cards = np.concatenate((np_non_bankers_cards_1, np_non_bankers_cards_2, np_non_bankers_cards_3), axis=1)

        return [np_bankers_cards, np_non_bankers_cards]

    def build_np_state3_0(self, player):
        player_np_state3 = np.vstack([self.state3[player].inhand_oh, self.state3[player].played_oh])
        return player_np_state3

    def static_card_weights(self):  #all n*54 cards, trump/round would be diff in every cardset
        #for 出牌大小的比较 in a round
        # 0: basis:         1 -13 *3, static. A=13, K=12, ..., 2=1
        # 1: leading suit:  14-26 *1, dynamic
        # 2: suit trump:    27-39 *1, static. A=39, K=38, ..., 2=27
        # 3: nomral 2:      40    *3, statis
        # 4: leading 2:     41    *1, dynamic
        # 5: master 2:      42    *1, static
        # 6: BJ:            43    *1, statis
        # 7: RJ:            44    *1, statis
        total_weights = np.zeros((self.n, 54))  #(54) indexed by oindex
        rounds_54 = np.repeat(self.full_poker.np_rounds[:,np.newaxis], 54, axis=1)  #name
        trump_suits_54 = np.repeat(self.full_poker.np_trumps[:,np.newaxis], 54, axis=1)

        card_names = self.full_poker.np_cards[:,:,dc.COL_NAME]
        round_trump_yes = (card_names == rounds_54)
        A_yes = (card_names == 1)
        suit_trumps = self.full_poker.np_cards[:,:,dc.COL_SUIT]
        suit_trumps_yes = (suit_trumps == trump_suits_54)
        round_suit_trump_yes = round_trump_yes & suit_trumps_yes
        suit_trump_A_yes = (suit_trumps_yes & A_yes)
        
        #normal.
        round_trump_and_suit_trump_no = ~(round_trump_yes | suit_trumps_yes)
        total_weights[round_trump_and_suit_trump_no] = card_names[round_trump_and_suit_trump_no]-1  #class 0
        total_weights[A_yes] = 13  #class 0. here would overlap the trump A and BJ, RJ, but will be correct in 'suit trump' part

        #suit trump
        suit_trumps_yes_index = np.where(suit_trumps_yes==True)
        total_weights[suit_trumps_yes_index[0], suit_trumps_yes_index[1]] = card_names[suit_trumps_yes_index[0], suit_trumps_yes_index[1]] + 25 #class 2
        total_weights[suit_trump_A_yes] += 13  #class 0

        #regular 2. replacing 'suit trump'
        round_trump_yes_index = np.where(round_trump_yes == True)
        total_weights[round_trump_yes_index[0], round_trump_yes_index[1]] = 40 #class 3

        #master 2. replacing 'regular 2'
        round_suit_trump_yes_index = np.where(round_suit_trump_yes==True)  #cardset index only
        total_weights[round_suit_trump_yes_index[0], round_suit_trump_yes_index[1]] = 42 #class 5

        
        # and BJ, RJ
        total_weights[:, 52] = 43 #class 6
        total_weights[:, 53] = 44 #class 7
        
        return total_weights

    def dynamic_card_weights(self, round_played_cards):  #all n*54 cards
        #for 出牌大小的比较 in a round
        # 0: basis:         1 -13 *3, static. A=13, K=12, ..., 2=1
        # 1: leading suit:  14-26 *1, dynamic
        # 2: suit trump:    27-39 *1, static. A=39, K=38, ..., 2=27
        # 3: nomral 2:      40    *3, statis
        # 4: leading 2:     41    *1, dynamic
        # 5: master 2:      42    *1, static
        # 6: BJ:            43    *1, statis
        # 7: RJ:            44    *1, statis
        total_weights = self.total_weights
        card_index_0 = np.arange(self.n)[:,np.newaxis].repeat(4, axis=1).reshape(-1)
        
        full_suits = self.full_poker.np_cards[:, :, dc.COL_SUIT]
        played_cards_suit = full_suits[card_index_0, round_played_cards.reshape(-1)].reshape(self.n, 4)
        leading_cards_suit = played_cards_suit[:,0]
        all_leading_suit_yes = (played_cards_suit==leading_cards_suit[:,np.newaxis])  #shape(n,4)==shape(n,1) works well. if shape(n,4)==shape(n,), FAILED. python could think shape(n,4)==shape(1,n)?
        
        played_cards_weight = total_weights[card_index_0, round_played_cards.reshape(-1)].reshape(self.n, 4)
        played_cards_weight = np.where(((played_cards_weight<=13) & all_leading_suit_yes), played_cards_weight+13, played_cards_weight)

        # alternative optm. NOT debug yet: remove the line with single '2'
        '''
        leading_round_trump_yes0 = (played_cards_weight==40)
        leading_round_trump_yes1 = np.sum(leading_round_trump_yes0, axis=1)
        leading_round_trump_yes2 = np.where(leading_round_trump_yes1>1) #if has only one regular 2, need not 40->41
        leading_round_trumps = leading_round_trump_yes0[leading_round_trump_yes2]
        leading_round_trump_index_1 = np.argmax(leading_round_trumps, axis=1)  #if 2 True in a line, the argmax is the position of first True
        leading_round_trumps[np.arange(len(leading_round_trumps)), leading_round_trump_index_1] += 1
        '''
        
        leading_round_trump_index = np.where(played_cards_weight==40)
        #to serch the leading_round_trump_index[0]中，各类数字，首先出现位置上的，对应leading_round_trump_index[1]
        leading_round_trump_index0s = list(set(leading_round_trump_index[0]))
        for leading_round_trump_index0 in leading_round_trump_index0s:
            leading_round_trump_index1 = np.where(leading_round_trump_index[0] == leading_round_trump_index0)
            if 1 == len(leading_round_trump_index1[0]) :
                continue #if has only one regular 2, need not 40->41
            else:
                leading_round_trump_index2 = leading_round_trump_index[1][leading_round_trump_index1[0][0]] #leading_round_trump_index1[0]这个位置就是首先出现的，leading_round_trump_index[1]的位置
                played_cards_weight[leading_round_trump_index0, leading_round_trump_index2] += 1
        
        return played_cards_weight
        
class PokerEnvironment_6_1(PokerEnvironment): #6 steps
    #id, n, keep_env, seed are mandatoy input param. add default due to UT class in verify
    def __init__(self, id=0, n=1, keep_env=False, seed=13, reward_times=5, input_format=[0,-1,0.5,1], input_play_rewards=[-1, 0.5, 1, 10]): #fmt[0](none) must be diff to fmt[1](discarded). during playing, agent should recognize this difference 
        super().__init__(id, n, keep_env, seed)
        
        #discard
        self.reward_times_10 = 10*reward_times+1  #support times_1=0
        self.reward_times_1  = 1 *reward_times+0
        #input_format: [not existing, discarded/played, in-hand, trump]
        self.net_input_format = input_format # not in-hand MUST BE: <=0
        
        #playing. reward TBD: 多赢vs少输
        #loss: 0 vs -1
        # round reward: Rwinner=template[2]';  Rlosser=template[2]'*template[0] #scored
        # round reward: Rwinner=template[1];   Rlosser=template[1]*template[0] #not scored
        # game reward : Rwinner+=template[3]'; Rlosser+=template[3]'*template[0]
        # note: template[2]' = template[2] * score/50; template[3]' = template[3] * raised_level
        self.play_reward_template = input_play_rewards  #[loss, round_winner, scored/50=(5~40)/50=0.1~0.8, game_winner]
        #print("PokerEnvironment_6_1 init ID ", self.id)
    
    def discard_step(self, discarded_card_oindexes):  # ordered by cardset (0~n-1)
#columns=['oindex', 'suit', 'name', 'score', 'trumps', 'who', 'played', 'discarded'])
        
        # a1 = shape(10,), a2 = shape(5,). to search a2's elements in a1
        # ydl = a1[:, None] = shape(10,1)
        # ydl2 = (ydl == a2) = shape(10,5).相当于做了repeat
        #consistency: np_start_xxx[n]=player_cards[0][n]: yes; full_poker.np_cards[n]: no
        np_discards = self.np_start_player_cards[:,:,dc.COL_DISCARDED]
        np_oindex = self.np_start_player_cards[:,:,dc.COL_OINDEX]
        np_oindex_T_6 = np_oindex[:,:,np.newaxis].repeat(6, axis=2).swapaxes(1,2)
        np_discarded_card_oindexes_18 = discarded_card_oindexes[:,:,np.newaxis].repeat(18, axis=2)
        oindex_index = np.where(np_oindex_T_6 == np_discarded_card_oindexes_18)
        np_discards[oindex_index[0], oindex_index[2]] = True
        
        #str state maybe good way for state description  
        self.states = self.full_poker.generate_str_key(np.arange(self.n), self.full_poker.np_bankers[:,np.newaxis])

        #from v5.0, discard_step() must 'dump' 6 cards in onego. rewards and done are meaningless. the RL becomes normal suprvised learning
        rewards = self.reward_times_10 #dummy, np.array([self.reward_times_10]*self.n) #dummy
        done = True #dummy
        
        return self.states, rewards, done 

    def discard_step2(self, discarded_card_oindexes):
        tick11 = time.time()
        #apply state2 (54) onehot
        _, reward, done = self.discard_step(discarded_card_oindexes)  #df_player_cards update

        discard_index_0 = np.arange(self.n)[:,np.newaxis].repeat(6, axis=1)
        self.state2s[discard_index_0.reshape(-1), discarded_card_oindexes.reshape(-1)] = self.net_input_format[1]

        
        cardsets = np.arange(self.n)
        bankers = self.full_poker.np_bankers
        oindexes = discarded_card_oindexes
        fmt = self.net_input_format[1]
        self.update_discard_state3(cardsets, bankers, oindexes, fmt)
        self.state3s_fraud = self.build_state3s_fraud()

        
        #update_discard_state3(self.net_input_format[1], discarded_card_oindex)
        #self.state3s[self.full_poker.banker].update_discard_state3(self.net_input_format[1], discarded_card_oindex)
        tick12 = time.time()
        meas.perfs.env.perf_diff_a += tick12 - tick11

        return self.state3s, self.state2s, reward, done
    
    
    def discard_done(self, discarding_oindexes): #sync to full_poker.players_cards was done
        tick11 = time.time()
        discard_index_0 = np.arange(self.n)[:,np.newaxis].repeat(6, axis=1)
        np_full_discard = self.full_poker.np_cards[:,:, dc.COL_DISCARDED]
        np_full_discard[discard_index_0.reshape(-1), discarding_oindexes.reshape(-1)] = True

        tick12 = time.time()
        meas.perfs.env.perf_diff_b += tick12 - tick11

        return

    def step_play_1card(self, players_x, fmt_x, oindexes_x, in_round_oindex=[]):
        tick11 = time.time()
        cardsets = np.arange(self.n)
        self.full_poker.np_cards[cardsets, oindexes_x, dc.COL_PLAYED] = True

        #if the player is banker, ... since players_cards[0] is shape(18)
        banker_yes = (players_x == self.full_poker.np_bankers)
        cards_banker_yes = self.players_cards[0][banker_yes]
        oindexes_x_18 = oindexes_x[banker_yes][:,np.newaxis].repeat(18, axis=1)
        cards_index_banker_yes = np.where(cards_banker_yes[:, :, dc.COL_OINDEX] == oindexes_x_18)
        self.players_cards[0][banker_yes, cards_index_banker_yes[1], dc.COL_PLAYED] = True

        #if the player is not banker, ... since players_cards[1] is shape(12) with permanent order: 0=E, 1=N, W=2
        banker_no = ~banker_yes
        cards_banker_no = self.players_cards[1][banker_no]
        oindexes_x_12 = oindexes_x[banker_no][:,np.newaxis].repeat(12, axis=1)
        cards_index_banker_no = np.where(cards_banker_no[np.arange(len(cards_banker_no)), players_x[banker_no]-1-1, :, dc.COL_OINDEX] == oindexes_x_12)
        try:
            self.players_cards[1][banker_no, players_x[banker_no]-1-1, cards_index_banker_no[1], dc.COL_PLAYED] = True
        except IndexError as e:
            print("shape mismatch")
            print("banker_no: ", banker_no)
            print("players_x: ", players_x)
            print("cards_index_banker_no: ", cards_index_banker_no)

        #self.state3s = np.zeros((n, 4, 5, 54))
        self.update_inhand_state3(cardsets, players_x, oindexes_x, self.net_input_format[0]) #fmt0=not inhand
        self.update_played_state3(cardsets, players_x, oindexes_x, fmt_x)  #fmt_x: the fmt of the played cards
        self.state3s_fraud = self.build_state3s_fraud(players_x)
        
        #for player2 in self.players[0:4]:
        #    self.state3[player2].update_played_state3(player, fmt_x, oindex)

        if self.render == True:
            in_round_oindex2 = []
            if 0 < len(in_round_oindex):
                np_in_round_oindex0 = np.array(in_round_oindex)
                np_in_round_oindex = np_in_round_oindex0[:,:,1]
                in_round_oindex2 = np_in_round_oindex.tolist()
            self.full_poker.render(cardsets, in_round_oindex=in_round_oindex2)

        tick12 = time.time()
        meas.perfs.env.perf_diff_c += tick12 - tick11

        return self.state3s, self.state3s_fraud

    def game_result(self, last_round_winners, scores):
        tick11 = time.time()
        scores_sn = np.sum(scores[:, (dc.Players.SOUTH-1, dc.Players.NORTH-1)], axis=1)
        scores_ew = np.sum(scores[:, (dc.Players.EAST-1, dc.Players.WEST-1)], axis=1)
        
        s_full_discard = self.full_poker.np_cards[:,:,dc.COL_DISCARDED].astype(bool)
        s_full_scores = self.full_poker.np_cards[:,:,dc.COL_SCORE]
        discard_score = np.sum(s_full_scores[s_full_discard].reshape(self.n, 6), axis=1)
        
        sn_win_yes = np.logical_or(dc.Players.SOUTH==last_round_winners, dc.Players.NORTH==last_round_winners)
        scores_sn  = np.where(sn_win_yes, scores_sn+discard_score, scores_sn)
        scores_ew = np.where(sn_win_yes, scores_ew, scores_ew+discard_score)
        
        #keep shape(n, 1)
        bankers_sn_yes = np.logical_or((self.full_poker.np_bankers == dc.Players.SOUTH), (self.full_poker.np_bankers == dc.Players.NORTH))
        sn_win_sn_banker = np.logical_and((scores_ew < 40), bankers_sn_yes)
        ew_win_sn_banker = np.logical_and((scores_ew >= 40), bankers_sn_yes)
        ew_win_ew_banker = np.logical_and((scores_sn < 40), ~bankers_sn_yes)
        sn_win_ew_banker = np.logical_and((scores_sn >= 40), ~bankers_sn_yes)
        
        level_raised = np.zeros((self.n))
        winners = np.zeros((self.n, 2))
        level_raised[sn_win_sn_banker] = ((40 - scores_ew[sn_win_sn_banker]-1)/20).astype(int) + 1   # -1, if ew=20, sn only raise 1
        level_raised[sn_win_sn_banker] = np.where(scores_ew[sn_win_sn_banker]==0, level_raised[sn_win_sn_banker]+1, level_raised[sn_win_sn_banker]) #ew=0, level +3
        level_raised[ew_win_sn_banker] = ((scores_ew[ew_win_sn_banker] - 40)/20).astype(int)
        level_raised[sn_win_ew_banker] = ((scores_sn[sn_win_ew_banker] - 40)/20).astype(int)
        level_raised[ew_win_ew_banker] = ((40 - scores_sn[ew_win_ew_banker]-1)/20).astype(int) + 1
        level_raised[ew_win_ew_banker] = np.where(scores_sn[ew_win_ew_banker]==0, level_raised[ew_win_ew_banker]+1, level_raised[ew_win_ew_banker])
        winners[np.logical_or(sn_win_sn_banker, sn_win_ew_banker)] = np.array([[dc.Players.SOUTH, dc.Players.NORTH]])   #they are referring to same memory data
        winners[np.logical_or(ew_win_sn_banker, ew_win_ew_banker)] = np.array([dc.Players.EAST, dc.Players.WEST])

        tick12 = time.time()
        meas.perfs.env.perf_diff_d += tick12 - tick11

        return winners, level_raised, scores_sn, scores_ew

    def game_result0(self, last_round_winner, scores):
        score_sn = scores[dc.Players.SOUTH] + scores[dc.Players.NORTH]
        score_ew = scores[dc.Players.EAST] +scores[dc.Players.WEST]
        
        s_full_discard = self.full_poker.df_cards['discarded']
        s_full_score = self.full_poker.df_cards['score']
        discard_score = s_full_score[s_full_discard.values==True].values
        if last_round_winner in [dc.Players.SOUTH, dc.Players.NORTH]:
            score_sn += np.sum(discard_score)
        elif last_round_winner in [dc.Players.EAST, dc.Players.WEST]:
            score_ew += np.sum(discard_score)
        else:
            print("game_result: wrong winner player")
            winner = [dc.Players.NONE, dc.Players.NONE]

        if self.full_poker.banker in [dc.Players.SOUTH, dc.Players.NORTH]:
            if score_ew >= 40:
                winner = [dc.Players.EAST, dc.Players.WEST]
                level_raised = int((score_ew - 40)/20)
            else:
                winner = [dc.Players.SOUTH, dc.Players.NORTH]
                level_raised = int((40 - score_ew)/20) + 1
                
        else:
            if score_sn >= 40:
                winner = [dc.Players.SOUTH, dc.Players.NORTH]
                level_raised = int((score_sn - 40)/20)
            else:
                winner = [dc.Players.EAST, dc.Players.WEST]
                level_raised = int((40 - score_sn)/20) + 1

        return winner, level_raised, score_sn, score_ew

    def round_result(self, round_played_cards):  #shape(self.n,4). can't speficy the cardset. verify tester need updates
        tick11 = time.time()
        score_index_0 = np.arange(self.n)[:,np.newaxis].repeat(4, axis=1).reshape(-1)
        np_round_played_cards = np.array(round_played_cards)   #shape(n,4,2)
        np_round_played_player = np_round_played_cards[:,:,0].astype(int)
        np_round_played_oindex = np_round_played_cards[:,:,1].astype(int)
        np_full_score = self.full_poker.np_cards[:, :, dc.COL_SCORE]

        player_score0 = np_full_score[score_index_0, np_round_played_oindex.reshape(-1)].reshape(self.n, 4)
        player_score = np.sum(player_score0, axis=1)
        card_weights_updated = self.dynamic_card_weights(np_round_played_oindex)
        
        winner_index = np.argmax(card_weights_updated, axis=1)
        winners = np_round_played_player[np.arange(self.n), winner_index]
        tick12 = time.time()
        meas.perfs.env.perf_diff_e += tick12 - tick11
        
        return winners, player_score
        
    #keep the v4.2 for cross checking in UT
    def round_result0(self, cardset, round_played_cards):
        np_round_played_cards = np.array(round_played_cards)
        np_round_played_oindex = np_round_played_cards[:,1]

        ''' 0
        df_full_cards = self.full_poker.df_cards
        np_full_oindex = df_full_cards['oindex'].values
        np_full_score = df_full_cards['score'].values
        np_full_trump = df_full_cards['trumps'].values
        '''
        np_full_oindex = self.full_poker.np_cards[cardset,:, dc.COL_OINDEX]
        np_full_score  = self.full_poker.np_cards[cardset,:, dc.COL_SCORE]
        np_full_trump  = self.full_poker.np_cards[cardset,:, dc.COL_TRUMP]
        
        ###################
        # winner assumption
        ###################
        index = np.argwhere(np_full_oindex==np_round_played_oindex[0]).reshape(-1)
        winner = round_played_cards[0][0]  #init assumption
        winner_oindex = round_played_cards[0][1]
        winner_is_trump = np_full_trump[index]
        winner_suit, winner_card_name = dc.oindex_2_suit_name(winner_oindex) #single card
        #0, other trump, normal 2, master 2, RJ, BJ
        winner_trump_class = self.full_poker.trump_class0(cardset, winner_suit, winner_card_name)
        player_score = np_full_score[index][0]

        for [player, card_oindex] in round_played_cards[1:4]:
            index = np.argwhere(np_full_oindex==card_oindex)[0].reshape(-1)
            player_score += np_full_score[index][0]
            player_is_trump = np_full_trump[index]
            player_trump_class = 0
            if winner_is_trump == True:
                if player_is_trump == False : #winner is trump but player not
                    #print("round_result 1")
                    continue
                else: #winner and player are trump
                    #2:self.round
                    player_suit, card_name = dc.oindex_2_suit_name(card_oindex) #single card
                    player_trump_class = self.full_poker.trump_class0(cardset, player_suit, card_name)
                    if winner_trump_class < player_trump_class:
                        winner = player
                        winner_oindex = card_oindex
                        winner_is_trump = True
                        winner_suit = player_suit
                        winner_trump_class = player_trump_class
                        #print("round_result 2")
                    elif winner_trump_class == 1 and winner_trump_class == player_trump_class: #both in other trump, expect normal 2
                        A_card_oindex = card_oindex
                        A_winner_oindex = winner_oindex
                        #oindex=52(BJ, %13=0) would not go here
                        if 0 == card_oindex % 13 :
                            A_card_oindex = card_oindex+13
    
                        if 0 == winner_oindex % 13 :
                            A_winner_oindex = winner_oindex+13
                            
                        if A_winner_oindex < A_card_oindex:
                            winner = player
                            winner_oindex = card_oindex
                            winner_is_trump = True
                            winner_suit = player_suit
                            winner_trump_class = player_trump_class
                            #print("round_result 3")
                            #done
                        else:
                            #print("round_result 4")   #xxx
                            continue
                    else:
                        #print("round_result 5")
                        continue
            elif player_is_trump == True : #player is trump but winner not
                winner = player
                winner_oindex = card_oindex
                winner_is_trump = True
                winner_suit, winner_card_name = dc.oindex_2_suit_name(card_oindex) #single card
                winner_trump_class = self.full_poker.trump_class0(cardset, winner_suit, winner_card_name)
                #print("round_result 6")
                #done
                
            else : #winner and player are not trump
                player_suit = dc.oindex_2_suit(card_oindex)[0] #single card
                if winner_suit == player_suit:
                    A_card_oindex = card_oindex
                    A_winner_oindex = winner_oindex
                    #oindex=52(BJ, %13=0) would not go here
                    if 0 == card_oindex % 13 :
                        A_card_oindex = card_oindex+13

                    if 0 == winner_oindex % 13 :
                        A_winner_oindex = winner_oindex+13
                        
                    if A_winner_oindex < A_card_oindex:
                        winner = player
                        winner_oindex = card_oindex
                        winner_is_trump = False
                        winner_suit = dc.oindex_2_suit(card_oindex)[0] #single card
                        winner_trump_class = player_trump_class
                        #print("round_result 7")
                    else:
                        #print("round_result 8")
                        continue
                else:
                    #print("round_result 9") #xxx
                    continue
                    
        return winner, player_score

    #keep v4.2 for UT purpose
    def collect_card_last_round0(self, last_winner):
        s_full_played = self.full_poker.np_cards[:, :, dc.COL_PLAYED]
        s_full_oindex = self.full_poker.np_cards[:, :, dc.COL_OINDEX]
        s_full_who = self.full_poker.np_cards[:, :, dc.COL_WHO]
        s_full_discarded = self.full_poker.np_cards[:, :, dc.COL_DISCARDED]
        last_round_cards = []
        
        for i in range(self.n):
            not_played_index = np.where(s_full_played[i]==False)
            discard_index = np.where(s_full_discarded[i]==True)
            last_round_index = list(set(not_played_index[0]) - set(discard_index[0]))
            s_oindex = s_full_oindex[i, last_round_index]
            s_who = s_full_who[i, last_round_index]
            pos = last_winner[i]-1
            seqs = self.players[pos:pos+4]
            last_round_card = [[player,s_oindex[s_who==player]] for player in seqs]
            last_round_cards.append(last_round_card)
        return last_round_cards

    def collect_card_last_round(self, last_winners):
        tick11 = time.time()
        s_full_played = self.full_poker.np_cards[:, :, dc.COL_PLAYED]
        s_full_oindex = self.full_poker.np_cards[:, :, dc.COL_OINDEX]
        s_full_who = self.full_poker.np_cards[:, :, dc.COL_WHO]
        s_full_discarded = self.full_poker.np_cards[:, :, dc.COL_DISCARDED]
        
        last_round_index = np.logical_not(np.logical_or(s_full_played, s_full_discarded))
        try :
            s_oindex = s_full_oindex[last_round_index].reshape(self.n, 4)
        except ValueError as e:
            print(e)
            ydl = last_round_index
        s_who = s_full_who[last_round_index].reshape(self.n, 4)
        
        #order: last winner at first place
        #last_winners_pos = np.where(s_who == last_winners[:,np.newaxis])  #(n,4) == (n,1) ... works
        last_who = np.zeros((self.n, 4), dtype=np.int32)
        last_oindex = np.zeros((self.n, 4), dtype=np.int32)
        for i in range(self.n):
            last_who[i] = self.players[last_winners[i]-1:last_winners[i]+4-1]   #player used as index
            
        for i in range(4):
            last_oindex_pos = np.where(s_who == last_who[:,i][:, np.newaxis])   #(n,4) == (n,1) ... works
            last_oindex[:,i] = s_oindex[np.arange(self.n), last_oindex_pos[1]]
        last_round_cards = np.concatenate((last_who[:,:,np.newaxis], last_oindex[:,:,np.newaxis]), axis=2)

        tick12 = time.time()
        meas.perfs.env.perf_diff_f += tick12 - tick11

        return last_round_cards

    def round_rewards0(self, round_winner, player_sequence, score_got):
        winner_pos = player_sequence.index(round_winner)
        round_winner2 = player_sequence[(winner_pos+2)%4]
        rewards = [0,0,0,0]
        for i, player in enumerate(player_sequence):
            if player in [round_winner, round_winner2]:
                if score_got > 0:
                    rewards[i] = self.play_reward_template[2]*score_got/4  #win and got scored
                else:
                    rewards[i] = self.play_reward_template[1]  #win
            else: #losser
                if score_got > 0:
                    rewards[i] = -self.play_reward_template[2]*score_got/4  #negtive
                else:
                    rewards[i] = self.play_reward_template[0]   #loss
                
        return rewards

    def round_rewards(self, round_winners, player_sequences, scores_got):
        tick11 = time.time()
        round_winners_pos = np.where(player_sequences == round_winners[:, np.newaxis])  #shape(n,4) == shape(n,1)
        round_winners2_pos = (round_winners_pos[1]+2) % 4  #scored/50=(5~40)/50=0.1~0.8)
        reward_template = np.where(scores_got>0, self.play_reward_template[2]*scores_got/50, self.play_reward_template[1])  #shape(n,)

        rewards = self.play_reward_template[0] * reward_template[:,np.newaxis].repeat(4, axis=1) #default as losser
        rewards[round_winners_pos[0], round_winners_pos[1]] = reward_template
        rewards[round_winners_pos[0], round_winners2_pos]   = reward_template
        tick12 = time.time()
        meas.perfs.env.perf_diff_g += tick12 - tick11

        return rewards

    def game_rewards(self, game_winners, game_rewards, level_raised):
        fmt0 = self.play_reward_template[0]  #game losser
        fmt3_2 = self.play_reward_template[3]*(level_raised+1)[np.newaxis,:].repeat(11,axis=0)
        south_winner_batch_yes = (dc.Players.SOUTH == game_winners).sum(axis=1).astype(bool)
        game_rewards[dc.Players.SOUTH][:, south_winner_batch_yes]  += fmt3_2[:, south_winner_batch_yes]
        game_rewards[dc.Players.SOUTH][:, ~south_winner_batch_yes] += fmt0 * fmt3_2[:, ~south_winner_batch_yes]
        game_rewards[dc.Players.NORTH] = game_rewards[dc.Players.SOUTH]
        game_rewards[dc.Players.EAST][:, ~south_winner_batch_yes] += fmt3_2[:, ~south_winner_batch_yes]
        game_rewards[dc.Players.EAST][:, south_winner_batch_yes] += fmt0 * fmt3_2[:, south_winner_batch_yes]
        game_rewards[dc.Players.WEST] = game_rewards[dc.Players.EAST]
   
        return game_rewards
        

    def guess_callback(self, actid, input_data=0):
        if 0 == actid:  #fmt for every cards. doesn't identify 'discard'
            guess_cards3 = input_data
            #in decide(), input shape=(<n,1>, 3, 54). 1=1by1; n=4in1. for guess, to identify fmt0, 2, 3
            #in learning(), input shape=(11*n*<4,1>, 3, 54); 1=1by1; 4=4in1(4 players)
            fmts = self.net_input_format
    
            # !!! limitation: assume round===2 with SPADE. the trump is same crossing cardsets !!!
            trumps0 = self.full_poker.np_cards[0,:,dc.COL_TRUMP][np.newaxis, :]
            trumps = np.where(trumps0 == True, fmts[3], fmts[2])[:, np.newaxis, :]  #shape不同，但只有一dim不同，且dim=1，可以+,-,*,/操做
            guess_cards = guess_cards3 * trumps
            _ = game_verify.checkpoints.checkpoints_entry(game_verify.CHKIDs.CHKID_23, guess_cards3, trumps, guess_cards)

            ret = guess_cards

        elif 1 == actid : #am I 庄家?
            state3s_batch = input_data  #标准state3
            #in CNN: input shape=(n,5,54,1)
            #in DNN: input shape=(n,5*54)
            state3s_batch_shape = state3s_batch.shape
            state3s_batch = state3s_batch.reshape(state3s_batch_shape[0], 5, 54)
            
            fmts = self.net_input_format
            discards = np.where(state3s_batch[:,0,:] == fmts[1], 1, 0)
            discard_sum6 = np.sum(discards, axis=1)
            banker_yes = (discard_sum6==6)
            
            ret = banker_yes
            
        elif 2 == actid : #庄家在[0，1，2，3]的位置
            #calculate at env.step_play_1card()
            ret = self.banker_pos #retrive direclty
            
        elif 3 == actid : #
            ret = self.net_input_format
            
        return ret

    def banker_position_for_guess(self, players_x, i=-1):
        #[S] - [S, E, N, W] = [0, -1, -2, -3]
        # xx % 4 = 0, 3, 2, 1. then banker_pos=[3, 2, 1] if player_x=[E, N, W]
        if -1 == i:
            #full batch, 4in1
            self.banker_pos = (self.full_poker.np_bankers - players_x) % 4
        else:
            #specific cardset of full batch, 1by1
            self.banker_pos = np.array([(self.full_poker.np_bankers[i] - players_x[i]) % 4])
            
        
    def card_discarded_for_guess(self):
        return self.full_poker.np_cards[:,:,dc.COL_DISCARDED]

#verify    
def AKQJ_test():
    global obj_counter
    obj_counter = 0
    gc.set_debug(gc.DEBUG_UNCOLLECTABLE) #|gc.DEBUG_COLLECTABLE) # | gc.DEBUG_LEAK)
    play_env = PokerEnvironment_6_1(777, keep_env=False, seed=13, reward_times=1, input_format=[0,-1,1,2])
    se1 = meas.get_size(play_env)

    rd.seed(13)
    play_env.reset(keep=False, render=False)
    obj_counter = 0
    print("size ater 1st reset..............................: ")
    print("size ater 1st reset..............................: ")
    print("size ater 1st reset..............................: ")
    print("size ater 1st reset..............................: ")
    print("size ater 1st reset..............................: ")
    se2 = meas.get_size(play_env, render=False) #False) #True)
    print("size1 & 2..............................: ", se1, se2)

    isgc = gc.isenabled() 
    threshold = gc.get_threshold() 
    print("AKQJ thred..............................: ", threshold, isgc)
    gc.set_threshold(int(1e+6), int(1e+6), int(1e+6))
    threshold = gc.get_threshold()     
    print("AKQJ thred..............................: ", threshold)

    for i in range(1):
        #play_env = PokerEnvironment_6_1(777, keep_env=True, seed=13, reward_times=1, input_format=[0,-1,1,2])
        #rd.seed(13)
        play_env.reset(keep=False, render=False)
        play_env.build_np_state3(dc.Players.SOUTH)

        obj_counter = 0
        print("size ater 2nd reset..............................: ")
        print("size ater 2nd reset..............................: ")
        print("size ater 2nd reset..............................: ")
        print("size ater 2nd reset..............................: ")
        print("size ater 2nd reset..............................: ")
        print("size ater 2nd reset..............................: ")
        se = meas.get_size(play_env, render=False) # False) True
        if i%GC_CYCLE == 0:
            gbs = gc.collect()
            print("AKQJ and size..............................: ", i, gbs, se)
    return
#AKQJ_test()


def play_test():
    #temp test
    '''
    ydl = np.random.random((50, 18, 8))
    
    for i in range(100000): #9 times faster than list()->array()
        tc11 = time.time()
        ydl2 = copy.deepcopy(ydl)
        tc12 = time.time()
        perf_diff_1 += tc12- tc11

    for i in range(100000):
        tc21 = time.time()
        ydl3 = list(ydl)
        ydl4 = np.array(ydl3)
        tc22 = time.time()
        perf_diff_2 += tc22- tc21

    #temp test
    '''
    
    
    n = 50
    np.random.seed(13)
    bankers1 = np.array([dc.Players.SOUTH, dc.Players.EAST, dc.Players.NORTH, dc.Players.WEST]*100).reshape(-1)
    bankers2 = np.array([dc.Players.SOUTH, dc.Players.SOUTH, dc.Players.SOUTH, dc.Players.SOUTH]*100).reshape(-1)  #only support S=banker
    play_env = PokerEnvironment_6_1(777, n, keep_env=False, seed=13, reward_times=1, input_format=[0,-1,1,2])
    for i in range(1000):
        play_env.reset(bankers2[:n], keep=False, render=False)
    
    
    #display
    meas.perfs.performance_report()
    return
#play_test()

