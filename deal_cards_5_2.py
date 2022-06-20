#added v4
#v4.1: optm the reset and saver
#v5.0: multiple deal set support, based off v4.1
#      numpy replace df in general. use enum only in 'print'
#      indexed by Player name, MUST be -1

import time
import numpy as np
import pandas as pd
import random as rd
from enum import Enum, IntEnum
from random import shuffle as rshuffle
import collections as cllt
import copy

ticks = time.time()
seed0 = int(ticks*100000) % (2**32-1)
#np.random.seed(seed0)
rd.seed(seed0)

#np.random.seed()  #empty input param = seeds from current time
rd.seed()  #empty input param = seeds from current time

#np.random.seed(13)  #keep same random sequence
rd.seed(13)  #keep same random sequence


class CardSuits(IntEnum):
    NONE       = 0
    SPADES     = 1
    HEARTS     = 2
    CLUBS      = 3
    DIAMONDS   = 4
    BLACKJOKER = 5  #must be this place
    REDJOKER   = 6  #must be this place. otherwise, the ‘T' print is wrong

#keep mapping
cardsuits_value_to_name = {0: CardSuits.NONE,
                           1: CardSuits.SPADES,
                           2: CardSuits.HEARTS,
                           3: CardSuits.CLUBS,
                           4: CardSuits.DIAMONDS,
                           5: CardSuits.BLACKJOKER,
                           6: CardSuits.REDJOKER }
#cardsuits_value_to_name_int = np.array([CardSuits.NONE, CardSuits.SPADES, CardSuits.HEARTS, CardSuits.CLUBS, CardSuits.DIAMONDS, CardSuits.BLACKJOKER, CardSuits.REDJOKER]) 

# player index of numpy = Player.xx - 1
class Players(IntEnum):
    NONE   = 0
    SOUTH  = 1
    EAST   = 2
    NORTH  = 3
    WEST   = 4

#keep
players_value_to_name = {0: Players.NONE,
                         1: Players.SOUTH,
                         2: Players.EAST,
                         3: Players.NORTH,
                         4: Players.WEST }


card_print_name = {1: 'A',
                   2: '2',
                   3: '3',
                   4: '4',
                   5: '5',
                   6: '6',
                   7: '7',
                   8: '8',
                   9: '9',
                   10: '10',
                   11: 'J',
                   12: 'Q',
                   13: 'K',
                   14: 'BJ',
                   15: 'RJ' }

brief_suit = {CardSuits.SPADES: 'S',
              CardSuits.HEARTS: 'H',
              CardSuits.CLUBS:  'C',
              CardSuits.DIAMONDS: 'D',
              CardSuits.BLACKJOKER: 'T',
              CardSuits.REDJOKER: 'T'}

perfix_space = {Players.NORTH: ' '*18,
                Players.WEST : ' ',
                Players.EAST : ' '*32,
                Players.SOUTH: ' '*18 }

#optm: gloabl data is faster than stack local data 2.5+ times
cards_T = np.array([[0 ,  1,   1 ,  0 ,  0,   0,   0,   0],
                    [1 ,  1,   2 ,  0 ,  0,   0,   0,   0],
                    [2 ,  1,   3 ,  0 ,  0,   0,   0,   0],
                    [3 ,  1,   4 ,  0 ,  0,   0,   0,   0],
                    [4 ,  1,   5 ,  5 ,  0,   0,   0,   0],
                    [5 ,  1,   6 ,  0 ,  0,   0,   0,   0],
                    [6 ,  1,   7 ,  0 ,  0,   0,   0,   0],
                    [7 ,  1,   8 ,  0 ,  0,   0,   0,   0],
                    [8 ,  1,   9 ,  0 ,  0,   0,   0,   0],
                    [9 ,  1,   10,  10,  0,   0,   0,   0],
                    [10,  1,   11,  0 ,  0,   0,   0,   0],
                    [11,  1,   12,  0 ,  0,   0,   0,   0],
                    [12,  1,   13,  10,  0,   0,   0,   0],
                    [13,  2,   1 ,  0 ,  0,   0,   0,   0],
                    [14,  2,   2 ,  0 ,  0,   0,   0,   0],
                    [15,  2,   3 ,  0 ,  0,   0,   0,   0],
                    [16,  2,   4 ,  0 ,  0,   0,   0,   0],
                    [17,  2,   5 ,  5 ,  0,   0,   0,   0],
                    [18,  2,   6 ,  0 ,  0,   0,   0,   0],
                    [19,  2,   7 ,  0 ,  0,   0,   0,   0],
                    [20,  2,   8 ,  0 ,  0,   0,   0,   0],
                    [21,  2,   9 ,  0 ,  0,   0,   0,   0],
                    [22,  2,   10,  10,  0,   0,   0,   0],
                    [23,  2,   11,  0 ,  0,   0,   0,   0],
                    [24,  2,   12,  0 ,  0,   0,   0,   0],
                    [25,  2,   13,  10,  0,   0,   0,   0],
                    [26,  3,   1 ,  0 ,  0,   0,   0,   0],
                    [27,  3,   2 ,  0 ,  0,   0,   0,   0],
                    [28,  3,   3 ,  0 ,  0,   0,   0,   0],
                    [29,  3,   4 ,  0 ,  0,   0,   0,   0],
                    [30,  3,   5 ,  5 ,  0,   0,   0,   0],
                    [31,  3,   6 ,  0 ,  0,   0,   0,   0],
                    [32,  3,   7 ,  0 ,  0,   0,   0,   0],
                    [33,  3,   8 ,  0 ,  0,   0,   0,   0],
                    [34,  3,   9 ,  0 ,  0,   0,   0,   0],
                    [35,  3,   10,  10,  0,   0,   0,   0],
                    [36,  3,   11,  0 ,  0,   0,   0,   0],
                    [37,  3,   12,  0 ,  0,   0,   0,   0],
                    [38,  3,   13,  10,  0,   0,   0,   0],
                    [39,  4,   1 ,  0 ,  0,   0,   0,   0],
                    [40,  4,   2 ,  0 ,  0,   0,   0,   0],
                    [41,  4,   3 ,  0 ,  0,   0,   0,   0],
                    [42,  4,   4 ,  0 ,  0,   0,   0,   0],
                    [43,  4,   5 ,  5 ,  0,   0,   0,   0],
                    [44,  4,   6 ,  0 ,  0,   0,   0,   0],
                    [45,  4,   7 ,  0 ,  0,   0,   0,   0],
                    [46,  4,   8 ,  0 ,  0,   0,   0,   0],
                    [47,  4,   9 ,  0 ,  0,   0,   0,   0],
                    [48,  4,   10,  10,  0,   0,   0,   0],
                    [49,  4,   11,  0 ,  0,   0,   0,   0],
                    [50,  4,   12,  0 ,  0,   0,   0,   0],
                    [51,  4,   13,  10,  0,   0,   0,   0],
                    [52,  5,   14,  0 ,  1,   0,   0,   0],
                    [53,  6,   15,  0 ,  1,   0,   0,   0]], dtype=np.int16 )

COL_OINDEX    = 0
COL_SUIT      = 1
COL_NAME      = 2
COL_SCORE     = 3
COL_TRUMP     = 4
COL_WHO       = 5
COL_PLAYED    = 6
COL_DISCARDED = 7
COL_END       = 8  #dummy for upper limit

class FullPokers :
    def __init__(self, n=1):
        self.n = n

        #optm: cards is built by loop is 10% slower than cards=list
        #optm: cards = np.array is faster than cards=list 20 times. n=50
        self.np_cards = np.array([cards_T]*n) 
                
        self.np_trumps = np.full((n,), CardSuits.NONE)   #主牌花色
        self.np_rounds = np.zeros((n,))   #打2，3，4 ...
        self.np_bankers = np.full((n,), Players.NONE)
        self.players = [Players.SOUTH, Players.EAST, Players.NORTH, Players.WEST, Players.SOUTH, Players.EAST, Players.NORTH]
        #self.seq_from_winner = {}
        self.seq_from_winner = np.zeros((4,4))  # skip NONE
        self.seq_from_winner[Players.SOUTH-1] = [Players.SOUTH, Players.EAST,  Players.NORTH, Players.WEST ]
        self.seq_from_winner[Players.EAST-1]  = [Players.EAST,  Players.NORTH, Players.WEST,  Players.SOUTH]
        self.seq_from_winner[Players.NORTH-1] = [Players.NORTH, Players.WEST,  Players.SOUTH, Players.EAST ]
        self.seq_from_winner[Players.WEST-1]  = [Players.WEST,  Players.SOUTH, Players.EAST,  Players.NORTH]


    def cards_shuffle2(self):
        #np.random.shuffle(n_cards)
        #optm: shuffle(index) is 2 times faster than shuffle(card) with 8 elements. n=50
        #optm: compare to loop choice(18->12->12->12), shuffle() is 4 times faster even the sort is not performed in 'choice' version
        #optm: but it is the timecost method
        shuffle_54 = np.arange(54) #optm: need not init it in every loop. 8% faster than init inside loop everytime
        for i, card in enumerate(self.np_cards):
            #optm: faster than choice(a,54), 1.5 times
            #optm: np.shuffle() 1.35 times faster than python random.shuffle()
            np.random.shuffle(shuffle_54) 
            self.np_cards[i] = card[shuffle_54]

    def cards_shuffle3(self):
        shuffle_template = np.random.random((self.n, 54)) #uint8 faster than int16 faster than float
        index_54 = np.argsort(shuffle_template, axis=1)
        index_n = np.arange(self.n)[:,np.newaxis].repeat(54, axis=1)
        np_cards = self.np_cards[index_n.reshape(-1), index_54.reshape(-1)]
        self.np_cards = np_cards.reshape(self.n, 54, COL_END)


    def cards_shuffle(self):
        #otpm: uint8: 10% fatser than float; int16: 7% faster than float @50000*50/100
        shuffle_template = np.random.randint(100, size=(self.n, 54), dtype=np.uint8)
        index_54 = np.argsort(shuffle_template, axis=1)
        index_n = np.arange(self.n)[:,np.newaxis].repeat(54, axis=1)
        np_cards = self.np_cards[index_n.reshape(-1), index_54.reshape(-1)]
        self.np_cards = np_cards.reshape(self.n, 54, COL_END)

    def cards_assign_to_player(self):
        who = self.np_cards[:,:,COL_WHO]
        #optm: faster than loop(n). 4 times faster when n=50
        first_player = self.seq_from_winner[self.np_bankers-1]
        fourth = first_player[:,3][:,np.newaxis]
        third  = first_player[:,2][:,np.newaxis]
        second = first_player[:,1][:,np.newaxis]
        first  = first_player[:,0][:,np.newaxis]
        who[:,0:12]  = fourth.repeat(12, axis=1)
        who[:,12:24] = third.repeat(12, axis=1)
        who[:,24:36] = second.repeat(12, axis=1)
        who[:,36:]   = first.repeat(18, axis=1)
        #don't sort on np_cards[]. i'd like to see the categrized by 'who'
        self.sort_for_full_cardsets()

    def cards_assign_to_player2(self):
        cards_54 = np.arange(54)
        first_player = self.seq_from_winner[self.np_bankers-1]
        fourth = first_player[:,3][:,np.newaxis]
        third  = first_player[:,2][:,np.newaxis]
        second = first_player[:,1][:,np.newaxis]
        first  = first_player[:,0][:,np.newaxis]
        
        for i in range(self.n) :
            shuffle_index_0 = np.random.choice(cards_54, size=18, replace=False)
            cards_36 = list(set(cards_54) - set(shuffle_index_0))
            shuffle_index_1 = np.random.choice(cards_36, size=12, replace=False)
            cards_24 = list(set(cards_36) - set(shuffle_index_1))
            shuffle_index_2 = np.random.choice(cards_24, size=12, replace=False)
            shuffle_index_3 = list(set(cards_24) - set(shuffle_index_2))
            
            who = self.np_cards[i,:,COL_WHO]
            who[shuffle_index_0] = first[i]
            who[shuffle_index_1] = second[i]
            who[shuffle_index_2] = third[i]
            who[shuffle_index_3] = fourth[i]
        #need not sort. it is already ordered by default
        

    def sort_for_full_cardsets(self):
        for player_cards in self.np_cards :
            index = np.argsort(player_cards[:, COL_OINDEX], axis=0)
            player_cards[:,:] = player_cards[index]
        return

    def set_trumps_banker(self, suit_trumps, name_trumps, bankers):
        #it is a reference only the 1st dim takes full scope (0-n). eg, aa[:,:,x]
        #it is NOT a reference if the 1st dim takes a slice, eg aa[[3,5], :, xx]
        np_suit = self.np_cards[:,:,COL_SUIT]
        np_name = self.np_cards[:,:,COL_NAME]
        np_trump = self.np_cards[:,:,COL_TRUMP]

        suits_trump_54 = suit_trumps[:,np.newaxis].repeat(54, axis=1)
        is_trump = (np_suit == suits_trump_54)
        np_trump[is_trump] = True
        
        names_trump_54 = name_trumps[:,np.newaxis].repeat(54, axis=1)
        is_name = (np_name == names_trump_54)
        np_trump[is_name] = True
        
        self.np_trumps  = suit_trumps
        self.np_rounds  = name_trumps
        self.np_bankers = bankers


    def render(self, cardsets, in_round_oindex=[]):
        #print cardset one by one

        # https://www.jianshu.com/p/7d7e7e160372
        '''
        print("显示方式：")
        print("\033[0mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[1mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[4mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[5mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[7mSuixinBlog: https://suixinblog.cn\033[0m")
        print("字体色：")
        print("\033[30mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[31mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[32mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[4;33mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[34mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[1;35mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[4;36mSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[37mSuixinBlog: https://suixinblog.cn\033[0m")
        print("背景色：")
        print("\033[1;37;40m\tSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[37;41m\tSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[37;42m\tSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[37;43m\tSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[37;44m\tSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[37;45m\tSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[37;46m\tSuixinBlog: https://suixinblog.cn\033[0m")
        print("\033[1;30;47m\tSuixinBlog: https://suixinblog.cn\033[0m")
        '''

        for i in cardsets:
            if 0 < len(in_round_oindex):  #when env.reset(), the in_round_list[] = empty
                str_mappings = self.ones_cardset_strs([i], [[Players.SOUTH, Players.EAST, Players.NORTH, Players.WEST]], in_round_oindex=in_round_oindex[i])
            else:
                str_mappings = self.ones_cardset_strs([i], [[Players.SOUTH, Players.EAST, Players.NORTH, Players.WEST]])
            str_mapping = str_mappings[0]
                    
            for player in [Players.NORTH, Players.WEST, Players.SOUTH] : #order IS sensitive
                print(end='\n')
    
                for suit in CardSuits:
                    if CardSuits.NONE == suit:
                        continue
                    
                    print_strs = []
                    print_str = str_mapping[player][brief_suit[suit]]
                    if Players.WEST == player :   #combine WEST and EAST in one row
                        print_str2 = str_mapping[Players.EAST][brief_suit[suit]]
                        len_str  = len(print_str)
                        c33 = print_str.count('\033[1;37;40m')
                        len_str -= c33*(10+4)  #8
                        c33 = print_str.count('\033[1;35m')
                        len_str -= c33*(7+4)
                        c33 = print_str.count('\033[4;36m')
                        len_str -= c33*(7+4)
                        print_str2_space = ' '*(len(perfix_space[Players.EAST]) - len_str - len(perfix_space[Players.WEST]) - 3)
    
                        print_strs.append([print_str, perfix_space[Players.WEST]])
                        print_strs.append([print_str2, print_str2_space])
                    else:
                        print_strs.append([print_str, perfix_space[player]])
    
                    if self.np_trumps[i] == suit :
                        T_perfix = 'T'
                    else :
                        T_perfix = ' '
                    for print_str, spaces in print_strs :
                        print(spaces, T_perfix, brief_suit[suit], ':', print_str, end='') #str_mapping[player][brief_suit[suit]])
                    print(end='\n')
                    
                    #ensure the BJ and RJ are in last position of 'players'
                    if CardSuits.BLACKJOKER == suit or CardSuits.REDJOKER == suit :
                        break
                continue
            
            print('\n banker: ', players_value_to_name[self.np_bankers[i]])

        return
            
    def ones_cardset_strs(self, cardsets, player_sets, in_round_oindex=[]):  #support cardsets > 1
        str_mappings = []
        for cardset, player_set in zip(cardsets, player_sets):
            str_W = {'S': ' ', 'H': ' ', 'C': ' ', 'D': ' ', 'T': ' '}
            str_N = {'S': ' ', 'H': ' ', 'C': ' ', 'D': ' ', 'T': ' '}
            str_E = {'S': ' ', 'H': ' ', 'C': ' ', 'D': ' ', 'T': ' '}
            str_S = {'S': ' ', 'H': ' ', 'C': ' ', 'D': ' ', 'T': ' '}
    
            str_mapping = {Players.WEST: str_W,
                           Players.NORTH: str_N,
                           Players.EAST: str_E,
                           Players.SOUTH: str_S}
                
            all_group_players = self.get_player_cards_by_players([cardset], [player_set]) #here, only one cardset with 1 player_set []
            for group_players in all_group_players: #per cardset
                for group_player in group_players:  #array version, per player
                #for group_player in group_players.keys():  #dict version, per player
                    for card in group_player:  #array version, per card
                    #for card in group_players[group_player]:  #dict version, per card
                        self.build_player_string_per_card(str_mapping[card[COL_WHO]], card, self.np_rounds[cardset], in_round_oindex=in_round_oindex)
                
                str_mappings.append(str_mapping)  #per cardset
            
        return str_mappings

    def build_player_string_per_card(self, str_dict, card, play_round, to_print=True, in_round_oindex=[]):  #one card only
        #print("\033[37;45m\tSuixinBlog: https://suixinblog.cn\033[0m")
        played_patten_perfix = ''
        played_patten_postfix = ''
        if True == to_print :
            if card[COL_OINDEX] in in_round_oindex:
                played_patten_perfix = '\033[4;36m'   #len=7; #'\033[37;45m'  #len = 8
                played_patten_postfix = '\033[0m'     #len = 4
            elif True == card[COL_PLAYED] :
                played_patten_perfix = '\033[1;37;40m'   #len=10; #'\033[37;45m'  #len = 8
                played_patten_postfix = '\033[0m'     #len = 4
            elif True == card[COL_DISCARDED]:
                played_patten_perfix = '\033[1;35m'  #len = 7, TBD: length detection
                played_patten_postfix = '\033[0m'     #len = 4
                

        if card[COL_NAME] == play_round :
            added_str= played_patten_perfix + brief_suit[card[COL_SUIT]] + card_print_name[card[COL_NAME]] + played_patten_postfix + ' '
            str_dict['T'] += added_str
        else:
            added_str = played_patten_perfix + card_print_name[card[COL_NAME]] + played_patten_postfix + ' '
            str_dict[brief_suit[card[COL_SUIT]]] += added_str

        
    def generate_str_key(self, cardsets, player_sets):  #support cardsets > 1
        #optm: timecosty. if apply str_state, optm is needed for all str related methods
        #optm: state3 need 0.0229, while str state need 1.935 for 1000 loop
        #the str key need sorted card sequence
        cardsets_str_states = []
        
        for cardset, player_set in zip(cardsets, player_sets):
            str_mappings = self.ones_cardset_strs([cardset], [player_set])
            str_mapping = str_mappings[0] #only 1
            cardset_str_states = []
            for player in player_set:
                str_keys = str_mapping[player]
                
                player_str_state = ''
                for suit in str_keys.keys():
                    leading = suit
                    '''
                    if brief_suit[self.np_trumps[cardset]] == suit :
                        leading = 'T' + suit
                    else:
                        leading = suit
                    '''
                    str_keys_splited = str_keys[suit].split(' ')
                    for name in str_keys_splited:
                        if ' ' == name or '' == name:
                            continue
                        else:
                            player_str_state += leading + name
                cardset_str_states.append(player_str_state)
            cardsets_str_states.append(cardset_str_states)

        return cardsets_str_states
        
    #array version. assume each 'player' will appply to all 'cardsets'
    #returned allsets_player_cards is orderd by cardsets
    #TBD: can't support banker is assigned to diff player. reshape would meet (12 mixed 18). assume: banker is SOUTH only in all cardset
    def get_player_cards_by_players(self, cardsets, player_sets):
        #cardsets = [1,3,4,6,8]
        #players = [[S,E,N,W] of 1, [S,E,N,W] of 3, [S,E,N,W] of 4, [S,E,N,W] of 6 etc]. all same players for every cardset
        cards_who = []
        np_cards_selected = self.np_cards[cardsets,:,:]
        for player in [Players.SOUTH, Players.EAST, Players.NORTH, Players.WEST]:
            group_player_idx = np.where(np_cards_selected[:, :, COL_WHO]== player)
            player_cards_selected = list(np_cards_selected[group_player_idx[0], group_player_idx[1], :].reshape(len(cardsets), -1, COL_END))
            cards_who.append(player_cards_selected)
        allsets_player_cards = np.array(cards_who, dtype=object).swapaxes(0,1)

        return allsets_player_cards

    #array version. but problem is it can't distingish the player in returned array
    #then, the invoker must know and remember the player_sets
    def get_player_cards_by_players3(self, cardsets, player_sets): #support >1 cardsets with <=4 players
        #cardsets = [1,3,4,6,8]
        #players = [[N, S] of 1, [S,W,E] of 3, [S,E,N,W] of 4, [W] of 6 etc]
        allsets_player_cards = []
        for cardset, players in zip(cardsets, player_sets):
            player_cards = []
            for player in players :
                group_player_idx = np.where(self.np_cards[cardset,:,COL_WHO] == player)[0]
                group_player = self.np_cards[cardset, group_player_idx]  #a bit slower(<5%) than index. but style looks better
                #sort happens at env.init_all_player_cards() once.here, could be generate key str, needless
                player_cards.append(group_player)  #CAN't be refered to. it is a copy!
            
            allsets_player_cards.append(player_cards)

        return allsets_player_cards
        
    #dict version
    def get_player_cards_by_players2(self, cardsets, player_sets): #support >1 cardsets with <=4 players
        #cardsets = [1,3,4,6,8]
        #players = [[N, S] of 1, [S,W,E] of 3, [S,E,N,W] of 4, [W] of 6 etc]
        allsets_player_cards = []
        for cardset, players in zip(cardsets, player_sets):
            player_cards = {}
            for player in players :
                group_player_idx = np.where(self.np_cards[cardset,:,COL_WHO] == player)[0]
                group_player = self.np_cards[cardset, group_player_idx]  #a bit slower(<5%) than index. but style looks better
                player_cards[player] = group_player  #can this be refered ??!
            
            allsets_player_cards.append(player_cards)

        return allsets_player_cards


    def trump_class(self, player_suits, card_names):  #indexed by cardset=n by default. len of card_oindexs and player_suits is equal=n
        #optm: 10 times faster than loop=cardsets when n=50,100
        #class=[0, other trump, normal 2, master 2, RJ, BJ]
        player_classes = np.zeros((self.n,))

        is_suit_trump = (player_suits == self.np_trumps)
        is_round_trump = (card_names == self.np_rounds)
        is_suit_round_trump = (is_suit_trump & is_round_trump)
        is_BJ = (player_suits == CardSuits.BLACKJOKER)
        is_RJ = (player_suits == CardSuits.REDJOKER)
        
        player_classes[is_suit_trump] = 1
        player_classes[is_round_trump] = 2
        player_classes[is_suit_round_trump] = 3
        player_classes[is_BJ] = 4
        player_classes[is_RJ] = 5
        
        return player_classes

    #keep the v4.2 for cross checking in UT
    def trump_class0(self, cardset, player_suit, card_name):
        #0, other trump, normal 2, master 2, RJ, BJ
        if player_suit == CardSuits.REDJOKER:
            player_class = 5
        elif player_suit == CardSuits.BLACKJOKER:
            player_class = 4
        else:
            if card_name == self.np_rounds[cardset]:
                player_class = 2
                if player_suit == self.np_trumps[cardset]:
                    player_class = 3
            elif player_suit == self.np_trumps[cardset]:
                player_class = 1
            else:
                player_class = 0

        return player_class
    
#############################
# public methods without invoking FullPoker class
#############################
def oindex_2_suit(oindex):
    np_oindex = np.array(oindex).reshape(-1)
    np_suits = (np_oindex/13).astype(int) + 1
    
    #optm: use numpy number instead of CardSuit. 
    #optm: remove the value_to_name[] translation. 8 times faster
    if 53 in np_oindex:
        index53 = np.where(np_oindex==53)[0][0]
        np_suits[index53] = CardSuits.REDJOKER
    
    if 52 in np_oindex:
        index52 = np.where(np_oindex==52)[0][0]
        np_suits[index52] = CardSuits.BLACKJOKER

    return np_suits

def oindex_2_suit_name(oindex):
    suits = oindex_2_suit(oindex)
    np_oindex = np.array(oindex)
    names = np_oindex%13 + 1
    return suits, names

            
#############################
# verify
#############################
def test_example():
    perf_diff_1 = 0
    perf_diff_2 = 0
    perf_diff_3 = 0
    perf_diff_4 = 0
    perf_diff_5 = 0
    
    np.random.seed(13)
    for i in range(1):
        tc11 = time.time()
        full_poker = FullPokers(10)
        tc12 = time.time()
        perf_diff_1 += tc12- tc11

        tc21 = time.time()
        full_poker.cards_shuffle()
        tc22 = time.time()
        perf_diff_2 += tc22- tc21
        '''        
        full_poker.set_trumps_banker(np.arange(3), 
                                     np.array([CardSuits.SPADES]*3), 
                                     np.array([2]*3), 
                                     np.array([Players.SOUTH]*3))
        '''
        tc31 = time.time()
        full_poker.set_trumps_banker(np.array([CardSuits.SPADES]*full_poker.n), 
                                     np.array([2]*full_poker.n), 
                                     np.array([Players.SOUTH]*full_poker.n))
        tc32 = time.time()
        perf_diff_3 += tc32- tc31


        tc41 = time.time()
        full_poker.cards_assign_to_player()
        tc42 = time.time()
        perf_diff_4 += tc42- tc41

        full_poker.render(np.arange(full_poker.n))

        card_oindexs = full_poker.np_cards[:,0,COL_OINDEX]
        player_suits = full_poker.np_cards[:,0,COL_SUIT]
        tc51 = time.time()
        winner_class = full_poker.trump_class(card_oindexs, player_suits)
        tc52 = time.time()
        perf_diff_5 += tc52- tc51


    print(perf_diff_1, perf_diff_2, perf_diff_3, perf_diff_4, perf_diff_5)

    cardsets = [0,1,4]
    #dict version can support various length of player_sets
    #player_sets = [[Players.SOUTH, Players.EAST, Players.NORTH, Players.WEST],
    #               [Players.SOUTH],
    #               [Players.EAST, Players.WEST]]
    #np array version only supports full length=4 of player_sets
    player_sets = [[Players.SOUTH, Players.EAST, Players.NORTH, Players.WEST],
                   [Players.SOUTH, Players.EAST, Players.NORTH, Players.WEST],
                   [Players.SOUTH, Players.EAST, Players.NORTH, Players.WEST] ]
    
    for i in range(0):
        tc11 = time.time()
        full_poker.generate_str_key(cardsets, player_sets)
        tc12 = time.time()
        perf_diff_1 += tc12- tc11

    #trump_class() verify input
    card_oindexs = np.array([3,9,10,16,27,29,34,37,40,52])
    player_suits = np.array([CardSuits.HEARTS, CardSuits.HEARTS, CardSuits.BLACKJOKER, CardSuits.DIAMONDS, CardSuits.SPADES, 
                             CardSuits.CLUBS, CardSuits.CLUBS, CardSuits.REDJOKER,CardSuits.DIAMONDS, CardSuits.SPADES])
    ydl = full_poker.trump_class(card_oindexs, player_suits)
    
    perf_diff_1 = 0
    perf_diff_2 = 0
    perf_diff_3 = 0
    n = 100
    for i in range(50000):
        full_poker = FullPokers(n)
        tc11 = time.time()
        full_poker.cards_shuffle4()
        tc12 = time.time()
        perf_diff_1 += tc12- tc11

    for i in range(50000):
        full_poker = FullPokers(n)
        tc21 = time.time()
        full_poker.cards_shuffle3()
        tc22 = time.time()
        perf_diff_2 += tc22- tc21

    for i in range(50000):
        full_poker = FullPokers(n)
        tc31 = time.time()
        full_poker.cards_shuffle()
        tc32 = time.time()
        perf_diff_3 += tc32- tc31

    print("random int16, float, uint8 ", perf_diff_1, perf_diff_2, perf_diff_3)

#test_example()
    

'''
np.random.shuffle(arr)
np.random.permutation([1, 4, 9, 12, 15])
'''
